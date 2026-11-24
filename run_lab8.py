import os, json, time, io, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import onnx
import onnxruntime as ort

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.ao.quantization import (
    QuantStub, DeQuantStub, get_default_qconfig,
    prepare, convert, quantize_dynamic
)

torch.manual_seed(42)
torch.set_num_threads(4)
torch.backends.quantized.engine = "onednn"

ROOT = "."
DEVICE = "cpu"
BACKEND = "onednn"

os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# --------------------------------------------------
# 1. DATA
# --------------------------------------------------

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train = datasets.FashionMNIST(
    "./data", train=True, download=False, transform=tf
)
test = datasets.FashionMNIST(
    "./data", train=False, download=False, transform=tf
)

train_loader = DataLoader(train, batch_size=128, shuffle=False)
test_loader = DataLoader(test, batch_size=256, shuffle=False)

calibration_indices = torch.randperm(len(train))[:500].tolist()
calibration_loader = DataLoader(
    Subset(train, calibration_indices),
    batch_size=50, shuffle=False
)

# --------------------------------------------------
# 2. MODEL ARCHITECTURES
# --------------------------------------------------

class TeacherCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256*3*3,256),
            nn.ReLU(), nn.Linear(256,10)
        )

    def forward(self,x):
        return self.classifier(self.features(x))


class CNN(nn.Module):
    def __init__(self, channels=(32,64,128), hidden=256):
        super().__init__()
        a,b,c = channels
        self.conv1 = nn.Conv2d(1,a,3,padding=1)
        self.conv2 = nn.Conv2d(a,b,3,padding=1)
        self.conv3 = nn.Conv2d(b,c,3,padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(c*3*3,hidden)
        self.fc2 = nn.Linear(hidden,10)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=self.pool(F.relu(self.conv3(x)))
        return self.fc2(F.relu(self.fc1(torch.flatten(x,1))))


class StaticTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.features = nn.Sequential(
            nn.Conv2d(1,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256*3*3,256),
            nn.ReLU(), nn.Linear(256,10)
        )
        self.dequant = DeQuantStub()

    def forward(self,x):
        x=self.quant(x)
        x=self.features(x)
        x=self.classifier(x)
        return self.dequant(x)


# --------------------------------------------------
# 3. CHECKPOINT LOADING
# --------------------------------------------------

def load_state(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def load_teacher():
    model = TeacherCNN()
    state = load_state("checkpoints/teacher.pt")
    model.load_state_dict(state)
    return model.eval()


def load_pruned():
    model = TeacherCNN()
    state = load_state("checkpoints/global_50.pt")

    # Fold pruning reparameterization into ordinary weights if present.
    clean = {}
    for key, value in state.items():
        if key.endswith(".weight_orig"):
            prefix = key[:-len("weight_orig")]
            mask = state.get(prefix + "weight_mask")
            clean[prefix + "weight"] = value * mask if mask is not None else value
        elif key.endswith(".weight_mask"):
            continue
        else:
            clean[key] = value

    model.load_state_dict(clean)
    return model.eval()


def load_structured():
    model = CNN((32,64,128),256)
    model.load_state_dict(load_state("checkpoints/structured_50.pt"))
    return model.eval()


def load_student():
    model = CNN((16,32,64),128)
    model.load_state_dict(load_state("checkpoints/student_distilled.pt"))
    return model.eval()


# --------------------------------------------------
# 4. METRICS
# --------------------------------------------------

def accuracy(model):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x,y in test_loader:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100 * correct / total


def latency(model):
    model.eval()
    x = torch.randn(1,1,28,28)
    with torch.inference_mode():
        for _ in range(20):
            model(x)
        times = []
        for _ in range(100):
            start = time.perf_counter()
            model(x)
            times.append((time.perf_counter()-start)*1000)
    return float(np.median(times))


def model_size(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024*1024)


def record(name, model):
    model.eval()
    row = {
        "variant": name,
        "accuracy_percent": accuracy(model),
        "latency_ms": latency(model),
        "size_MB": model_size(model),
        "parameters": sum(p.numel() for p in model.parameters())
    }
    print(row)
    return row


# --------------------------------------------------
# 5. MANUAL QUANTIZATION
# --------------------------------------------------

teacher = load_teacher()
weight = teacher.features[0].weight.detach().clone()

manual_rows = []

for bits in [8,4,2]:
    qmax = 2**(bits-1)-1

    # Per-tensor symmetric quantization
    scale = weight.abs().max() / qmax
    q = torch.round(weight / scale).clamp(-qmax,qmax)
    reconstructed = q * scale
    mse = torch.mean((weight-reconstructed)**2).item()

    manual_rows.append({
        "bits":bits,
        "method":"per_tensor",
        "MSE":mse
    })

    # Per-output-channel symmetric quantization
    scales = weight.abs().amax(dim=(1,2,3),keepdim=True) / qmax
    scales = scales.clamp(min=1e-12)
    q_channel = torch.round(weight/scales).clamp(-qmax,qmax)
    reconstructed_channel = q_channel*scales
    mse_channel = torch.mean(
        (weight-reconstructed_channel)**2
    ).item()

    manual_rows.append({
        "bits":bits,
        "method":"per_channel",
        "MSE":mse_channel
    })

pd.DataFrame(manual_rows).to_csv(
    "results/manual_quantization.csv",index=False
)

print("\nMANUAL QUANTIZATION")
print(pd.DataFrame(manual_rows).to_string(index=False))


# --------------------------------------------------
# 6. BASELINE, PRUNED AND DISTILLED
# --------------------------------------------------

baseline = load_teacher()
pruned = load_structured()
distilled = load_student()

rows = [
    record("baseline",baseline),
    record("pruned",pruned),
    record("distilled",distilled)
]


# --------------------------------------------------
# 7. DYNAMIC QUANTIZATION
# --------------------------------------------------

dynamic = quantize_dynamic(
    copy.deepcopy(baseline),
    {nn.Linear},
    dtype=torch.qint8
).eval()

rows.append(record("dynamic_quant",dynamic))

torch.save(dynamic.state_dict(),"checkpoints/dynamic_quant.pt")


# --------------------------------------------------
# 8. STATIC POST-TRAINING QUANTIZATION
# --------------------------------------------------

static = StaticTeacher()
static.load_state_dict(load_state("checkpoints/teacher.pt"))
static.eval()

# Fuse Conv+ReLU and Linear+ReLU before calibration.
static_fused = torch.ao.quantization.fuse_modules(
    static,
    [
        ["features.0","features.1"],
        ["features.3","features.4"],
        ["features.6","features.7"],
        ["classifier.1","classifier.2"]
    ],
    inplace=False
)

static_fused.qconfig = get_default_qconfig(BACKEND)
prepare(static_fused,inplace=True)

print("\nCalibrating static quantization with 500 images...")
with torch.inference_mode():
    for x,_ in calibration_loader:
        static_fused(x)

static_quant = convert(static_fused,inplace=False).eval()

static_accuracy = accuracy(static_quant)
static_latency = latency(static_quant)
static_size = model_size(static_quant)

rows.append({
    "variant":"static_quant",
    "accuracy_percent":static_accuracy,
    "latency_ms":static_latency,
    "size_MB":static_size,
    "parameters":sum(p.numel() for p in baseline.parameters())
})

print("Static quantization:",rows[-1])
torch.save(static_quant.state_dict(),"checkpoints/static_quant.pt")


# --------------------------------------------------
# 9. COMBINED PRUNING + QUANTIZATION
# --------------------------------------------------

pruned_quant = quantize_dynamic(
    copy.deepcopy(pruned),{nn.Linear},dtype=torch.qint8
).eval()

distilled_quant = quantize_dynamic(
    copy.deepcopy(distilled),{nn.Linear},dtype=torch.qint8
).eval()

rows.append(record("pruned+quant",pruned_quant))
rows.append(record("distilled+quant",distilled_quant))

torch.save(pruned_quant.state_dict(),"checkpoints/pruned_quant.pt")
torch.save(distilled_quant.state_dict(),"checkpoints/distilled_quant.pt")


# --------------------------------------------------
# 10. ONNX EXPORT AND VALIDATION
# --------------------------------------------------

baseline.eval()
sample = torch.randn(1,1,28,28)

torch.onnx.export(
    baseline,
    sample,
    "onnx_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    dynamo=False
)

onnx_model = onnx.load("onnx_model.onnx")
onnx.checker.check_model(onnx_model)

session = ort.InferenceSession(
    "onnx_model.onnx",
    providers=["CPUExecutionProvider"]
)

with torch.inference_mode():
    torch_output = baseline(sample).numpy()

onnx_output = session.run(
    ["output"],
    {"input":sample.numpy()}
)[0]

max_error = float(np.max(np.abs(torch_output-onnx_output)))
print("\nONNX maximum absolute error:",max_error)

assert max_error < 1e-4, (
    f"ONNX numerical validation failed: {max_error}"
)

# ONNX Runtime latency
for _ in range(20):
    session.run(["output"],{"input":sample.numpy()})

onnx_times = []
for _ in range(100):
    start=time.perf_counter()
    session.run(["output"],{"input":sample.numpy()})
    onnx_times.append((time.perf_counter()-start)*1000)

onnx_latency = float(np.median(onnx_times))
onnx_size = os.path.getsize("onnx_model.onnx")/(1024*1024)

print("ONNX Runtime latency:",onnx_latency,"ms")
print("ONNX file size:",onnx_size,"MB")


# --------------------------------------------------
# 11. COMPARISON TABLE
# --------------------------------------------------

comparison = pd.DataFrame(rows)
comparison.to_csv("results/comparison.csv",index=False)
print("\nFINAL COMPARISON")
print(comparison.to_string(index=False))


# --------------------------------------------------
# 12. PLOTS
# --------------------------------------------------

plt.figure()
plt.scatter(
    comparison["latency_ms"],
    comparison["accuracy_percent"]
)
for _,r in comparison.iterrows():
    plt.annotate(
        r["variant"],
        (r["latency_ms"],r["accuracy_percent"])
    )
plt.xlabel("CPU latency (ms/image)")
plt.ylabel("Test accuracy (%)")
plt.title("Accuracy vs Latency")
plt.grid(True)
plt.savefig("plots/accuracy_latency.png",dpi=160,bbox_inches="tight")
plt.close()

plt.figure()
plt.scatter(
    comparison["size_MB"],
    comparison["accuracy_percent"]
)
for _,r in comparison.iterrows():
    plt.annotate(
        r["variant"],
        (r["size_MB"],r["accuracy_percent"])
    )
plt.xlabel("Serialized state dictionary size (MB)")
plt.ylabel("Test accuracy (%)")
plt.title("Accuracy vs Model Size")
plt.grid(True)
plt.savefig("plots/accuracy_size.png",dpi=160,bbox_inches="tight")
plt.close()


# --------------------------------------------------
# 13. METRICS AND REPORT
# --------------------------------------------------

metrics = {
    "quantization_backend":BACKEND,
    "device":"CPU",
    "manual_quantization":manual_rows,
    "comparison":rows,
    "onnx_max_absolute_error":max_error,
    "onnx_runtime_latency_ms":onnx_latency,
    "onnx_size_MB":onnx_size,
    "calibration_images":500
}

with open("results/metrics.json","w") as f:
    json.dump(metrics,f,indent=2)

report = f"""
# Lab 8: Quantization and Export

## Environment
- Device: CPU
- PyTorch: {torch.__version__}
- Quantization backend: {BACKEND}
- ONNX: {onnx.__version__}
- ONNX Runtime: {ort.__version__}

The installed PyTorch build supports only the onednn quantization
backend. FBGEMM and QNNPACK were not available in this environment.

## Manual quantization
See results/manual_quantization.csv.
Symmetric quantization uses zero-point 0 and signed integer ranges.
Per-channel scaling uses separate scales for output channels.

## Dynamic quantization
Linear layers were quantized to int8 using post-training dynamic
quantization. Weights are quantized ahead of inference; activations
are quantized dynamically during execution.

## Static quantization
Static quantization used 500 randomly selected FashionMNIST training
images for calibration. The subset must represent real deployment
inputs because activation ranges estimated during calibration
determine quantization scales and clipping thresholds.

## ONNX
- Export: onnx_model.onnx
- Maximum absolute error: {max_error:.8f}
- Required tolerance: 0.0001
- ONNX Runtime latency: {onnx_latency:.4f} ms
- ONNX size: {onnx_size:.4f} MB

## Comparison
See results/comparison.csv.

## Deployment recommendations
- Cloud: baseline or dynamic quantization, selected according to
  measured throughput and accuracy requirements.
- Edge CPU: structured-pruned or dynamically quantized model,
  after benchmarking on the actual target CPU.
- Mobile: use a mobile-supported runtime and backend; validate
  operators and latency on the target device.
- TinyML: none of these models is automatically TinyML-ready.
  Further architecture reduction, integer-only quantization,
  memory profiling, and MCU-specific validation are required.

## Limitations
The comparison uses CPU measurements on this development machine.
Model size refers to serialized state dictionaries, not peak runtime
memory. ONNX Runtime latency is measured separately and is not
directly interchangeable with PyTorch latency. Energy was not measured.
"""

with open("LAB8_REPORT.md","w",encoding="utf-8") as f:
    f.write(report)

print("\nLAB 8 EXPERIMENTS COMPLETED")
print("See LAB8_REPORT.md, results/, plots/, and onnx_model.onnx")
