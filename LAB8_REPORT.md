
# Lab 8: Quantization and Export

## Environment
- Device: CPU
- PyTorch: 2.15.0+cpu
- Quantization backend: onednn
- ONNX: 1.23.2
- ONNX Runtime: 1.32.0

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
- Maximum absolute error: 0.00000191
- Required tolerance: 0.0001
- ONNX Runtime latency: 0.3329 ms
- ONNX size: 3.6735 MB

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
