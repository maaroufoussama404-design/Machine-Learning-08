# Lab 8: Quantization and Export

## Overview
Manual quantization, dynamic and static post-training quantization, pruning and distillation combinations, and ONNX export.

## Main files
- `run_lab8.py` — experiment pipeline
- `LAB8_REPORT.md` — results and deployment recommendations
- `results/` — metrics and quantization results
- `plots/` — accuracy-latency and accuracy-size comparisons
- `checkpoints/` — model checkpoints
- `onnx_model.onnx` — exported ONNX model
- `project_milestone/lab4/` — Lab 4 pipeline, trained baseline and metrics

## Reproduction
Run `python run_lab8.py` with the required dependencies installed.
Experiments use FashionMNIST and CPU execution.
Static quantization uses the available oneDNN backend.
