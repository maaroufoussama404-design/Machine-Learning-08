# Lab 4 - Data Pipeline and Quality

## Overview
Reproducible machine-learning pipeline using the UCI Adult Census Income dataset from OpenML.

## Requirements
- Python 3.12
- Dependencies listed in requirements.txt
- Internet connection for the initial dataset download

## Run
Run from the project root:

    .\.venv\Scripts\python.exe pipeline.py --config config.yaml

## Processing
- Dataset schema, missing-value and range validation
- Stratified 70/30 train/test split
- Training-only imputation, clipping, scaling and encoding
- Duplicate removal on training data only
- Engineered features: capital_net and hours_per_age
- Logistic Regression evaluated using accuracy and F1

## Results
See the results directory for validation reports, label-quality audit, manual review, model comparison, fitted preprocessor and execution summary.

## Reproducibility
Random seed: 42. Processed dataset SHA256 is stored in config.yaml.

## Manual review limitation
The 50-row review checks documented data constraints and category consistency. It does not independently verify the true income labels.
