import argparse
import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC = [
    "age", "fnlwgt", "education-num", "capital-gain",
    "capital-loss", "hours-per-week"
]

CATEGORICAL = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

FEATURES = NUMERIC + CATEGORICAL
COLUMNS = FEATURES + ["income"]

RANGES = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for col in CATEGORICAL:
            X[col] = X[col].replace("?", np.nan)

        if self.enabled:
            X["capital_net"] = X["capital-gain"] - X["capital-loss"]
            X["hours_per_age"] = (
                X["hours-per-week"] / (X["age"] - 16)
            )

        return X


class TrainQuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.columns_ = list(frame.columns)
        self.lower_ = frame.quantile(self.lower)
        self.upper_ = frame.quantile(self.upper)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X, columns=self.columns_)
        return frame.clip(self.lower_, self.upper_, axis=1)


def validate_data(raw):
    report = {
        "rows": int(len(raw)),
        "columns": int(len(raw.columns)),
        "expected_shape": [48842, 15],
        "columns_match": list(raw.columns) == COLUMNS,
        "numeric_type_violations": {},
        "range_violations": {},
        "question_mark_counts": {},
        "null_counts": {
            col: int(count)
            for col, count in raw.isna().sum().items()
        },
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "unknown_income_labels": [],
    }

    if list(raw.columns) != COLUMNS:
        raise ValueError(
            f"Unexpected columns. Expected {COLUMNS}; got {list(raw.columns)}"
        )

    typed = raw.copy()

    for col in NUMERIC:
        values = pd.to_numeric(raw[col], errors="coerce")
        invalid = raw[col].notna() & (
            values.isna() | (values % 1 != 0)
        )

        report["numeric_type_violations"][col] = int(invalid.sum())

        if invalid.any() or values.isna().any():
            typed[col] = values
        else:
            typed[col] = values.astype("int64")

    for col in CATEGORICAL + ["income"]:
        report["question_mark_counts"][col] = int(
            raw[col].astype("string").eq("?").sum()
        )
        typed[col] = raw[col].astype("category")

    for col, (low, high) in RANGES.items():
        values = typed[col]
        invalid = values.notna() & ~values.between(low, high)

        report["range_violations"][col] = {
            "count": int(invalid.sum()),
            "allowed": [low, high],
            "observed_min": int(values.min()),
            "observed_max": int(values.max()),
        }

    labels = set(typed["income"].astype(str).unique())
    valid_labels = {">50K", "<=50K", ">50K.", "<=50K."}
    report["unknown_income_labels"] = sorted(labels - valid_labels)

    if report["unknown_income_labels"]:
        raise ValueError(
            f"Unexpected target labels: {report['unknown_income_labels']}"
        )

    return typed, report


def duplicate_conflict_audit(data):
    grouped = data.groupby(
        FEATURES, dropna=False, observed=True
    )["income"].nunique(dropna=False)

    conflicts = grouped[grouped > 1]

    row_conflicts = (
        data.groupby(FEATURES, dropna=False, observed=True)["income"]
        .transform("nunique")
        .gt(1)
    )

    return {
        "conflicting_feature_groups": int(len(conflicts)),
        "rows_in_conflicting_groups": int(row_conflicts.sum()),
        "definition": (
            "Identical values across all 14 features but different income labels."
        ),
    }


def build_preprocessor(engineered):
    numeric = NUMERIC.copy()

    if engineered:
        numeric += ["capital_net", "hours_per_age"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", TrainQuantileClipper(0.01, 0.99)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    columns = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, CATEGORICAL),
    ])

    return Pipeline([
        ("features", FeatureEngineer(enabled=engineered)),
        ("transform", columns),
    ])


def calculate_hash(matrices, labels):
    digest = hashlib.sha256()

    for name, matrix in zip(
        ["train", "test"], matrices
    ):
        digest.update(name.encode())

        if sparse.issparse(matrix):
            matrix = matrix.tocsr()
            digest.update(str(matrix.shape).encode())
            digest.update(matrix.indptr.tobytes())
            digest.update(matrix.indices.tobytes())
            digest.update(matrix.data.tobytes())
        else:
            digest.update(np.ascontiguousarray(matrix).tobytes())

    for name, values in zip(["y_train", "y_test"], labels):
        digest.update(name.encode())
        digest.update(
            "\n".join(map(str, values)).encode("utf-8")
        )

    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config_path = root / args.config

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results_dir = root / "results"
    data_dir = root / "data"
    results_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                results_dir / "pipeline.log",
                mode="w",
                encoding="utf-8"
            ),
        ],
    )
    log = logging.getLogger("lab4")

    # 1. INGESTION: retain the raw values unchanged.
    log.info("Fetching Adult dataset from OpenML.")

    dataset = fetch_openml(
        name="adult",
        version=2,
        as_frame=True
    )

    raw = dataset.data.copy()
    target = dataset.target

    if isinstance(target, pd.DataFrame):
        target = target.iloc[:, 0]

    raw["income"] = target
    raw = raw[COLUMNS].copy()

    log.info(
        "INGESTION: %d rows, %d columns",
        raw.shape[0], raw.shape[1]
    )

    raw.to_csv(data_dir / "adult_raw.csv", index=False)

    if raw.shape != (48842, 15):
        raise ValueError(
            f"Expected 48842 x 15; received {raw.shape}"
        )

    # 2. VALIDATION.
    typed, validation = validate_data(raw)

    (results_dir / "validation_report.json").write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8"
    )

    log.info("Validation report saved.")
    log.info("Missing '?' counts: %s", validation["question_mark_counts"])
    log.info("Range checks: %s", validation["range_violations"])

    if any(validation["numeric_type_violations"].values()):
        raise ValueError("Numeric type violations found. See validation_report.json.")

    if any(item["count"] for item in validation["range_violations"].values()):
        raise ValueError("Range violations found. See validation_report.json.")

    if validation["null_counts"]["income"] > 0:
        raise ValueError("Missing target labels found. See validation_report.json.")

    # 3. LABEL QUALITY.
    audit = duplicate_conflict_audit(typed)

    (results_dir / "label_quality_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8"
    )

    # Reproducible sample; human inspection is still required.
    review = raw.sample(n=50, random_state=config["random_seed"]).copy()
    review.insert(0, "review_id", range(1, 51))
    review["manual_review_status"] = "PENDING"
    review["reviewer_notes"] = ""

    review.to_csv(
        results_dir / "manual_review_sample.csv",
        index=False
    )

    # 4. SPLIT BEFORE CLEANING OR FITTING TRANSFORMATIONS.
    X = typed[FEATURES].copy()
    y = typed["income"].astype("string").str.rstrip(".")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=config["random_seed"],
        stratify=y
    )

    # Remove exact duplicate feature + label records from training only.
    training = X_train.copy()
    training["income"] = y_train

    duplicate_mask = training.duplicated(keep="first")
    removed = int(duplicate_mask.sum())

    X_train = X_train.loc[~duplicate_mask].copy()
    y_train = y_train.loc[~duplicate_mask].copy()

    log.info("Training duplicates removed: %d", removed)
    log.info("Test rows retained unchanged: %d", len(X_test))

    # 5. BASELINES: independently fit both pipelines on training only.
    comparison = []
    fitted_preprocessor = None
    hash_matrices = None

    for engineered in [False, True]:
        label = (
            "with_engineered_features"
            if engineered else "without_engineered_features"
        )

        preprocessing = build_preprocessor(engineered)

        model = Pipeline([
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(
                max_iter=2000,
                random_state=config["random_seed"]
            )),
        ])

        log.info("Training Logistic Regression: %s", label)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, pos_label=">50K")

        comparison.append({
            "model": "LogisticRegression",
            "features": label,
            "accuracy": float(accuracy),
            "f1": float(f1),
        })

        log.info(
            "%s: accuracy=%.4f, F1=%.4f",
            label, accuracy, f1
        )

        if engineered:
            joblib.dump(model, results_dir / "baseline_model.joblib")
            fitted_preprocessor = model.named_steps["preprocessing"]

            transformed_train = fitted_preprocessor.transform(X_train)
            transformed_test = fitted_preprocessor.transform(X_test)

            hash_matrices = [transformed_train, transformed_test]

    pd.DataFrame(comparison).to_csv(
        results_dir / "baseline_comparison.csv",
        index=False
    )

    # Save the fitted transformer (not the classifier).
    joblib.dump(
        fitted_preprocessor,
        results_dir / "preprocessor.joblib"
    )

    # 6. HASH the processed matrices and labels.
    sha256 = calculate_hash(
        hash_matrices,
        [y_train, y_test]
    )

    config["reproducibility"] = {
        "random_seed": config["random_seed"],
        "processed_dataset_sha256": sha256,
        "hash_method": (
            "SHA256 over engineered preprocessing outputs for train and test "
            "in CSR form, plus train and test labels."
        ),
        "raw_rows": int(len(raw)),
        "raw_columns": int(len(raw.columns)),
        "train_rows_after_deduplication": int(len(X_train)),
        "test_rows_untouched": int(len(X_test)),
        "training_duplicates_removed": removed,
    }

    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8"
    )

    summary = {
        "validation": validation,
        "label_quality_audit": audit,
        "training_duplicates_removed": removed,
        "test_rows_untouched": int(len(X_test)),
        "baseline_results": comparison,
        "processed_dataset_sha256": sha256,
        "manual_review_status": "PENDING",
    }

    (results_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n========== LAB 4 RESULTS ==========")
    print("Raw dataset:", raw.shape)
    print("Training duplicates removed:", removed)
    print("Conflicting feature groups:", audit["conflicting_feature_groups"])

    for result in comparison:
        print(
            result["features"],
            "| Accuracy:", round(result["accuracy"], 4),
            "| F1:", round(result["f1"], 4)
        )

    print("SHA256:", sha256)
    print("Manual review: results/manual_review_sample.csv")
    print("Pipeline completed.")


if __name__ == "__main__":
    main()


