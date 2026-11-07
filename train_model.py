"""
Train an IT support ticket text classifier.

Dataset: tasksource/it-support-tickets on Hugging Face.
The dataset has a source-provided train/test split. If that split is not
available, the script creates a stratified holdout split.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATASET_ID = "tasksource/it-support-tickets"
ARTIFACT_DIR = Path("artifacts")
RANDOM_STATE = 42

# These are suggested operational destinations created for this project.
# They are NOT labels supplied by the original dataset.
TEAM_ROUTING = {
    "Active Directory": "Identity & Access Support",
    "Computer-Services": "End-User Computing Support",
    "EOL": "IT Asset Lifecycle Support",
    "Fileservice": "File & Storage Support",
    "O365": "Collaboration & Productivity Support",
    "Software": "Application Support",
    "Support general": "General IT Service Desk",
}


def label_names_from_feature(label_feature, values):
    """Convert ClassLabel integers to their original readable class names."""
    if hasattr(label_feature, "int2str"):
        return [label_feature.int2str(int(value)) for value in values]
    return [str(value) for value in values]


def clean_frame(frame, label_feature):
    required = {"text", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {sorted(missing)}. "
            f"Available columns: {list(frame.columns)}"
        )

    frame = frame[["text", "label"]].copy()
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame = frame[frame["text"].str.len() > 0].copy()
    frame["label"] = label_names_from_feature(label_feature, frame["label"])
    return frame.reset_index(drop=True)


def make_pipeline(model):
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", model),
        ]
    )


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)

    print(f"Downloading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID)

    if "train" not in dataset:
        raise ValueError(
            "A 'train' split was not found. Inspect the dataset card and "
            "update this loader to match its current split names."
        )

    label_feature = dataset["train"].features["label"]
    train_df = clean_frame(dataset["train"].to_pandas(), label_feature)

    # Prefer the dataset's original test split when available.
    if "test" in dataset:
        test_df = clean_frame(dataset["test"].to_pandas(), label_feature)
        X_train = train_df["text"]
        y_train = train_df["label"]
        X_test = test_df["text"]
        y_test = test_df["label"]
        split_note = "Used the dataset's source-provided train/test split."
    else:
        all_df = train_df
        X_train, X_test, y_train, y_test = train_test_split(
            all_df["text"],
            all_df["label"],
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=all_df["label"],
        )
        split_note = (
            "No source test split was found; created a stratified 80/20 split."
        )

    # Make a validation split from training data to compare two baselines.
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    comparison = {}
    print("\nComparing baseline models on the validation split...")
    for name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(X_fit, y_fit)
        predictions = pipeline.predict(X_valid)
        comparison[name] = {
            "validation_accuracy": float(accuracy_score(y_valid, predictions)),
            "validation_macro_f1": float(
                f1_score(y_valid, predictions, average="macro")
            ),
        }
        print(
            f"{name}: accuracy={comparison[name]['validation_accuracy']:.3f}, "
            f"macro-F1={comparison[name]['validation_macro_f1']:.3f}"
        )

    # Logistic Regression is used for the deployed baseline because it exposes
    # class probability estimates for the dashboard. These scores are not
    # guaranteed to be calibrated probabilities.
    final_model = make_pipeline(
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    )
    final_model.fit(X_train, y_train)

    test_predictions = final_model.predict(X_test)
    class_names = sorted(y_train.unique())

    report = classification_report(
        y_test,
        test_predictions,
        labels=class_names,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "dataset_id": DATASET_ID,
        "split_note": split_note,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "classes": class_names,
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
        "test_macro_f1": float(
            f1_score(y_test, test_predictions, average="macro")
        ),
        "classification_report": report,
        "validation_model_comparison": comparison,
    }

    joblib.dump(final_model, ARTIFACT_DIR / "ticket_classifier.joblib")

    metadata = {
        "dataset_id": DATASET_ID,
        "classes": class_names,
        "team_routing": TEAM_ROUTING,
        "model_name": "TF-IDF + Logistic Regression",
        "confidence_note": (
            "The displayed score is the model's estimated class probability; "
            "it is not a guarantee that the prediction is correct."
        ),
    }
    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (ARTIFACT_DIR / "evaluation.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        test_predictions,
        labels=class_names,
        display_labels=class_names,
        xticks_rotation=45,
        values_format="d",
    )
    display.figure_.tight_layout()
    display.figure_.savefig(
        ARTIFACT_DIR / "confusion_matrix.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(display.figure_)

    print("\nFinal holdout test results")
    print(f"Accuracy: {metrics['test_accuracy']:.3f}")
    print(f"Macro F1: {metrics['test_macro_f1']:.3f}")
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"\n{split_note}")
    print(f"Artifacts saved in: {ARTIFACT_DIR.resolve()}")


if __name__ == "__main__":
    main()
