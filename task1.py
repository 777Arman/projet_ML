"""
task1.py

Task 1 — Baseline Classification (SVM and Decision Tree)

Important:
    1. Read the project description carefully before starting.
    2. Complete the section marked with TODO.
    3. Do not modify the loading or evaluation code.

Usage:
    python task1.py --data_dir ./coco_filtered
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

from utils.eval_detection import evaluate_detection

LABEL_NAMES = {0: "background", 1: "person", 2: "car", 3: "truck"}
CLASS_NAMES = ["background", "person", "car", "truck"]
FOREGROUND  = [1, 2, 3]


def load_features(path):
    """Load features, labels, boxes, and image_ids from a .npz file."""
    data = np.load(path)
    return (
        data["features"],   # (N, 2048) float32
        data["labels"],     # (N,)      int64
        data["boxes"],      # (N, 4)    float32  [x1, y1, x2, y2]
        data["image_ids"],  # (N,)      int64
    )


def plot_confusion_matrix(cm, title, output_path):
    """Plot and save a confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def print_classification_report(y_true, y_pred, model_name):
    """
    Print per-class precision, recall, F1, support for all four classes,
    and foreground-only macro F1 as the summary number.
    """
    print(f"\nClassification Report — {model_name}")
    print(classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=3,
        zero_division=0,
    ))
    # foreground-only macro F1 — this is the number to compare across tasks
    fg_mask = np.isin(y_true, FOREGROUND)
    fg_report = classification_report(
        y_true[fg_mask], y_pred[fg_mask],
        labels=FOREGROUND,
        target_names=["person", "car", "truck"],
        digits=3,
        zero_division=0,
        output_dict=True,
    )
    fg_macro_f1 = fg_report["macro avg"]["f1-score"]
    print(f"  Foreground-only macro F1 : {fg_macro_f1:.3f}\n")
    return fg_macro_f1


def build_predictions_df(image_ids, boxes, y_pred, confidence, y_true):
    """Build the predictions dataframe used by evaluate_detection."""
    return pd.DataFrame({
        "image_id":        image_ids,
        "x1":              boxes[:, 0],
        "y1":              boxes[:, 1],
        "x2":              boxes[:, 2],
        "y2":              boxes[:, 3],
        "predicted_label": y_pred,
        "confidence":      confidence,
        "true_label":      y_true,
    })


def build_ground_truth_df(regions_csv, image_ids=None):
    """
    Load ground truth boxes from regions.csv (foreground only).
    Pass image_ids to restrict to a specific set of images — always pass
    ids_val here to avoid comparing predictions against images not in the
    validation set.
    """
    label_map = {"person": 1, "car": 2, "truck": 3}
    df = pd.read_csv(regions_csv)
    df["class_label"] = df["class_label"].map(label_map)
    df = df.dropna(subset=["class_label"])
    df["class_label"] = df["class_label"].astype(int)
    df = df[["image_id", "x1", "y1", "x2", "y2", "class_label"]]
    if image_ids is not None:
        df = df[df["image_id"].isin(image_ids)]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./coco_filtered")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path("outputs/task1")
    output_dir.mkdir(parents=True, exist_ok=True)

    # load features
    # boxes and image_ids are not used for training but are needed to build
    # the predictions dataframe for mAP evaluation
    print("Loading features...")
    X_train, y_train, _,          _       = load_features(data_dir / "features_train.npz")
    X_val,   y_val,   boxes_val,  ids_val = load_features(data_dir / "features_val.npz")
    ground_truth_df = build_ground_truth_df(data_dir / "regions.csv", image_ids=set(ids_val.tolist()))


    print(f"  Train: {X_train.shape[0]} regions | Val: {X_val.shape[0]} regions")

    for split_name, y in [("train", y_train), ("val", y_val)]:
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n  {split_name} label distribution:")
        for u, c in zip(unique, counts):
            print(f"    {LABEL_NAMES[u]:<12}: {c:>8} ({100*c/len(y):.1f}%)")

    # ------------------------------------------------------------------ #
    # T O D O: train your classifiers here                                   #
    # ------------------------------------------------------------------ #
    #
    # (a) Multiclass SVM with a linear kernel:
    #
    #   from sklearn.svm import LinearSVC
    #   svm = LinearSVC(max_iter=2000, random_state=42)
    #   svm.fit(X_train, y_train)
    #
    #   predictions and confidence scores:
    #   y_pred_svm = svm.predict(X_val)
    #   scores_svm = svm.decision_function(X_val)   # shape (N, n_classes)
    #   conf_svm   = scores_svm[range(len(X_val)), y_pred_svm]
    #
    # (b) Decision Tree with a fixed maximum depth:
    #
    #   from sklearn.tree import DecisionTreeClassifier
    #   tree = DecisionTreeClassifier(max_depth=10, random_state=42)
    #   tree.fit(X_train, y_train)
    #
    #   predictions and confidence scores:
    #   y_pred_tree = tree.predict(X_val)
    #   probas_tree = tree.predict_proba(X_val)     # shape (N, n_classes)
    #   conf_tree   = probas_tree[range(len(X_val)), y_pred_tree]
    #
    # Assign your results to the variables below.

    from sklearn.svm import LinearSVC
    svm = LinearSVC(max_iter=2000, random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_val)
    scores_svm = svm.decision_function(X_val)
    conf_svm   = scores_svm[np.arange(len(X_val)), y_pred_svm]

    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(max_depth=10, random_state=42)
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_val)
    probas_tree = tree.predict_proba(X_val)
    conf_tree   = probas_tree[np.arange(len(X_val)), y_pred_tree]

    # ------------------------------------------------------------------ #
    # Evaluation — do not modify below this line                         #
    # ------------------------------------------------------------------ #
    assert svm  is not None, "Train your SVM before running evaluation."
    assert tree is not None, "Train your Decision Tree before running evaluation."

    results = {}

    for model_name, y_pred, confidence in [
        ("SVM",           y_pred_svm,  conf_svm),
        ("Decision Tree", y_pred_tree, conf_tree),
    ]:
        print(f"\nEvaluating {model_name}...")

        # classification report and foreground-only macro F1 (before NMS)
        fg_macro_f1 = print_classification_report(y_val, y_pred, model_name)

        # confusion matrix (before NMS)
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2, 3])
        plot_confusion_matrix(
            cm,
            title=f"Confusion Matrix — {model_name}",
            output_path=output_dir / f"cm_{model_name.lower().replace(' ', '_')}.png"
        )

        # save predictions dataframe for inspection and mAP evaluation
        preds_df = build_predictions_df(
            ids_val, boxes_val, y_pred, confidence, y_val
        )
        csv_name = f"predictions_{model_name.lower().replace(' ', '_')}.csv"
        preds_df.to_csv(output_dir / csv_name, index=False)
        print(f"  Predictions saved to outputs/task1/{csv_name}")

        # mAP evaluation (after NMS, using provided utility)
        print(f"\n  mAP evaluation for {model_name}:")
        detection_results = evaluate_detection(preds_df, ground_truth_df, verbose=True)

        # store results for compare.py
        per_class = detection_results["map_per_class"]
        results[model_name] = {
            "fg_macro_f1": round(fg_macro_f1, 4),
            "map":         round(detection_results["map"], 4),
            "map_per_class": {
                "person": round(per_class[0], 4) if len(per_class) > 0 else float("nan"),
                "car":    round(per_class[1], 4) if len(per_class) > 1 else float("nan"),
                "truck":  round(per_class[2], 4) if len(per_class) > 2 else float("nan"),
            }
        }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to outputs/task1/results.json")
    print("\nTask 1 complete. Check outputs/task1/ for results.")


if __name__ == "__main__":
    main()

# corrected