"""
task4.py

Task 4 — Decision Tree Depth Analysis

Trains Decision Trees with max_depth ∈ {1, 2, 3, 5, 8, 12, 20, None} and
plots the bias-variance tradeoff curve (train F1 vs val F1 as a function of depth).

Usage:
    python task4.py --data_dir ./coco_filtered
"""

import os
os.environ["OMP_NUM_THREADS"]     = "1"
os.environ["MKL_NUM_THREADS"]     = "1"
os.environ["OPENBLAS_NUM_THREADS"]= "1"
os.environ["LOKY_MAX_CPU_COUNT"]  = "4"

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from utils.eval_detection import evaluate_detection

LABEL_NAMES = {0: "background", 1: "person", 2: "car", 3: "truck"}
CLASS_NAMES  = ["background", "person", "car", "truck"]
FOREGROUND   = [1, 2, 3]

DEPTHS = [1, 2, 3, 5, 8, 12, 20, None]


def load_features(path):
    data = np.load(path)
    return (
        data["features"],
        data["labels"],
        data["boxes"],
        data["image_ids"],
    )


def plot_confusion_matrix(cm, title, output_path):
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


def fg_macro_f1(y_true, y_pred):
    """Foreground-only macro F1 (person, car, truck)."""
    mask = np.isin(y_true, FOREGROUND)
    return f1_score(y_true[mask], y_pred[mask], labels=FOREGROUND,
                    average="macro", zero_division=0)


def build_predictions_df(image_ids, boxes, y_pred, confidence, y_true):
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
    output_dir = Path("outputs/task4")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TASK 4 - Decision Tree Depth Analysis")
    print("=" * 70)

    print("\nLoading features...")
    X_train, y_train, _,         _       = load_features(data_dir / "features_train.npz")
    X_val,   y_val,   boxes_val, ids_val = load_features(data_dir / "features_val.npz")
    ground_truth_df = build_ground_truth_df(
        data_dir / "regions.csv", image_ids=set(ids_val.tolist())
    )
    print(f"  Train: {X_train.shape[0]} regions | Val: {X_val.shape[0]} regions")

    # ------------------------------------------------------------------ #
    # Train Decision Trees for each depth                                #
    # ------------------------------------------------------------------ #
    records       = []  # for the results table and plot
    best_val_f1   = -1
    best_depth    = None
    best_y_pred   = None

    print(f"\nTraining Decision Trees for depths: {DEPTHS}\n")
    print(f"  {'depth':<8} {'train F1':>10} {'val F1':>10} {'n_leaves':>10}")
    print("  " + "-" * 42)

    for depth in DEPTHS:
        label = str(depth) if depth is not None else "None"

        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)

        y_pred_train = tree.predict(X_train)
        y_pred_val   = tree.predict(X_val)

        train_f1 = fg_macro_f1(y_train, y_pred_train)
        val_f1   = fg_macro_f1(y_val,   y_pred_val)
        n_leaves = tree.get_n_leaves()

        print(f"  {label:<8} {train_f1:>10.4f} {val_f1:>10.4f} {n_leaves:>10}")

        records.append({
            "depth":    depth,
            "label":    label,
            "train_f1": round(train_f1, 4),
            "val_f1":   round(val_f1,   4),
            "n_leaves": int(n_leaves),
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_depth  = depth
            best_y_pred = y_pred_val
            best_tree   = tree

    # ------------------------------------------------------------------ #
    # Bias-variance tradeoff plot                                        #
    # ------------------------------------------------------------------ #
    x_labels  = [r["label"] for r in records]
    x_pos     = list(range(len(records)))
    train_f1s = [r["train_f1"] for r in records]
    val_f1s   = [r["val_f1"]   for r in records]
    best_idx  = next(i for i, r in enumerate(records) if r["depth"] == best_depth)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_pos, train_f1s, "o-", color="steelblue",  label="Train F1 (foreground macro)")
    ax.plot(x_pos, val_f1s,   "s-", color="tomato",     label="Val F1 (foreground macro)")
    ax.axvline(best_idx, color="green", linestyle="--", alpha=0.7,
               label=f"Best val depth = {best_depth}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("max_depth  (None = fully grown)")
    ax.set_ylabel("Foreground macro F1")
    ax.set_title("Decision Tree — Bias-Variance Tradeoff by Tree Depth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / "bias_variance_tradeoff.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  Saved: {plot_path}")

    # ------------------------------------------------------------------ #
    # Full evaluation at best depth                                      #
    # ------------------------------------------------------------------ #
    best_label = str(best_depth) if best_depth is not None else "None"
    print(f"\n{'='*70}")
    print(f"Best depth: {best_label}  (val F1 = {best_val_f1:.4f})")
    print("=" * 70)

    print(f"\nClassification Report — Decision Tree (depth={best_label})")
    print(classification_report(y_val, best_y_pred,
                                 target_names=CLASS_NAMES, digits=3, zero_division=0))

    cm_best = confusion_matrix(y_val, best_y_pred, labels=[0, 1, 2, 3])
    plot_confusion_matrix(
        cm_best,
        f"Confusion Matrix — Decision Tree (depth={best_label})",
        output_dir / f"cm_best_depth_{best_label}.png"
    )

    # mAP at best depth
    probas_best = best_tree.predict_proba(X_val)
    conf_best   = probas_best[np.arange(len(X_val)), best_y_pred]
    preds_df    = build_predictions_df(ids_val, boxes_val, best_y_pred, conf_best, y_val)
    preds_df.to_csv(output_dir / f"predictions_depth_{best_label}.csv", index=False)
    print(f"\n  mAP evaluation for Decision Tree (depth={best_label}):")
    det = evaluate_detection(preds_df, ground_truth_df, verbose=True)
    pc  = det["map_per_class"]

    # ------------------------------------------------------------------ #
    # Comparison table: best tree vs best SVM                           #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("COMPARISON: Best Decision Tree vs Best SVM (Task 1)")
    print("=" * 70)

    svm_f1  = None
    svm_map = None
    task1_path = Path("outputs/task1/results.json")
    if task1_path.exists():
        with open(task1_path) as f:
            t1 = json.load(f)
        svm_f1  = t1["SVM"]["fg_macro_f1"]
        svm_map = t1["SVM"]["map"]

    print(f"\n  {'Model':<35} {'F1 macro (fg)':>14} {'mAP@0.5':>10}")
    print("  " + "-" * 62)
    print(f"  {'Decision Tree (depth=' + best_label + ')':<35} {best_val_f1:>14.4f} "
          f"{det['map']:>10.4f}")
    if svm_f1 is not None:
        print(f"  {'Linear SVM (Task 1)':<35} {svm_f1:>14.4f} {svm_map:>10.4f}")

    # ------------------------------------------------------------------ #
    # Save results                                                       #
    # ------------------------------------------------------------------ #
    results = {
        "depths": records,
        "best_depth": best_label,
        "best_val_f1": round(best_val_f1, 4),
        "best_depth_map": round(det["map"], 4),
        "best_depth_map_per_class": {
            "person": round(pc[0], 4) if len(pc) > 0 else float("nan"),
            "car":    round(pc[1], 4) if len(pc) > 1 else float("nan"),
            "truck":  round(pc[2], 4) if len(pc) > 2 else float("nan"),
        },
        "svm_comparison": {
            "fg_macro_f1": svm_f1,
            "map": svm_map,
        } if svm_f1 is not None else {}
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to outputs/task4/results.json")

    print("\n" + "=" * 70)
    print("✓ TASK 4 COMPLETE - All results saved to outputs/task4/")
    print("=" * 70)


if __name__ == "__main__":
    main()
