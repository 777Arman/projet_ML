"""
task3.py

Task 3 — SVM Kernel Comparison

Compares:
    1. Linear SVM on original ResNet features (reused from Task 1)
    2. Approximate RBF SVM via RBFSampler + LinearSVC, with cross-validation
       on gamma ∈ {0.0001, 0.001, 0.01}

Usage:
    python task3.py --data_dir ./coco_filtered
"""

# Fix for sklearn/joblib/openmp threading issue on Windows with Python 3.14
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from utils.eval_detection import evaluate_detection

LABEL_NAMES = {0: "background", 1: "person", 2: "car", 3: "truck"}
CLASS_NAMES  = ["background", "person", "car", "truck"]
FOREGROUND   = [1, 2, 3]

GAMMA_VALUES  = [0.0001, 0.001, 0.01]
N_COMPONENTS  = 1000   # RBFSampler output dimensionality
CV_FOLDS      = 3
CV_SUBSET     = 30000  # use a stratified subset for CV to keep it tractable


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


def print_classification_report(y_true, y_pred, model_name):
    print(f"\nClassification Report — {model_name}")
    print(classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=3,
        zero_division=0,
    ))
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


def select_gamma_cv(X_train, y_train):
    """
    Select the best RBF gamma via 3-fold cross-validation on a stratified
    subset of the training set (for tractability).

    Returns the best gamma and a dict of {gamma: mean_f1}.
    """
    print(f"\nCross-validation to select gamma (subset={CV_SUBSET}, folds={CV_FOLDS})...")
    print(f"  Gamma values tested: {GAMMA_VALUES}")

    # Stratified subset to keep class proportions
    rng = np.random.RandomState(42)
    indices = np.arange(len(y_train))
    classes, counts = np.unique(y_train, return_counts=True)
    subset_idx = []
    for cls, cnt in zip(classes, counts):
        cls_idx = indices[y_train == cls]
        n = min(cnt, int(CV_SUBSET * cnt / len(y_train)))
        subset_idx.append(rng.choice(cls_idx, n, replace=False))
    subset_idx = np.concatenate(subset_idx)
    rng.shuffle(subset_idx)

    X_sub = X_train[subset_idx]
    y_sub = y_train[subset_idx]
    print(f"  Subset size: {len(y_sub)} samples")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    gamma_scores = {}

    for gamma in GAMMA_VALUES:
        pipe = Pipeline([
            ("rbf", RBFSampler(gamma=gamma, n_components=N_COMPONENTS, random_state=42)),
            ("svm", SGDClassifier(loss="hinge", max_iter=1000, random_state=42,
                                  tol=1e-3, n_jobs=1)),
        ])
        # score on foreground-only macro F1
        scores = cross_val_score(pipe, X_sub, y_sub,
                                 cv=cv, scoring="f1_macro", n_jobs=1)
        gamma_scores[gamma] = float(scores.mean())
        print(f"  gamma={gamma:.4f}  →  mean macro F1 = {scores.mean():.4f} "
              f"(± {scores.std():.4f})")

    best_gamma = max(gamma_scores, key=gamma_scores.get)
    print(f"\n  Best gamma: {best_gamma}  (F1={gamma_scores[best_gamma]:.4f})")
    return best_gamma, gamma_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./coco_filtered")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path("outputs/task3")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TASK 3 - SVM Kernel Comparison")
    print("=" * 70)

    print("\nLoading features...")
    X_train, y_train, _,         _       = load_features(data_dir / "features_train.npz")
    X_val,   y_val,   boxes_val, ids_val = load_features(data_dir / "features_val.npz")
    ground_truth_df = build_ground_truth_df(
        data_dir / "regions.csv", image_ids=set(ids_val.tolist())
    )
    print(f"  Train: {X_train.shape[0]} regions | Val: {X_val.shape[0]} regions")

    results = {}

    # ------------------------------------------------------------------ #
    # Model 1: Linear SVM (reuse Task 1 result if available)             #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("Model 1: Linear SVM")
    print("=" * 70)

    task1_path = Path("outputs/task1/results.json")
    if task1_path.exists():
        with open(task1_path) as f:
            task1_results = json.load(f)
        print("  Reusing Task 1 SVM results (already trained on same data).")
        results["Linear SVM"] = task1_results["SVM"]
        print(f"  Foreground macro F1 : {task1_results['SVM']['fg_macro_f1']:.4f}")
        print(f"  mAP@0.5             : {task1_results['SVM']['map']:.4f}")
        # Load predictions to rebuild y_pred for confusion matrix
        pred_path = Path("outputs/task1/predictions_svm.csv")
        if pred_path.exists():
            preds_t1 = pd.read_csv(pred_path)
            y_pred_linear = preds_t1["predicted_label"].values
        else:
            y_pred_linear = None
    else:
        print("  Task 1 results not found — retraining Linear SVM...")
        svm_linear = SGDClassifier(loss="hinge", max_iter=2000, random_state=42,
                                   tol=1e-3, n_jobs=1)
        svm_linear.fit(X_train, y_train)
        y_pred_linear = svm_linear.predict(X_val)
        scores_linear = svm_linear.decision_function(X_val)
        conf_linear   = scores_linear[np.arange(len(X_val)), y_pred_linear]

        fg_macro_f1 = print_classification_report(y_val, y_pred_linear, "Linear SVM")
        preds_df = build_predictions_df(ids_val, boxes_val, y_pred_linear, conf_linear, y_val)
        preds_df.to_csv(output_dir / "predictions_linear_svm.csv", index=False)
        det = evaluate_detection(preds_df, ground_truth_df, verbose=True)
        pc  = det["map_per_class"]
        results["Linear SVM"] = {
            "fg_macro_f1": round(fg_macro_f1, 4),
            "map":         round(det["map"], 4),
            "map_per_class": {
                "person": round(pc[0], 4) if len(pc) > 0 else float("nan"),
                "car":    round(pc[1], 4) if len(pc) > 1 else float("nan"),
                "truck":  round(pc[2], 4) if len(pc) > 2 else float("nan"),
            }
        }

    # Plot confusion matrix for linear SVM
    if y_pred_linear is not None:
        cm_linear = confusion_matrix(y_val, y_pred_linear, labels=[0, 1, 2, 3])
        plot_confusion_matrix(
            cm_linear, "Confusion Matrix — Linear SVM",
            output_dir / "cm_linear_svm.png"
        )

    # ------------------------------------------------------------------ #
    # Model 2: Approximate RBF SVM (RBFSampler + LinearSVC)             #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("Model 2: Approximate RBF SVM (RBFSampler + LinearSVC)")
    print("=" * 70)
    print(f"  n_components = {N_COMPONENTS}")

    best_gamma, gamma_scores = select_gamma_cv(X_train, y_train)
    results["gamma_cv"] = {"scores": gamma_scores, "best_gamma": best_gamma}

    print(f"\nTraining RBF SVM on full training set (gamma={best_gamma})...")
    rbf_sampler = RBFSampler(gamma=best_gamma, n_components=N_COMPONENTS, random_state=42)
    print("  Fitting RBFSampler and transforming train set...")
    X_train_rbf = rbf_sampler.fit_transform(X_train)
    print("  Transforming val set...")
    X_val_rbf   = rbf_sampler.transform(X_val)

    svm_rbf = SGDClassifier(loss="hinge", max_iter=2000, random_state=42,
                             tol=1e-3, n_jobs=1)
    print("  Training LinearSVC on RBF-transformed features...")
    svm_rbf.fit(X_train_rbf, y_train)
    print("  Training done.")

    y_pred_rbf = svm_rbf.predict(X_val_rbf)
    scores_rbf = svm_rbf.decision_function(X_val_rbf)
    conf_rbf   = scores_rbf[np.arange(len(X_val_rbf)), y_pred_rbf]

    fg_macro_f1_rbf = print_classification_report(y_val, y_pred_rbf, "Approximate RBF SVM")

    cm_rbf = confusion_matrix(y_val, y_pred_rbf, labels=[0, 1, 2, 3])
    plot_confusion_matrix(
        cm_rbf, f"Confusion Matrix — Approximate RBF SVM (γ={best_gamma})",
        output_dir / "cm_rbf_svm.png"
    )

    preds_df_rbf = build_predictions_df(ids_val, boxes_val, y_pred_rbf, conf_rbf, y_val)
    preds_df_rbf.to_csv(output_dir / "predictions_rbf_svm.csv", index=False)
    print("  Predictions saved to predictions_rbf_svm.csv")

    print(f"\n  mAP evaluation for Approximate RBF SVM:")
    det_rbf = evaluate_detection(preds_df_rbf, ground_truth_df, verbose=True)
    pc_rbf  = det_rbf["map_per_class"]
    results["Approximate RBF SVM"] = {
        "best_gamma":   best_gamma,
        "fg_macro_f1":  round(fg_macro_f1_rbf, 4),
        "map":          round(det_rbf["map"], 4),
        "map_per_class": {
            "person": round(pc_rbf[0], 4) if len(pc_rbf) > 0 else float("nan"),
            "car":    round(pc_rbf[1], 4) if len(pc_rbf) > 1 else float("nan"),
            "truck":  round(pc_rbf[2], 4) if len(pc_rbf) > 2 else float("nan"),
        }
    }

    # ------------------------------------------------------------------ #
    # Comparison table                                                   #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"\n{'Model':<35} {'F1 macro (fg)':>14} {'mAP@0.5':>10}")
    print("-" * 62)
    for name in ["Linear SVM", "Approximate RBF SVM"]:
        r = results[name]
        gamma_str = f"  (γ={r['best_gamma']})" if "best_gamma" in r else ""
        print(f"  {name + gamma_str:<33} {r['fg_macro_f1']:>14.4f} {r['map']:>10.4f}")

    print("\n  Per-class mAP:")
    print(f"  {'Model':<35} {'person':>8} {'car':>8} {'truck':>8}")
    print("  " + "-" * 62)
    for name in ["Linear SVM", "Approximate RBF SVM"]:
        r = results[name]
        pc = r["map_per_class"]
        print(f"  {name:<35} {pc['person']:>8.4f} {pc['car']:>8.4f} {pc['truck']:>8.4f}")

    print(f"\n  Selected gamma: {best_gamma}  (via {CV_FOLDS}-fold CV, "
          f"subset={CV_SUBSET})")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to outputs/task3/results.json")

    print("\n" + "=" * 70)
    print("✓ TASK 3 COMPLETE - All results saved to outputs/task3/")
    print("=" * 70)


if __name__ == "__main__":
    main()
