"""
task2.py

Task 2 — Class Imbalance Handling

Compares three strategies to mitigate class imbalance:
    1. Region-level augmentation (Task 2a)
    2. SMOTE on feature vectors (Task 2b)
    3. Class-weighted SVM (baseline)

Usage:
    python task2.py --data_dir ./coco_filtered
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
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

from utils.eval_detection import evaluate_detection

LABEL_NAMES = {0: "background", 1: "person", 2: "car", 3: "truck"}
CLASS_NAMES = ["background", "person", "car", "truck"]
FOREGROUND  = [1, 2, 3]


def load_features(path):
    """Load features, labels, boxes, and image_ids from a .npz file."""
    data = np.load(path)
    return (
        data["features"],
        data["labels"],
        data["boxes"],
        data["image_ids"],
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
    """Print classification metrics and return foreground-only macro F1."""
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
    """Load ground truth boxes from regions.csv (foreground only)."""
    label_map = {"person": 1, "car": 2, "truck": 3}
    df = pd.read_csv(regions_csv)
    df["class_label"] = df["class_label"].map(label_map)
    df = df.dropna(subset=["class_label"])
    df["class_label"] = df["class_label"].astype(int)
    df = df[["image_id", "x1", "y1", "x2", "y2", "class_label"]]
    if image_ids is not None:
        df = df[df["image_id"].isin(image_ids)]
    return df


def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE to balance foreground classes only.
    Background is not resampled.
    Uses a conservative strategy: double the minority classes (car, truck).
    """
    print("\nApplying SMOTE to foreground classes...")
    
    # Separate background and foreground
    bg_mask = y_train == 0
    fg_mask = ~bg_mask
    
    X_bg, y_bg = X_train[bg_mask], y_train[bg_mask]
    X_fg, y_fg = X_train[fg_mask], y_train[fg_mask]
    
    print(f"  Before SMOTE:")
    unique, counts = np.unique(y_fg, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {LABEL_NAMES[u]:<12}: {c:>8}")
    
    # Apply SMOTE with conservative sampling strategy
    # Target: double the minority classes (car: ~7k -> ~14k, truck: ~7k -> ~14k)
    sampling_strategy = {
        2: 14000,  # car
        3: 14000   # truck
    }
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=5)
    X_fg_resampled, y_fg_resampled = smote.fit_resample(X_fg, y_fg)
    
    print(f"  After SMOTE:")
    unique, counts = np.unique(y_fg_resampled, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {LABEL_NAMES[u]:<12}: {c:>8}")
    
    # Subsample background to avoid memory overflow (keep 3x largest fg class)
    n_fg_max = y_fg_resampled.shape[0]
    n_bg_keep = min(len(y_bg), n_fg_max * 3)
    rng = np.random.RandomState(random_state)
    bg_idx = rng.choice(len(y_bg), size=n_bg_keep, replace=False)
    X_bg_sub = X_bg[bg_idx]
    y_bg_sub = y_bg[bg_idx]

    # Combine subsampled background and resampled foreground
    X_combined = np.vstack([X_bg_sub, X_fg_resampled])
    y_combined = np.concatenate([y_bg_sub, y_fg_resampled])

    # Shuffle
    shuffle_idx = rng.permutation(len(y_combined))
    X_combined = X_combined[shuffle_idx]
    y_combined = y_combined[shuffle_idx]
    
    print(f"  Total after SMOTE: {len(y_combined)} samples")
    return X_combined, y_combined


def train_and_evaluate(X_train, y_train, X_val, y_val, boxes_val, ids_val,
                       ground_truth_df, model_name, output_dir):
    """Train SVM and evaluate on validation set."""
    print(f"\nTraining {model_name}...")
    
    svm = SGDClassifier(loss='hinge', max_iter=2000, random_state=42, tol=1e-3, n_jobs=1)
    print(f"  Start SVM fit ({X_train.shape[0]} samples)...")
    svm.fit(X_train, y_train)
    print("  SVM fit done.")
    
    y_pred = svm.predict(X_val)
    scores = svm.decision_function(X_val)
    conf   = scores[np.arange(len(X_val)), y_pred]
    
    # Classification metrics
    fg_macro_f1 = print_classification_report(y_val, y_pred, model_name)
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2, 3])
    safe_name = model_name.lower().replace(' ', '_').replace('-', '_')
    plot_confusion_matrix(
        cm,
        title=f"Confusion Matrix — {model_name}",
        output_path=output_dir / f"cm_{safe_name}.png"
    )
    
    # Save predictions
    preds_df = build_predictions_df(ids_val, boxes_val, y_pred, conf, y_val)
    csv_name = f"predictions_{safe_name}.csv"
    preds_df.to_csv(output_dir / csv_name, index=False)
    print(f"  Predictions saved to {csv_name}")
    
    # mAP evaluation
    print(f"\n  mAP evaluation for {model_name}:")
    detection_results = evaluate_detection(preds_df, ground_truth_df, verbose=True)
    
    per_class = detection_results["map_per_class"]
    return {
        "fg_macro_f1": round(fg_macro_f1, 4),
        "map":         round(detection_results["map"], 4),
        "map_per_class": {
            "person": round(per_class[0], 4) if len(per_class) > 0 else float("nan"),
            "car":    round(per_class[1], 4) if len(per_class) > 1 else float("nan"),
            "truck":  round(per_class[2], 4) if len(per_class) > 2 else float("nan"),
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./coco_filtered")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path("outputs/task2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load validation data (same for all experiments)
    print("Loading validation features...")
    X_val, y_val, boxes_val, ids_val = load_features(data_dir / "features_val.npz")
    ground_truth_df = build_ground_truth_df(data_dir / "regions.csv", image_ids=set(ids_val.tolist()))
    print(f"  Val: {X_val.shape[0]} regions")

    results = {}

    # ------------------------------------------------------------------ #
    # Strategy 1: Region-level augmentation (Task 2a)                    #
    # ------------------------------------------------------------------ #
    augmented_file = data_dir / "features_train_augmented.npz"
    if augmented_file.exists():
        print("\n" + "="*70)
        print("Strategy 1: Region-level augmentation")
        print("="*70)
        X_train_aug, y_train_aug, _, _ = load_features(augmented_file)
        print(f"  Train (augmented): {X_train_aug.shape[0]} regions")
        
        for split_name, y in [("train_augmented", y_train_aug)]:
            unique, counts = np.unique(y, return_counts=True)
            print(f"\n  {split_name} label distribution:")
            for u, c in zip(unique, counts):
                print(f"    {LABEL_NAMES[u]:<12}: {c:>8} ({100*c/len(y):.1f}%)")
        
        results["Region-level Augmentation"] = train_and_evaluate(
            X_train_aug, y_train_aug, X_val, y_val, boxes_val, ids_val,
            ground_truth_df, "Region-level Augmentation", output_dir
        )
    else:
        print(f"\nWarning: {augmented_file} not found.")
        print("Run extract_features_augmented.py first to generate augmented features.")

    # ------------------------------------------------------------------ #
    # Strategy 2: SMOTE on feature vectors (Task 2b)                     #
    # ------------------------------------------------------------------ #
    print("\n" + "="*70)
    print("Strategy 2: SMOTE on feature vectors")
    print("="*70)
    
    X_train_orig, y_train_orig, _, _ = load_features(data_dir / "features_train.npz")
    print(f"  Train (original): {X_train_orig.shape[0]} regions")
    
    X_train_smote, y_train_smote = apply_smote(X_train_orig, y_train_orig)
    
    results["SMOTE"] = train_and_evaluate(
        X_train_smote, y_train_smote, X_val, y_val, boxes_val, ids_val,
        ground_truth_df, "SMOTE", output_dir
    )

    # ------------------------------------------------------------------ #
    # Strategy 3: Class-weighted SVM (baseline)                          #
    # ------------------------------------------------------------------ #
    print("\n" + "="*70)
    print("Strategy 3: Class-weighted SVM")
    print("="*70)
    print(f"  Train (original): {X_train_orig.shape[0]} regions")
    
    print("\nTraining Class-weighted SVM...")
    svm_weighted = SGDClassifier(loss='hinge', max_iter=2000, random_state=42, tol=1e-3, n_jobs=1, class_weight='balanced')
    print(f"  Start SVM (weighted) fit ({X_train_orig.shape[0]} samples)...")
    svm_weighted.fit(X_train_orig, y_train_orig)
    print("  SVM (weighted) fit done.")
    
    y_pred_weighted = svm_weighted.predict(X_val)
    scores_weighted = svm_weighted.decision_function(X_val)
    conf_weighted   = scores_weighted[np.arange(len(X_val)), y_pred_weighted]
    
    fg_macro_f1_weighted = print_classification_report(y_val, y_pred_weighted, "Class-weighted SVM")
    
    cm_weighted = confusion_matrix(y_val, y_pred_weighted, labels=[0, 1, 2, 3])
    plot_confusion_matrix(
        cm_weighted,
        title="Confusion Matrix — Class-weighted SVM",
        output_path=output_dir / "cm_class_weighted_svm.png"
    )
    
    preds_df_weighted = build_predictions_df(ids_val, boxes_val, y_pred_weighted, conf_weighted, y_val)
    preds_df_weighted.to_csv(output_dir / "predictions_class_weighted_svm.csv", index=False)
    print(f"  Predictions saved to predictions_class_weighted_svm.csv")
    
    print(f"\n  mAP evaluation for Class-weighted SVM:")
    detection_results_weighted = evaluate_detection(preds_df_weighted, ground_truth_df, verbose=True)
    
    per_class_weighted = detection_results_weighted["map_per_class"]
    results["Class-weighted SVM"] = {
        "fg_macro_f1": round(fg_macro_f1_weighted, 4),
        "map":         round(detection_results_weighted["map"], 4),
        "map_per_class": {
            "person": round(per_class_weighted[0], 4) if len(per_class_weighted) > 0 else float("nan"),
            "car":    round(per_class_weighted[1], 4) if len(per_class_weighted) > 1 else float("nan"),
            "truck":  round(per_class_weighted[2], 4) if len(per_class_weighted) > 2 else float("nan"),
        }
    }

    # ------------------------------------------------------------------ #
    # Comparison summary                                                 #
    # ------------------------------------------------------------------ #
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print("\nForeground Macro F1:")
    for method, res in results.items():
        print(f"  {method:<30}: {res['fg_macro_f1']:.4f}")
    
    print("\nmAP@0.5:")
    for method, res in results.items():
        print(f"  {method:<30}: {res['map']:.4f}")
    
    # Add Task 1 baseline for reference
    task1_results_path = Path("outputs/task1/results.json")
    if task1_results_path.exists():
        with open(task1_results_path) as f:
            task1_results = json.load(f)
        results["Baseline (Task 1)"] = task1_results["SVM"]
        print("\nBaseline (Task 1 SVM):")
        print(f"  Foreground Macro F1: {task1_results['SVM']['fg_macro_f1']:.4f}")
        print(f"  mAP@0.5            : {task1_results['SVM']['map']:.4f}")

    # Save all results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to outputs/task2/results.json")
    print("\nTask 2 complete. Check outputs/task2/ for results.")


if __name__ == "__main__":
    main()
