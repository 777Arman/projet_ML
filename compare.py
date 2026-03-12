"""
compare.py

Reads pre-saved results from outputs/task{1,2,3,4}/results.json and
produces a unified comparison table and bar chart of foreground macro F1
and mAP@0.5 across all tasks.

Usage:
    python compare.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("outputs/compare")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
# Load results                                                        #
# ------------------------------------------------------------------ #

def load_json(path):
    with open(path) as f:
        return json.load(f)


def collect_models():
    """
    Returns a list of dicts:
        {"task": str, "model": str, "f1": float, "map": float}
    in the order they appear across tasks 1-4.
    """
    models = []

    # Task 1
    t1 = load_json("outputs/task1/results.json")
    models.append({"task": "Task 1", "model": "Linear SVM",
                   "f1": t1["SVM"]["fg_macro_f1"], "map": t1["SVM"]["map"]})
    models.append({"task": "Task 1", "model": "Decision Tree (baseline)",
                   "f1": t1["Decision Tree"]["fg_macro_f1"],
                   "map": t1["Decision Tree"]["map"]})

    # Task 2
    t2 = load_json("outputs/task2/results.json")
    for name in ["Region-level Augmentation", "SMOTE", "Class-weighted SVM"]:
        if name in t2:
            models.append({"task": "Task 2", "model": name,
                           "f1": t2[name]["fg_macro_f1"], "map": t2[name]["map"]})

    # Task 3
    t3 = load_json("outputs/task3/results.json")
    models.append({"task": "Task 3", "model": "Approx. RBF SVM",
                   "f1": t3["Approximate RBF SVM"]["fg_macro_f1"],
                   "map": t3["Approximate RBF SVM"]["map"]})

    # Task 4 — best depth Decision Tree
    t4 = load_json("outputs/task4/results.json")
    best_d = t4["best_depth"]
    models.append({"task": "Task 4", "model": f"Decision Tree (depth={best_d})",
                   "f1": t4["best_val_f1"], "map": t4["best_depth_map"]})

    return models


# ------------------------------------------------------------------ #
# Print table                                                         #
# ------------------------------------------------------------------ #

def print_table(models):
    sep  = "  " + "-" * 64
    head = f"  {'Task':<8} {'Model':<34} {'F1 macro (fg)':>14} {'mAP@0.5':>10}"
    print("\n" + "=" * 68)
    print("  ALL-TASK MODEL COMPARISON")
    print("=" * 68)
    print(head)
    print(sep)
    prev_task = None
    for m in models:
        if prev_task and m["task"] != prev_task:
            print(sep)
        prev_task = m["task"]
        print(f"  {m['task']:<8} {m['model']:<34} {m['f1']:>14.4f} {m['map']:>10.4f}")
    print(sep)
    best_f1  = max(models, key=lambda m: m["f1"])
    best_map = max(models, key=lambda m: m["map"])
    print(f"\n  Best F1 : {best_f1['model']}  ({best_f1['f1']:.4f})")
    print(f"  Best mAP: {best_map['model']}  ({best_map['map']:.4f})")
    print("=" * 68 + "\n")


# ------------------------------------------------------------------ #
# Bar chart                                                           #
# ------------------------------------------------------------------ #

TASK_COLORS = {
    "Task 1": "#4C72B0",
    "Task 2": "#55A868",
    "Task 3": "#C44E52",
    "Task 4": "#8172B2",
}


def plot_comparison(models):
    labels     = [m["model"] for m in models]
    f1_vals    = [m["f1"]    for m in models]
    map_vals   = [m["map"]   for m in models]
    bar_colors = [TASK_COLORS[m["task"]] for m in models]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width / 2, f1_vals,  width, label="Fg macro F1",
                   color=bar_colors, alpha=0.9)
    bars2 = ax.bar(x + width / 2, map_vals, width, label="mAP@0.5",
                   color=bar_colors, alpha=0.5, hatch="//")

    # value labels on top of each bar
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)

    # task colour legend patches
    from matplotlib.patches import Patch
    task_patches = [Patch(facecolor=c, label=t) for t, c in TASK_COLORS.items()]
    metric_legend = ax.legend(loc="upper left", fontsize=9)
    ax.add_artist(metric_legend)
    ax.legend(handles=task_patches, loc="upper right", fontsize=9, title="Task")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Comparison — Foreground Macro F1 and mAP@0.5 Across Tasks")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = OUTPUT_DIR / "comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    models = collect_models()
    print_table(models)
    plot_comparison(models)

    # Save summary JSON
    out_json = OUTPUT_DIR / "results.json"
    with open(out_json, "w") as f:
        json.dump(models, f, indent=2)
    print(f"  Saved: {out_json}")


if __name__ == "__main__":
    main()
