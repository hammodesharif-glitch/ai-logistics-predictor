import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]


def main():
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())
    oof = pd.read_csv(ROOT / "reports" / "oof_predictions.csv")
    y, p, cv = oof["y"], oof["oof"], m["cv"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

    fpr, tpr, _ = roc_curve(y, p)
    ax[0].plot(fpr, tpr, lw=2.5, color="#1f6feb", label=f"LightGBM ensemble (AUC {cv['roc_auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="grey", label="Random (AUC 0.500)")
    ax[0].set(title="ROC curve · grouped 5-fold CV", xlabel="False positive rate", ylabel="True positive rate (recall)")

    prec, rec, _ = precision_recall_curve(y, p)
    ax[1].plot(rec, prec, lw=2.5, color="#d1242f", label=f"LightGBM ensemble (AP {cv['pr_auc']:.3f})")
    ax[1].axhline(cv["base_rate"], ls="--", color="grey", label=f"Random (AP {cv['base_rate']:.3f})")
    ax[1].scatter(cv["recall"], cv["precision"], s=80, color="black", zorder=3,
                  label=f"Operating point ({cv['recall']:.0%} recall, {cv['precision']:.0%} precision)")
    ax[1].set(title="Precision-recall curve · grouped 5-fold CV", xlabel="Recall", ylabel="Precision")

    names = ["ROC-AUC", "PR-AUC", "Recall", "Precision", "F1"]
    keys = ["roc_auc", "pr_auc", "recall", "precision", "f1"]
    x = range(len(names))
    for off, (k, lab, c) in zip((-0.2, 0.2), [("cv", "Grouped 5-fold CV", "#1f6feb"),
                                              ("temporal", "Future hold-out (after May 2014)", "#8250df")]):
        vals = [m[k][key] for key in keys]
        bars = ax[2].bar([i + off for i in x], vals, 0.4, color=c, label=lab)
        ax[2].bar_label(bars, fmt="%.2f", fontsize=8)
    ax[2].set(title="Evaluation summary", xticks=list(x), xticklabels=names, ylim=(0, 1.05))

    for a in ax:
        a.legend(fontsize=8, loc="lower right" if a is ax[0] else "upper right")
        a.grid(alpha=0.3)
    fig.tight_layout()
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "reports" / "figures" / "results.png", dpi=150)


if __name__ == "__main__":
    main()
