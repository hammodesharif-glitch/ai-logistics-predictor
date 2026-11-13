import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import build, load
from model import SeedEnsemble

ROOT = Path(__file__).resolve().parents[1]


def scores(y, p, t):
    yhat = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return dict(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), brier=brier_score_loss(y, p),
                recall=recall_score(y, yhat), precision=precision_score(y, yhat), f1=f1_score(y, yhat),
                threshold=float(t), confusion=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
                base_rate=float(y.mean()), n=int(len(y)))


def pick_threshold(y, p, min_recall=0.80):
    grid = np.unique(np.round(p, 4))
    ok = [(precision_score(y, p >= t, zero_division=0), t) for t in grid if recall_score(y, p >= t) >= min_recall]
    return max(ok)[1]


def main():
    df = load(ROOT / "SCMS_Delivery_History_Dataset.csv")
    X, y = build(df), df["is_late"].values

    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, df["ASN/DN #"]):
        oof[te] = SeedEnsemble().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    threshold = pick_threshold(y, oof)

    cut = int(len(df) * 0.8)
    temporal = SeedEnsemble().fit(X.iloc[:cut], y[:cut]).predict_proba(X.iloc[cut:])[:, 1]

    model = SeedEnsemble().fit(X, y)
    contrib = model.contributions(X)[:, :-1]
    importance = pd.Series(np.abs(contrib).mean(0), X.columns).sort_values(ascending=False)

    metrics = dict(cv=scores(y, oof, threshold), temporal=scores(y[cut:], temporal, threshold),
                   temporal_split_date=str(df["Scheduled Delivery Date"].iloc[cut].date()),
                   importance=importance.round(5).to_dict())
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(dict(model=model, threshold=threshold, columns=list(X.columns),
                     categories={c: list(X[c].cat.categories) for c in X.select_dtypes("category")}),
                ROOT / "models" / "delay_model.joblib")
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(dict(ID=df["ID"], y=y, oof=oof)).to_csv(ROOT / "reports" / "oof_predictions.csv", index=False)
    for k in ("cv", "temporal"):
        m = metrics[k]
        print(f"{k:9s} ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f} | recall {m['recall']:.1%} | "
              f"precision {m['precision']:.1%} | F1 {m['f1']:.3f}")


if __name__ == "__main__":
    main()
