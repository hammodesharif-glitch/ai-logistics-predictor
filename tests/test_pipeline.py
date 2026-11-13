import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import build, load


def test_target_is_parsed():
    df = load(ROOT / "SCMS_Delivery_History_Dataset.csv")
    assert df["Scheduled Delivery Date"].notna().all()
    assert 0.05 < df["is_late"].mean() < 0.25


def test_no_outcome_leakage():
    X = build(load(ROOT / "SCMS_Delivery_History_Dataset.csv"))
    banned = ("Delivered", "Recorded", "days_late", "is_late")
    assert not [c for c in X.columns if any(b in c for b in banned)]


def test_features_are_numeric_or_categorical():
    X = build(load(ROOT / "SCMS_Delivery_History_Dataset.csv"))
    assert all(str(t) == "category" or np.issubdtype(t, np.number) for t in X.dtypes)
