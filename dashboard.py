import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from features import build, load

st.set_page_config(page_title="Shipment Delay Predictor", page_icon="🚢", layout="wide")


@st.cache_data
def data():
    df = load(ROOT / "SCMS_Delivery_History_Dataset.csv")
    oof = pd.read_csv(ROOT / "reports" / "oof_predictions.csv")
    return df.merge(oof[["ID", "oof"]], on="ID"), json.loads((ROOT / "reports" / "metrics.json").read_text())


@st.cache_resource
def bundle():
    return joblib.load(ROOT / "models" / "delay_model.joblib")


def align(X, b):
    for c, cats in b["categories"].items():
        X[c] = pd.Categorical(X[c].astype(str), categories=cats)
    return X[b["columns"]]


def rate_by(df, col, min_n=50):
    g = df.groupby(col, observed=True)["is_late"].agg(late_rate="mean", shipments="size").reset_index()
    return g[g.shipments >= min_n].sort_values("late_rate", ascending=False)


df, metrics = data()
b = bundle()
page = st.sidebar.radio("View", ["Overview", "Model performance", "Predict a shipment", "Risk watchlist"])
st.sidebar.metric("Temporal hold-out ROC-AUC", f"{metrics['temporal']['roc_auc']:.3f}")
st.sidebar.metric("Cross-validated ROC-AUC", f"{metrics['cv']['roc_auc']:.3f}")
st.sidebar.caption("LightGBM seed ensemble · 46 leakage-safe features · SCMS supply-chain data (10,324 lines)")

if page == "Overview":
    st.title("🚢 Shipment Delay Intelligence")
    late = df[df.is_late == 1]
    c = st.columns(4)
    c[0].metric("Shipment lines", f"{len(df):,}")
    c[1].metric("Late delivery rate", f"{df.is_late.mean():.1%}")
    c[2].metric("Median delay when late", f"{late.days_late.median():.0f} days")
    c[3].metric("Countries served", df.Country.nunique())
    monthly = df.set_index("Scheduled Delivery Date").resample("QE")["is_late"].agg(["mean", "size"]).reset_index()
    st.plotly_chart(px.line(monthly[monthly["size"] >= 30], x="Scheduled Delivery Date", y="mean", markers=True,
                            labels={"mean": "Late rate"}, title="Late rate by quarter"), width="stretch")
    l, r = st.columns(2)
    for col, box, n in [("Country", l, 100), ("Shipment Mode", r, 50), ("Vendor INCO Term", l, 50), ("Fulfill Via", r, 50)]:
        g = rate_by(df, col, n).head(12)
        box.plotly_chart(px.bar(g, x="late_rate", y=col, orientation="h", color="late_rate", hover_data=["shipments"],
                                color_continuous_scale="Reds", title=f"Late rate by {col} (≥{n} lines)")
                         .update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
                         .update_xaxes(tickformat=".0%"), width="stretch")

elif page == "Model performance":
    st.title("📈 Model performance")
    for name, key in [("Temporal hold-out (train on past, test on the most recent 20%)", "temporal"),
                      ("Grouped 5-fold CV (no shipment split across folds)", "cv")]:
        m = metrics[key]
        st.subheader(name)
        c = st.columns(5)
        for i, (k, lab) in enumerate([("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("recall", "Recall"),
                                      ("precision", "Precision"), ("f1", "F1")]):
            c[i].metric(lab, f"{m[k]:.3f}" if "auc" in k or k == "f1" else f"{m[k]:.1%}")
    st.caption(f"Previous model: ROC-AUC 0.78, recall 68%, precision 26%. PR-AUC baseline (random) = "
               f"{metrics['cv']['base_rate']:.3f}.")
    y, p = df.is_late.values, df.oof.values
    t = st.slider("Decision threshold", 0.02, 0.9, float(b["threshold"]), 0.01)
    pred = p >= t
    c = st.columns(4)
    c[0].metric("Delays caught", f"{(pred & (y == 1)).sum() / y.sum():.1%}")
    c[1].metric("Alert precision", f"{(pred & (y == 1)).sum() / max(pred.sum(), 1):.1%}")
    c[2].metric("Alerts raised", f"{pred.sum():,}")
    c[3].metric("Missed delays", f"{(~pred & (y == 1)).sum():,}")
    l, r = st.columns(2)
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    l.plotly_chart(go.Figure([go.Scatter(x=fpr, y=tpr, name="Model"),
                              go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="Random")])
                   .update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="Recall"),
                   width="stretch")
    r.plotly_chart(go.Figure(go.Scatter(x=rec, y=prec, fill="tozeroy"))
                   .update_layout(title="Precision-recall curve", xaxis_title="Recall", yaxis_title="Precision"),
                   width="stretch")
    imp = pd.Series(metrics["importance"]).head(15).sort_values()
    st.plotly_chart(px.bar(x=imp.values, y=imp.index, orientation="h", title="Global feature impact (mean |SHAP|)",
                           labels={"x": "mean |SHAP| (log-odds)", "y": ""}), width="stretch")

elif page == "Predict a shipment":
    st.title("🎯 Predict delay risk")
    sid = st.selectbox("Start from an existing shipment line (ID)", df.ID.tolist(), index=len(df) - 1)
    row = df.index[df.ID == sid][0]
    base = df.loc[row]
    c = st.columns(3)
    edits = {
        "Country": c[0].selectbox("Country", sorted(df.Country.unique()), sorted(df.Country.unique()).index(base.Country)),
        "Shipment Mode": c[1].selectbox("Mode", ["Air", "Air Charter", "Ocean", "Truck"],
                                        ["Air", "Air Charter", "Ocean", "Truck"].index(base["Shipment Mode"])
                                        if base["Shipment Mode"] in ["Air", "Air Charter", "Ocean", "Truck"] else 0),
        "Vendor": c[2].selectbox("Vendor", sorted(df.Vendor.unique()), sorted(df.Vendor.unique()).index(base.Vendor)),
        "Line Item Value": c[0].number_input("Line value (USD)", 0.0, value=float(base["Line Item Value"])),
        "Line Item Quantity": c[1].number_input("Quantity", 0, value=int(base["Line Item Quantity"])),
        "Scheduled Delivery Date": pd.Timestamp(c[2].date_input("Scheduled delivery", base["Scheduled Delivery Date"])),
    }
    edited = df.copy()
    for k, v in edits.items():
        edited.at[row, k] = v
    X = align(build(edited).loc[[row]], b)
    prob = b["model"].predict_proba(X)[0, 1]
    contrib = pd.Series(b["model"].contributions(X)[0, :-1], X.columns)
    level = "HIGH" if prob >= max(b["threshold"] * 3, 0.5) else "ELEVATED" if prob >= b["threshold"] else "LOW"
    st.metric("Probability of late delivery", f"{prob:.1%}", f"{level} risk · alert threshold {b['threshold']:.1%}",
              delta_color="inverse" if level != "LOW" else "normal")
    if not np.isnan(base.days_late):
        st.caption(f"Historical outcome of the original line: {'late' if base.is_late else 'on time'} "
                   f"({int(base.days_late)} days vs schedule)")
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:12]).iloc[::-1]
    st.plotly_chart(px.bar(x=top.values, y=top.index, orientation="h", color=top.values > 0,
                           color_discrete_map={True: "#d62728", False: "#2ca02c"},
                           title="Why? Feature contributions (SHAP, log-odds)", labels={"x": "", "y": ""})
                    .update_layout(showlegend=False), width="stretch")

else:
    st.title("⚠️ Risk watchlist")
    st.caption("Out-of-fold risk scores: every line is scored by a model that never saw its shipment.")
    cols = ["ID", "Scheduled Delivery Date", "Country", "Shipment Mode", "Vendor", "Line Item Value", "oof", "is_late"]
    country = st.multiselect("Country", sorted(df.Country.unique()))
    view = df[df.Country.isin(country)] if country else df
    st.dataframe(view.nlargest(200, "oof")[cols].rename(columns={"oof": "risk", "is_late": "was_late"}),
                 width="stretch", hide_index=True,
                 column_config={"risk": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
