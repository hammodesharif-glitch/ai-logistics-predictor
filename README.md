# 🚢 Shipment Delay Predictor

[![Streamlit](https://img.shields.io/badge/Live_demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://ai-logistics-predictor.streamlit.app/) · [LinkedIn](https://www.linkedin.com/in/mohamad-al-charif)

Predicts, **before dispatch**, whether a pharmaceutical supply-chain shipment line will be delivered late, and explains why. It is trained on the USAID SCMS delivery history (10,324 lines, 43 countries, 2006–2015, 11.5% late).

## Results

| Evaluation | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Previous model | 0.78 | – | 68% | 26% | – |
| **Grouped 5-fold CV** | **0.900** | **0.519** | **80.1%** | **39.0%** | **0.525** |
| **Temporal hold-out** (train < May 2014, test after) | **0.836** | **0.350** | **76.4%** | **32.2%** | **0.453** |

A random classifier would score a PR-AUC equal to the base rate, about 0.11–0.13. The decision threshold is chosen on out-of-fold predictions: it is the best-precision threshold that still keeps recall at 80% or higher.

## What makes it reliable

- **Correct target.** Dates are parsed with their real formats. The original pipeline mis-parsed them and found 0 late shipments.
- **No leakage.** Delivery and recorded dates are never used as features (a unit test checks this). Lines from the same shipment (`ASN/DN #`) never appear on both sides of a fold, and a strict *train-on-past / test-on-future* split is reported too.
- **46 engineered features.** These cover planning lead times (PQ→PO→schedule), shipment and PO consolidation size, the daily load per vendor and per country, value/weight/freight/insurance ratios, flags for unreported or linked weight and freight, calendar effects, and native categorical handling for vendor, site, product, INCO term and route.
- **Model.** A LightGBM seed ensemble with averaged probabilities.
- **Explainability.** Exact per-prediction SHAP contributions (TreeSHAP via `pred_contrib`), both globally and for each shipment.

## Dashboard

`streamlit run dashboard.py`

- **Overview**: real KPIs, the quarterly late-rate trend, and risk by country, mode, INCO term and fulfilment channel
- **Model performance**: both evaluations, ROC and PR curves, and an interactive threshold slider that shows the recall/precision trade-off
- **Predict a shipment**: edit any shipment's country, mode, vendor, value, quantity or date, then get the probability and a SHAP explanation
- **Risk watchlist**: the highest-risk lines, ranked by honest out-of-fold scores

## Run it

```bash
git clone https://github.com/hammodesharif-glitch/ai-logistics-predictor.git
cd ai-logistics-predictor
pip install -r requirements.txt
python src/train.py
pytest -q
streamlit run dashboard.py
```

## Structure

```
src/features.py   data loading, target, feature engineering
src/model.py      LightGBM seed ensemble + SHAP contributions
src/train.py      grouped CV, temporal hold-out, threshold selection, artefacts
dashboard.py      Streamlit dashboard
tests/            pipeline and leakage tests
models/ reports/  trained model, metrics.json, out-of-fold predictions
```
