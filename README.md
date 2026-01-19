# 🚢 AI Logistics Predictor

**Predicting shipment delays with 78% accuracy (ROC-AUC) to help logistics managers make proactive decisions**

[[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://your-deployed-link.com)
](https://ai-logistics-predictor.streamlit.app/)
* **LinkedIn:** [linkedin.com/in/mohamad-al-charif](https://www.linkedin.com/in/mohamad-al-charif-5b7922278)
* 
## 📊 Project Overview
Built as part of my M1 AI studies. This end-to-end ML system analyzes 10,324 historical shipments to predict delivery delays before they happen.

**Why this matters:** Late deliveries cost global logistics companies billions annually. This model catches 68% of delays with explainable AI that business stakeholders can trust.

## 🎯 Key Features
- **Interactive Dashboard**: 4-page Streamlit app with executive KPIs, risk heatmaps, and real-time predictions
- **Smart Feature Engineering**: Frequency encoding for high-cardinality countries, one-hot for shipment modes
- **Hyperparameter Tuning**: GridSearchCV optimization for 10K dataset
- **Business-Ready**: Explainable predictions with actionable recommendations

## 📈 Model Performance
| Metric | Value | Interpretation |
|--------|-------|----------------|
| ROC-AUC | 0.78 | Strong discrimination |
| Recall (Late) | 68% | Catches 2/3 delays |
| Precision (Late) | 26% | Optimized for catching delays |

**Key Insight:** Model prioritizes catching delays over minimizing false alarms - better to prepare for a delay that doesn't happen!

## 🛠️ Tech Stack
- **ML**: Python, scikit-learn, pandas, RandomForest
- **Visualization**: Streamlit, Plotly, matplotlib
- **Deployment**: Streamlit Cloud (or local)

## 🚀 Quick Start
```bash
# Clone the repo
git clone https://github.com/YourUsername/ai-logistics-predictor.git
cd ai-logistics-predictor

# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Launch dashboard
streamlit run dashboard.py
```


## 💡 Business Impact
- **Top Risk Country**: Burundi (38.8% late rate)
- **Most Reliable Mode**: Air freight (9.6% delays)
- **Weight Factor**: Heavy cargo (>1000kg) has 2.4x higher delay rates

