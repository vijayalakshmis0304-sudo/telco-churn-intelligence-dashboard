# 📡 Telco Customer Churn Intelligence
### End-to-End ML Pipeline · XGBoost · SHAP Explainability · Interactive Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-AUC--ROC%200.845-brightgreen?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-7B61FF?style=flat)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![Dashboard](https://img.shields.io/badge/Dashboard-HTML%2FJS-00C2FF?style=flat)
![Dataset](https://img.shields.io/badge/Dataset-7%2C043%20customers-blue?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📸 Dashboard Preview

![Telco Churn Dashboard](Dashboard-Preview/dashboard-preview.png)

> 🔴 **[▶ View Live Dashboard](https://vijayalakshmis0304-sudo.github.io/telco-churn-intelligence-dashboard/Dashboard-HTML/telco_churn_dashboard.html)**

---

## 🔍 Project Overview

A full machine learning project predicting customer churn for a telecommunications company, built on the IBM Watson Telco dataset (7,043 customers · 21 features). The pipeline covers everything from raw data cleaning and exploratory analysis to model training, SHAP-based explainability, business impact quantification, and an interactive HTML dashboard.

**Business problem:** 26.5% of customers churned, representing ~$219K in annual revenue at risk. The goal was to identify *who* is likely to leave and *why* — so retention teams can act before it happens.

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Overall churn rate | **26.5%** (1,869 / 7,043) |
| Best model | **XGBoost** |
| AUC-ROC | **0.8452** |
| F1 Score | **0.6029** |
| High-risk customers identified | **282** (top 20% predicted probability) |
| Annual revenue at risk | **$219,154** |
| Estimated savings if 30% retained | **$65,746 / year** |

---

## 🧠 Top Churn Drivers (SHAP)

Ranked by mean absolute SHAP value across the test set:

| Rank | Feature | SHAP Importance |
|---|---|---|
| 1 | Tenure | 0.516 |
| 2 | Contract: Two Year | 0.514 |
| 3 | Internet: Fiber Optic | 0.346 |
| 4 | Contract: One Year | 0.233 |
| 5 | Payment: Electronic Check | 0.227 |
| 6 | Monthly Charges | 0.206 |
| 7 | Total Charges | 0.191 |
| 8 | Internet: No Service | 0.141 |
| 9 | Paperless Billing | 0.131 |
| 10 | Online Security | 0.113 |

---

## 📁 Repository Structure

```
telco-churn-intelligence-dashboard/
│
├── Code/
│   └── telco_churn_analysis.ipynb        # Full ML pipeline (Jupyter / Colab)
│
├── Dashboard-HTML/
│   └── telco_churn_dashboard.html        # Interactive analytics dashboard
│
├── Dashboard-Preview/
│   └── dashboard-preview.png             # Screenshot for README preview
│
├── Dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Source dataset (IBM Watson)
│
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

**Analysis & Modelling**
- Python 3.10+
- pandas · NumPy · SciPy
- scikit-learn (Logistic Regression, Random Forest, preprocessing, metrics)
- XGBoost
- SHAP

**Visualisation**
- Matplotlib · Seaborn (notebook plots)
- Chart.js 4.4 (dashboard charts)
- Vanilla HTML / CSS / JavaScript (dashboard UI)

---

## 🚀 Getting Started

### Run the notebook (Google Colab — recommended)

1. Open `Code/telco_churn_analysis.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Upload `WA_Fn-UseC_-Telco-Customer-Churn.csv` to the Colab files panel
3. Run all cells (`Runtime → Run all`)

Dependencies install automatically in the first cell:
```python
!pip install xgboost shap -q
```

### Run locally

```bash
git clone https://github.com/vijayalakshmis0304-sudo/telco-churn-intelligence-dashboard.git
cd telco-churn-intelligence-dashboard

pip install -r requirements.txt

jupyter notebook Code/telco_churn_analysis.ipynb
```

### View the dashboard

Open `Dashboard-HTML/telco_churn_dashboard.html` directly in any modern browser — no server or installation required.

Or use the **[live demo link](https://vijayalakshmis0304-sudo.github.io/telco-churn-intelligence-dashboard/Dashboard-HTML/telco_churn_dashboard.html)** above.

---

## 📋 Notebook Walkthrough

**1 · Setup & Data Loading**
Install dependencies, import libraries, load the CSV, and inspect the raw dataset shape and churn distribution.

**2 · Data Cleaning & Preprocessing**
Fix `TotalCharges` (11 blank values coerced to NaN and median-filled), drop `customerID`, binary-encode Yes/No columns, one-hot encode multi-value categoricals, and scale numeric features. Final feature matrix: 30 columns.

**3 · Train / Test Split**
80/20 stratified split (5,634 train · 1,409 test) preserving the 26.5% churn ratio in both sets.

**4 · Exploratory Data Analysis**
Four-panel overview chart covering churn distribution, monthly charges by churn status, churn rate by contract type, and tenure histograms. Correlation heatmap across numeric features.

**5 · Hypothesis Testing (SciPy)**

- **H1 — Monthly Charges vs Churn:** t-test, p < 0.0001. Churned customers pay 22% more per month ($74.44 vs $61.27).
- **H2 — Contract Type vs Churn:** Chi-squared, p < 0.0001. Month-to-month churn = 42.7%; two-year = 2.8%.
- **H3 — Tenure vs Churn:** t-test, p < 0.0001. Churned avg tenure = 18.0 months vs 37.6 months for retained.

**6 · Model Training & Comparison**

Three classifiers trained and evaluated:

| Model | AUC-ROC | F1 Score |
|---|---|---|
| XGBoost | **0.8452** | 0.6029 |
| Logistic Regression | 0.8421 | 0.6040 |
| Random Forest | 0.8204 | 0.5482 |

**7 · SHAP Explainability**
TreeExplainer applied to the XGBoost model. Bar chart of global mean-absolute SHAP values and beeswarm dot plot showing directional feature impact per prediction.

**8 · Risk Segmentation**
Predictions bucketed into three tiers — Low Risk (< 0.3), Medium Risk (0.3–0.6), High Risk (> 0.6) — and saved to `churn_predictions.csv`.

**9 · Business Impact Report**
Quantifies annual revenue at risk from high-risk customers and models potential savings from a 30% retention rate via targeted offers.

**10 · Export**
Saves all CSVs and PNGs for use in the dashboard and reporting.

---

## 📈 Dashboard

`Dashboard-HTML/telco_churn_dashboard.html` is a self-contained, single-file analytics dashboard built with Chart.js and vanilla JS — no frameworks, no build step.

**Sections:**
- Executive KPI cards (churn rate, high-risk count, AUC-ROC, revenue at risk, avg monthly charge)
- EDA charts: contract type breakdown, churn by tenure band, payment method analysis, internet service comparison
- ML section: model comparison table, predicted risk segments (donut), risk tier bars
- SHAP feature importance ranking
- AI-generated business recommendations
- Power BI DAX measures (copy-paste ready)

---

## 💡 Business Recommendations

Based on the ML output and SHAP analysis:

**Immediate (High Risk tier — 282 customers):** Targeted outreach to month-to-month + fiber optic customers with churn probability above 0.60. ~$219K annual revenue at risk.

**Contract upgrade campaign:** Incentivise month-to-month customers to switch to annual or two-year plans. Churn drops from 42.7% to 2.8%. Even 10% conversion saves ~$165K in customer lifetime value.

**Payment method migration:** Electronic check users churn at 45.3% — 3× the rate of auto-pay customers. Offer a 5% bill credit to migrate to bank transfer or credit card.

**Fiber optic bundle:** Add OnlineSecurity + TechSupport as a discounted bundle for fiber customers (41.9% churn) — SHAP confirms these add-ons significantly reduce churn probability.

**Early-tenure loyalty programme:** 47.4% of customers churn within their first 12 months. Deploy onboarding touchpoints at months 1, 3, and 6, and offer a loyalty discount at the month-11 renewal window.

---

## 📂 Dataset

**IBM Watson Telco Customer Churn** (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)

- 7,043 rows · 21 columns
- Features: demographics (gender, SeniorCitizen, Partner, Dependents), services (PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies), account (Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, tenure)
- Target: `Churn` (Yes / No)

Available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and the IBM Watson Analytics community.

---

## 🪪 License

This project is open source under the [MIT License](LICENSE).

---

*MSc Data Science · Telco Churn Intelligence · XGBoost AUC-ROC 0.845 · SHAP Explainability · 7,043 records*
