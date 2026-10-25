# AI-Driven GST Invoice Anomaly Detection with Explainable Fraud Risk Analytics

**BIP Project ID:** `2026MIN500`  
**Institution:** Bannari Amman Institute of Technology  
**Department:** Artificial Intelligence and Data Science  
**Academic Year:** 2024–2025 (S5 Project Work I)  

### Team Members:
- **Nishaanth Mohanasundaram** (`7376242AL160`)
- **Hari Pranav Murugan** (`7376242AL130`)
- **Rithik P** (`7376242AL177`)

### Faculty Guide:
- **Chozharajan P (AD11026)**, Assistant Professor Level III, Dept of AI & DS

---

## 📌 Project Overview
An end-to-end Explainable AI (XAI) forensic audit solution designed to detect fraudulent Goods and Services Tax (GST) invoices in India. Targets major tax evasion schemes including:
- **Fake Invoicing / Non-supply of Goods** to fraudulently monetize Input Tax Credit (ITC).
- **Circular Trading / Carousel Billing Loops**.
- **Tax Slab Arbitrage & Mathematical Calculation Tampering**.
- **Shell / Dormant Entity Sudden Volume Surges**.
- **Off-Hour Batch Dumping & Syntax-Invalid GSTINs**.

---

## ⚙️ Key System Capabilities
1. **Domain Feature Preprocessing:** CBIC-compliant GSTIN regex and checksum checking, statutory tax slab reconciliation, vendor baseline deviations, and round-sum indicators.
2. **Unsupervised Anomaly Detection:** **Isolation Forest** (primary model) alongside **One-Class SVM** and **Local Outlier Factor (LOF)** comparative benchmarks.
3. **Calibrated Fraud Risk Scoring:** Maps raw decision functions into an intuitive **0 to 100** score categorized into **Low Risk (<35)**, **Medium Risk (35–70)**, and **High Risk (>70)**.
4. **Explainable AI (XAI):** **SHAP (SHapley Additive exPlanations)** local feature attributions paired with automated, natural-language legal reason codes for auditors.
5. **Interactive Forensic Dashboard:** Streamlit UI with executive KPIs, filterable invoice explorer, single-invoice SHAP waterfall diagrams, vendor risk heatmaps, batch CSV ingestion, and CSV audit trail export.

---

## 🚀 How to Run the Project

### 1. Requirements
Ensure Python 3.10+ is installed on your system. Required packages:
```bash
py -m pip install pandas scikit-learn shap streamlit matplotlib seaborn plotly
```

### 2. Run the Full Model Training Pipeline
To generate the realistic GST dataset, train the Isolation Forest model, evaluate metrics, and cache models:
```bash
py train_pipeline.py
```
*Expected Evaluation Output: ROC-AUC: ~0.9966, Accuracy: ~99%, F1-Score (Anomaly): ~0.92.*

### 3. Run Automated Tests
```bash
py -m unittest discover -s tests -p "test_*.py"
```

### 4. Launch the Interactive Forensic Dashboard
Double-click `run_dashboard.bat` or run:
```bash
py -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 📁 Repository Structure
```
gst_invoice_anomaly_detection/
├── app.py                      # Streamlit interactive forensic dashboard
├── generate_dataset.py         # Realistic GST invoice & fraud typology generator
├── train_pipeline.py           # Training, evaluation & model persistence pipeline
├── run_dashboard.bat           # One-click dashboard launcher
├── data/
│   ├── gst_invoices.csv        # Synthetic GST invoice transaction dataset
│   └── scored_gst_invoices.csv # Invoices scored with Fraud Risk Score & XAI columns
├── models/
│   ├── anomaly_detector.joblib # Saved Isolation Forest & benchmark models
│   └── preprocessor.joblib     # Saved feature preprocessor & vendor baselines
├── src/
│   ├── __init__.py
│   ├── preprocessor.py         # GST domain syntax validation & feature extraction
│   ├── anomaly_models.py       # Isolation Forest, LOF, One-Class SVM & risk calibration
│   └── explainability.py       # SHAP attributions & natural-language reason code synthesizer
├── tests/
│   ├── test_data_pipeline.py   # Unit tests for preprocessing & dataset generation
│   └── test_model_and_xai.py   # Unit tests for models & SHAP explanations
└── documentation/
    └── PROJECT_REPORT.md       # Complete academic project report (Chapters 1 to 5 + IEEE Refs)
```
