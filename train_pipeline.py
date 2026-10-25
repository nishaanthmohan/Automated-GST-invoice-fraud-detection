"""
Training and Evaluation Pipeline for GST Invoice Anomaly Detection
Preprocesses data, fits models, evaluates detection metrics, and caches models.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from generate_dataset import generate_gst_dataset
from src.preprocessor import GSTPreprocessor
from src.anomaly_models import GSTAnomalyDetector

def run_training_pipeline(data_path="data/gst_invoices.csv", model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Check or generate dataset
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Generating synthetic GST invoice dataset...")
        df_raw = generate_gst_dataset(num_records=2500, anomaly_ratio=0.08, output_path=data_path)
    else:
        print(f"Loading GST invoice dataset from {data_path}...")
        df_raw = pd.read_csv(data_path)

    # 2. Feature Engineering & Preprocessing
    print("Extracting domain compliance features and behavioral metrics...")
    preprocessor = GSTPreprocessor()
    df_enriched = preprocessor.extract_features(df_raw, is_training=True)
    X = preprocessor.get_feature_matrix(df_enriched)
    
    # 3. Model Training
    print("Fitting Isolation Forest and benchmark anomaly models...")
    detector = GSTAnomalyDetector(contamination=0.08, random_state=42)
    detector.fit(X)

    # 4. In-sample Prediction & Risk Scoring
    print("Computing calibrated Fraud Risk Scores and categorization...")
    predictions = detector.predict_risk(X, enriched_df=df_enriched)
    
    for key, val in predictions.items():
        df_enriched[key] = val

    # 5. Evaluate Performance against Ground Truth Anomalies
    if "is_anomaly" in df_enriched.columns:
        y_true = df_enriched["is_anomaly"].values
        y_pred = df_enriched["isolation_forest_anomaly"].values
        scores = df_enriched["fraud_risk_score"].values
        
        auc = roc_auc_score(y_true, scores)
        print("\n" + "="*50)
        print("  MODEL EVALUATION RESULTS (ISOLATION FOREST)")
        print("="*50)
        print(f"ROC-AUC Score: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("="*50 + "\n")

    # 6. Save Scored Invoices and Fitted Pipeline Artifacts
    output_scored_path = "data/scored_gst_invoices.csv"
    df_enriched.to_csv(output_scored_path, index=False)
    print(f"Scored dataset saved to: {output_scored_path}")

    detector.save(os.path.join(model_dir, "anomaly_detector.joblib"))
    joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.joblib"))
    print("All models and preprocessors saved successfully!")

    return df_enriched, preprocessor, detector

if __name__ == "__main__":
    run_training_pipeline()
