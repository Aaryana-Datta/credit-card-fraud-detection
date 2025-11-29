# Credit Card Fraud Detection (Hybrid ML Model)

This project implements a **hybrid fraud detection system** using:
- **XGBoost supervised classifier**
- **Isolation Forest anomaly detector**
- Fusion weight α for combining probabilities

The dataset is highly imbalanced (0.17% fraud), and the model achieves:
- **ROC-AUC ≈ 0.98**
- **Fraud Recall ≈ 0.86**
- Strong performance under class imbalance

## Features
- Hybrid prediction combining supervised + anomaly detection
- Experimented with different fusion weights α
- Achieved stable results across α values
- Streamlit app for live predictions
- API-ready architecture (`utils.py`)
