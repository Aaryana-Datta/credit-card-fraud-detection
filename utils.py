import numpy as np
import joblib

# Load artifacts (ensure these files are in the 'artifacts' folder)
xgb_model = joblib.load("artifacts/xgb_model.joblib")
iso_forest = joblib.load("artifacts/iso_forest.joblib")
anomaly_scaler = joblib.load("artifacts/anomaly_scaler.joblib")

def compute_anomaly_scores(X):
    """Compute normalized anomaly scores using Isolation Forest."""
    raw = -iso_forest.decision_function(X)  
    raw = np.array(raw).reshape(-1, 1)
    return anomaly_scaler.transform(raw).ravel()

def predict_transaction(X, alpha=0.7, threshold=0.5):
    """
    Inputs: X = list of feature lists e.g. [[f1,...,f30], ...]
    Outputs: predictions with xgb_prob, anomaly_score, combined_score, label
    """
    X = np.array(X)

    # XGBoost fraud probability
    xgb_prob = xgb_model.predict_proba(X)[:, 1]

    # anomaly score
    anomaly_score = compute_anomaly_scores(X)

    # hybrid fuse
    combined = alpha * xgb_prob + (1 - alpha) * anomaly_score

    # final label
    pred = (combined >= threshold).astype(int)

    # prepare results
    results = []
    for p, a, c, label in zip(xgb_prob, anomaly_score, combined, pred):
        results.append({
            "xgb_prob": float(p),
            "anomaly_score": float(a),
            "combined_score": float(c),
            "pred_label": int(label)
        })
    return results
