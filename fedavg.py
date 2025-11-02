import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# Config
SUMMARY_JSON = "3_pipeline_output_summary.json"
TEST_CSV = "IP-Based Flows Pre-Processed Test.csv"
LABEL_COL = "is_attack"
CAT_COLS = ["traffic", "service"]

# ----------------------
# Load summary
with open(SUMMARY_JSON, "r") as f:
    summary = json.load(f)

global_update_present = summary.get("global_update_present", False)
print(f"Features: {summary['features']}")
print(f"Train samples: {summary['train_samples']}")
print(f"Test samples: {summary['test_samples']}")
print(f"Number of shards: {summary['num_shards']}")
print(f"Global update present: {global_update_present}")

if not global_update_present:
    raise RuntimeError("No global update found in summary.json")

# Load coef and intercept as arrays
coef = np.array(summary["global_update_coef"], dtype=float)       # shape: (1, n_features)
intercept = np.array(summary["global_update_intercept"], dtype=float).ravel()

print("\nFedAvg global model:")
print("Coefficient shape:", coef.shape)
print("Intercept:", intercept)

# ----------------------
# Load test set
test_df = pd.read_csv(TEST_CSV)
X_test_df = test_df.drop(columns=[LABEL_COL]).copy()
y_test = test_df[LABEL_COL].astype(int).values

# Encode categorical columns consistently
for c in CAT_COLS:
    le = LabelEncoder()
    X_test_df[c] = le.fit_transform(X_test_df[c].astype(str))

# Scale features
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test_df.values)

# ----------------------
# Compute predictions using linear model
y_pred_scores = X_test @ coef.T + intercept
y_pred = (y_pred_scores > 0.5).astype(int).ravel()  # simple threshold for binary classification

# ----------------------
# Evaluate
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_scores)

print(f"\nMetrics:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# ----------------------
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ----------------------
# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_scores)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# ----------------------
# Distribution of predicted probabilities
plt.figure(figsize=(6,5))
sns.histplot(y_pred_scores.ravel(), bins=50, kde=True)
plt.title("Distribution of Model Output Scores")
plt.xlabel("Predicted score")
plt.ylabel("Count")
plt.show()