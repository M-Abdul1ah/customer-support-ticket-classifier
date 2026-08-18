import json
import joblib
import pandas as pd

from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. Load calibrated model
# ==========================================

print("Loading calibrated model...")

vectorizer = joblib.load(
    "models/tfidf_vectorizer_calibrated.joblib"
)

model = joblib.load(
    "models/classifier_calibrated.joblib"
)

with open("models/metadata_calibrated.json", "r") as f:
    metadata = json.load(f)

print("Calibrated model loaded successfully!")


# ==========================================
# 2. Load dataset
# ==========================================

print("\nLoading dataset...")

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)

df = dataset["train"].to_pandas()


# ==========================================
# 3. Recreate test split
# ==========================================

from sklearn.model_selection import train_test_split

X = df["instruction"]
y = df["intent"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. Transform test data
# ==========================================

print("\nTransforming test data...")

X_test_tfidf = vectorizer.transform(X_test)


# ==========================================
# 5. Predictions
# ==========================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test_tfidf)

probabilities = model.predict_proba(X_test_tfidf)

confidence = probabilities.max(axis=1)


# ==========================================
# 6. Model performance
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== CALIBRATED MODEL PERFORMANCE ==========")

print(f"Accuracy: {accuracy:.4f}")
print(f"Total samples: {len(y_test)}")
print(f"Correct predictions: {(y_pred == y_test).sum()}")
print(f"Incorrect predictions: {(y_pred != y_test).sum()}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 7. Confidence statistics
# ==========================================

confidence_series = pd.Series(confidence)

print("\n========== CONFIDENCE STATISTICS ==========")

print(confidence_series.describe())


# ==========================================
# 8. Error analysis
# ==========================================

results = pd.DataFrame({
    "text": X_test.values,
    "actual": y_test.values,
    "predicted": y_pred,
    "confidence": confidence
})

errors = results[
    results["actual"] != results["predicted"]
].sort_values("confidence")


print("\n========== INCORRECT PREDICTIONS ==========")

print(f"Total errors: {len(errors)}")

if len(errors) > 0:
    print(errors.to_string(index=False))


# ==========================================
# 9. Low-confidence predictions
# ==========================================

low_confidence = results[
    results["confidence"] < 0.50
].sort_values("confidence")


print("\n========== LOW CONFIDENCE PREDICTIONS ==========")

print(
    f"Predictions below 0.50 confidence: "
    f"{len(low_confidence)}"
)

if len(low_confidence) > 0:
    print(low_confidence.to_string(index=False))


# ==========================================
# 10. Save analysis
# ==========================================

output_path = "reports/confidence_analysis_calibrated.csv"

results.to_csv(
    output_path,
    index=False
)

print(f"\nSaved analysis: {output_path}")

print("\n========== CALIBRATED CONFIDENCE ANALYSIS COMPLETE ==========")