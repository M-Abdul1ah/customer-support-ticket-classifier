import joblib
import pandas as pd

from datasets import load_dataset
from sklearn.model_selection import train_test_split


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

X_test_tfidf = vectorizer.transform(X_test)


# ==========================================
# 5. Predictions + confidence
# ==========================================

y_pred = model.predict(X_test_tfidf)

probabilities = model.predict_proba(X_test_tfidf)

confidence = probabilities.max(axis=1)


# ==========================================
# 6. Build results dataframe
# ==========================================

results = pd.DataFrame({
    "actual": y_test.values,
    "predicted": y_pred,
    "confidence": confidence
})


# ==========================================
# 7. Threshold analysis
# ==========================================

thresholds = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.92,
    0.93,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99
]

analysis = []

for threshold in thresholds:

    automated = results[
        results["confidence"] >= threshold
    ]

    review = results[
        results["confidence"] < threshold
    ]

    automated_count = len(automated)
    review_count = len(review)

    if automated_count > 0:
        automation_accuracy = (
            automated["actual"] == automated["predicted"]
        ).mean()
    else:
        automation_accuracy = 0

    automation_rate = (
        automated_count / len(results)
    )

    analysis.append({
        "threshold": threshold,
        "automated_tickets": automated_count,
        "review_tickets": review_count,
        "automation_rate": automation_rate,
        "automation_accuracy": automation_accuracy
    })


analysis_df = pd.DataFrame(analysis)


# ==========================================
# 8. Display results
# ==========================================

print("\n========== CALIBRATED THRESHOLD ANALYSIS ==========")

display_df = analysis_df.copy()

display_df["automation_rate"] = (
    display_df["automation_rate"] * 100
).round(2).astype(str) + "%"

display_df["automation_accuracy"] = (
    display_df["automation_accuracy"] * 100
).round(2).astype(str) + "%"

print(display_df.to_string(index=False))


# ==========================================
# 9. Save results
# ==========================================

output_path = "reports/threshold_analysis_calibrated.csv"

analysis_df.to_csv(
    output_path,
    index=False
)

print(f"\nSaved analysis: {output_path}")

print("\n========== CALIBRATED THRESHOLD ANALYSIS COMPLETE ==========")