"""
Phase 3 -- Error Analysis

Loads the same train/test split used in train_calibrated.py (same
random_state and stratify), re-runs the saved calibrated model on the
held-out test set, and surfaces misclassified examples grouped by
(actual intent -> predicted intent) so we can see WHERE the model
gets confused, not just its overall accuracy.

This does not retrain anything -- it only evaluates the model that
is already saved in models/.
"""

import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Load data (same split as training)
# ==========================================
print("Loading dataset...")
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
df = dataset["train"].to_pandas()

X = df["instruction"]
y = df["intent"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Test set size: {len(X_test)}")

# ==========================================
# 2. Load saved model + vectorizer
# ==========================================
print("Loading saved model...")
model = joblib.load("models/classifier_calibrated.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer_calibrated.joblib")

# ==========================================
# 3. Predict on the held-out test set
# ==========================================
X_test_tfidf = vectorizer.transform(X_test)
predictions = model.predict(X_test_tfidf)

results = pd.DataFrame({
    "text": X_test.values,
    "actual_intent": y_test.values,
    "predicted_intent": predictions,
})

errors = results[results["actual_intent"] != results["predicted_intent"]].copy()

print(f"\nTotal test examples: {len(results)}")
print(f"Total misclassified: {len(errors)}")
print(f"Error rate: {len(errors) / len(results):.4%}")

# ==========================================
# 4. Group errors by (actual -> predicted) pair
# ==========================================
print("\n========== TOP CONFUSION PAIRS ==========")
confusion_pairs = (
    errors.groupby(["actual_intent", "predicted_intent"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
print(confusion_pairs.head(20).to_string(index=False))

# ==========================================
# 5. Show a few example misclassifications per top pair
# ==========================================
print("\n========== SAMPLE MISCLASSIFIED EXAMPLES ==========")
top_pairs = confusion_pairs.head(5)
for _, row in top_pairs.iterrows():
    actual = row["actual_intent"]
    predicted = row["predicted_intent"]
    examples = errors[
        (errors["actual_intent"] == actual) & (errors["predicted_intent"] == predicted)
    ]["text"].head(3)
    print(f"\nActual: {actual}  ->  Predicted: {predicted}  ({row['count']} cases)")
    for text in examples:
        print(f"   - {text}")

# ==========================================
# 6. Save full error log for later reference
# ==========================================
errors.to_csv("notebooks/error_analysis_results.csv", index=False)
print("\nSaved full error log to notebooks/error_analysis_results.csv")
