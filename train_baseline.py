from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. Load dataset
# ==========================================

print("Loading dataset...")

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)

df = dataset["train"].to_pandas()

print(f"Dataset loaded: {len(df)} rows")


# ==========================================
# 2. Select features and target
# ==========================================

X = df["instruction"]
y = df["intent"]

print(f"Number of intent classes: {y.nunique()}")


# ==========================================
# 3. Train / Test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ==========================================
# 4. TF-IDF Vectorization
# ==========================================

print("\n========== TF-IDF ==========")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Training matrix shape: {X_train_tfidf.shape}")
print(f"Testing matrix shape:  {X_test_tfidf.shape}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")


# ==========================================
# 5. Train Logistic Regression
# ==========================================

print("\n========== MODEL TRAINING ==========")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# ==========================================
# 6. Generate predictions
# ==========================================

print("\n========== PREDICTIONS ==========")

y_pred = model.predict(X_test_tfidf)

print(f"Predictions generated: {len(y_pred)}")


# ==========================================
# 7. Model evaluation
# ==========================================

print("\n========== MODEL PERFORMANCE ==========")

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# 8. Error analysis
# ==========================================

print("\n========== ERROR ANALYSIS ==========")

errors = df.loc[
    X_test.index,
    ["instruction", "intent"]
].copy()

errors["predicted_intent"] = y_pred

errors = errors[
    errors["intent"] != errors["predicted_intent"]
]

print(f"Total errors: {len(errors)}")


print("\n========== SAMPLE ERRORS ==========")

if len(errors) == 0:

    print("No classification errors found!")

else:

    for _, row in errors.head(20).iterrows():

        print("\nTicket:")
        print(row["instruction"])

        print(f"Actual:    {row['intent']}")
        print(f"Predicted: {row['predicted_intent']}")


# ==========================================
# 9. Baseline complete
# ==========================================

print("\n========== BASELINE COMPLETE ==========")

print("Model: TF-IDF + Logistic Regression")
print(f"Accuracy: {accuracy:.4f}")
print(f"Total test samples: {len(y_test)}")
print(f"Total errors: {len(errors)}")