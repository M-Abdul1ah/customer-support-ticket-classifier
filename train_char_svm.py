from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
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
# 4. Character TF-IDF
# ==========================================

print("\n========== CHARACTER TF-IDF ==========")

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_df=0.95
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Training matrix shape: {X_train_tfidf.shape}")
print(f"Testing matrix shape:  {X_test_tfidf.shape}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")


# ==========================================
# 5. Train Linear SVM
# ==========================================

print("\n========== MODEL TRAINING ==========")

model = LinearSVC(
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# ==========================================
# 6. Predictions
# ==========================================

print("\n========== PREDICTIONS ==========")

y_pred = model.predict(X_test_tfidf)

print(f"Predictions generated: {len(y_pred)}")


# ==========================================
# 7. Evaluation
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 8. Error Analysis
# ==========================================

errors = X_test[y_test != y_pred]

print("\n========== ERROR ANALYSIS ==========")

print(f"Total errors: {len(errors)}")

print("\n========== SAMPLE ERRORS ==========")

for index in errors.index[:20]:

    position = X_test.index.get_loc(index)

    print("\nTicket:")
    print(X_test.loc[index])

    print("Actual:   ", y_test.loc[index])
    print("Predicted:", y_pred[position])


# ==========================================
# 9. Experiment Summary
# ==========================================

print("\n========== EXPERIMENT COMPLETE ==========")

print("Model: Character TF-IDF + Linear SVM")
print(f"Accuracy: {accuracy:.4f}")
print(f"Total test samples: {len(y_test)}")
print(f"Total errors: {len(errors)}")
