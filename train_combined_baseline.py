from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack


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
# 4. WORD-LEVEL TF-IDF
# ==========================================

print("\n========== WORD TF-IDF ==========")

word_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_word = word_vectorizer.fit_transform(X_train)
X_test_word = word_vectorizer.transform(X_test)

print(f"Word training shape: {X_train_word.shape}")
print(f"Word vocabulary: {len(word_vectorizer.vocabulary_)}")


# ==========================================
# 5. CHARACTER-LEVEL TF-IDF
# ==========================================

print("\n========== CHARACTER TF-IDF ==========")

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_df=0.95
)

X_train_char = char_vectorizer.fit_transform(X_train)
X_test_char = char_vectorizer.transform(X_test)

print(f"Character training shape: {X_train_char.shape}")
print(f"Character vocabulary: {len(char_vectorizer.vocabulary_)}")


# ==========================================
# 6. COMBINE FEATURES
# ==========================================

print("\n========== COMBINING FEATURES ==========")

X_train_combined = hstack([
    X_train_word,
    X_train_char
])

X_test_combined = hstack([
    X_test_word,
    X_test_char
])

print(f"Combined training shape: {X_train_combined.shape}")
print(f"Combined testing shape:  {X_test_combined.shape}")


# ==========================================
# 7. Train Logistic Regression
# ==========================================

print("\n========== MODEL TRAINING ==========")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_combined, y_train)

print("Model training completed!")


# ==========================================
# 8. Predictions
# ==========================================

print("\n========== PREDICTIONS ==========")

y_pred = model.predict(X_test_combined)

print(f"Predictions generated: {len(y_pred)}")


# ==========================================
# 9. Evaluation
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 10. Error Analysis
# ==========================================

errors = X_test[y_test != y_pred]

print("\n========== ERROR ANALYSIS ==========")
print(f"Total errors: {len(errors)}")

print("\n========== SAMPLE ERRORS ==========")

for index in errors.index[:20]:
    print("\nTicket:")
    print(X_test.loc[index])

    print("Actual:   ", y_test.loc[index])
    print("Predicted:", y_pred[list(X_test.index).index(index)])


print("\n========== EXPERIMENT COMPLETE ==========")
print("Model: Word + Character TF-IDF + Logistic Regression")
print(f"Accuracy: {accuracy:.4f}")
print(f"Total test samples: {len(y_test)}")
print(f"Total errors: {len(errors)}")