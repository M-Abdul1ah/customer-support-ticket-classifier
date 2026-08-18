from pathlib import Path
import json

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "classifier.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading dataset...")

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)

df = dataset["train"].to_pandas()

print(f"Dataset loaded: {len(df)} rows")


# ==========================================
# 2. FEATURES AND TARGET
# ==========================================

X = df["instruction"]
y = df["intent"]

print(f"Number of intent classes: {y.nunique()}")


# ==========================================
# 3. TRAIN / TEST SPLIT
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
# 4. CHARACTER TF-IDF
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
# 5. TRAIN LINEAR SVM
# ==========================================

print("\n========== MODEL TRAINING ==========")

model = LinearSVC(
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# ==========================================
# 6. EVALUATION
# ==========================================

print("\n========== EVALUATION ==========")

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 7. SAVE MODEL
# ==========================================

print("\n========== SAVING MODEL ==========")

joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump(model, MODEL_PATH)

print(f"Vectorizer saved: {VECTORIZER_PATH}")
print(f"Model saved:      {MODEL_PATH}")


# ==========================================
# 8. SAVE METADATA
# ==========================================

metadata = {
    "model_type": "Character TF-IDF + Linear SVM",
    "accuracy": round(float(accuracy), 4),
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "num_classes": int(y.nunique()),
    "vectorizer": {
        "analyzer": "char",
        "ngram_range": [3, 5],
        "min_df": 2,
        "max_df": 0.95,
        "vocabulary_size": len(vectorizer.vocabulary_)
    },
    "classifier": {
        "type": "LinearSVC",
        "random_state": 42
    }
}

with open(METADATA_PATH, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print(f"Metadata saved:   {METADATA_PATH}")


# ==========================================
# 9. FINAL SUMMARY
# ==========================================

print("\n========== TRAINING COMPLETE ==========")

print(f"Model: Character TF-IDF + Linear SVM")
print(f"Accuracy: {accuracy:.4f}")
print(f"Classes: {y.nunique()}")

print("\nSaved artifacts:")
print("- tfidf_vectorizer.joblib")
print("- classifier.joblib")
print("- metadata.json")