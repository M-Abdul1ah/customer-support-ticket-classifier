from datasets import load_dataset
import pandas as pd

# ==========================================
# 1. Load dataset
# ==========================================

print("Loading dataset...")

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)

df = dataset["train"].to_pandas()

print("\nDataset loaded successfully!")


# ==========================================
# 2. Basic dataset information
# ==========================================

print("\n========== DATASET SHAPE ==========")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


print("\n========== COLUMNS ==========")
print(df.columns.tolist())


# ==========================================
# 3. Sample records
# ==========================================

print("\n========== FIRST 5 RECORDS ==========")
print(df.head())


# ==========================================
# 4. Data types
# ==========================================

print("\n========== DATA TYPES ==========")
print(df.dtypes)


# ==========================================
# 5. Missing values
# ==========================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# ==========================================
# 6. Duplicate rows
# ==========================================

print("\n========== DUPLICATES ==========")
print(f"Duplicate rows: {df.duplicated().sum()}")


# ==========================================
# 7. Unique values
# ==========================================

print("\n========== UNIQUE VALUES ==========")

for column in df.columns:
    print(f"{column}: {df[column].nunique()} unique values")


# ==========================================
# 8. Category distribution
# ==========================================

if "category" in df.columns:

    print("\n========== CATEGORY DISTRIBUTION ==========")

    print(
        df["category"]
        .value_counts()
        .to_string()
    )


# ==========================================
# 9. Intent distribution
# ==========================================

if "intent" in df.columns:

    print("\n========== INTENT DISTRIBUTION ==========")

    print(
        df["intent"]
        .value_counts()
        .to_string()
    )


# ==========================================
# 10. Instruction/text analysis
# ==========================================

if "instruction" in df.columns:

    df["text_length"] = (
        df["instruction"]
        .astype(str)
        .str.len()
    )

    print("\n========== TEXT LENGTH ==========")

    print(df["text_length"].describe())


# ==========================================
# 11. Category → Intent relationship
# ==========================================

if "category" in df.columns and "intent" in df.columns:

    print("\n========== CATEGORY → INTENT ==========")

    category_intent = (
        df.groupby("category")["intent"]
        .nunique()
        .sort_values(ascending=False)
    )

    print(category_intent)


# ==========================================
# 12. Final summary
# ==========================================

print("\n========== EDA STEP 1 COMPLETE ==========")
print("Dataset is ready for deeper analysis.")