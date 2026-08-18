from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. Load dataset
# ==========================================

print("Loading dataset...")

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)

df = dataset["train"].to_pandas()

print("Dataset loaded successfully!")


# ==========================================
# 2. Create output directory
# ==========================================

import os

os.makedirs("reports/figures", exist_ok=True)


# ==========================================
# 3. Intent distribution
# ==========================================

intent_counts = (
    df["intent"]
    .value_counts()
    .sort_values()
)

plt.figure(figsize=(10, 8))

intent_counts.plot(kind="barh")

plt.title("Intent Distribution")
plt.xlabel("Number of Tickets")
plt.ylabel("Intent")

plt.tight_layout()

plt.savefig(
    "reports/figures/intent_distribution.png",
    dpi=150
)

plt.close()

print("Saved: intent_distribution.png")


# ==========================================
# 4. Category distribution
# ==========================================

category_counts = (
    df["category"]
    .value_counts()
    .sort_values()
)

plt.figure(figsize=(10, 6))

category_counts.plot(kind="barh")

plt.title("Category Distribution")
plt.xlabel("Number of Tickets")
plt.ylabel("Category")

plt.tight_layout()

plt.savefig(
    "reports/figures/category_distribution.png",
    dpi=150
)

plt.close()

print("Saved: category_distribution.png")


# ==========================================
# 5. Instruction length distribution
# ==========================================

df["text_length"] = (
    df["instruction"]
    .astype(str)
    .str.len()
)

plt.figure(figsize=(10, 6))

plt.hist(df["text_length"], bins=30)

plt.title("Customer Ticket Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Number of Tickets")

plt.tight_layout()

plt.savefig(
    "reports/figures/text_length_distribution.png",
    dpi=150
)

plt.close()

print("Saved: text_length_distribution.png")


# ==========================================
# 6. Intent examples
# ==========================================

print("\n========== SAMPLE TICKETS BY INTENT ==========")

sample_intents = [
    "cancel_order",
    "track_order",
    "payment_issue",
    "recover_password",
    "change_shipping_address",
]

for intent in sample_intents:

    samples = df[df["intent"] == intent]["instruction"].head(3)

    print(f"\n--- {intent} ---")

    for text in samples:
        print(f"- {text}")


# ==========================================
# 7. Final summary
# ==========================================

print("\n========== EDA STEP 2 COMPLETE ==========")

print("Figures saved to:")
print("reports/figures/")