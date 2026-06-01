import pandas as pd
import random

print("📂 Reading train_snli.txt...")

# File has no header - 3 columns: text1, text2, label
df = pd.read_csv(
    "data/train_snli.txt",
    sep="\t",
    header=None,
    names=["text1", "text2", "label"],
    on_bad_lines="skip"
)

print(f"Total rows loaded: {len(df)}")

# Drop rows with missing values
df = df.dropna()

# Convert label to integer
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# Keep only valid labels (0 and 1)
df = df[df["label"].isin([0, 1])]

print(f"Valid rows: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# Balance dataset - 500 from each class
plagiarized = df[df["label"] == 1].sample(min(500, len(df[df["label"] == 1])), random_state=42)
not_plagiarized = df[df["label"] == 0].sample(min(500, len(df[df["label"] == 0])), random_state=42)

final_df = pd.concat([plagiarized, not_plagiarized]).sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
final_df.to_csv("data/dataset.csv", index=False)

print(f"\n✅ dataset.csv created with {len(final_df)} rows!")
print(f"   Plagiarized     : {final_df['label'].sum()}")
print(f"   Not Plagiarized : {(final_df['label'] == 0).sum()}")