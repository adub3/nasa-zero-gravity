import pandas as pd
from pathlib import Path

FOLDER = Path(r"nasa-zero-gravity\src\data\trainingdat\output_fullid")
OUTPUT = FOLDER / "predictor_only_complete_filled.csv"

df = pd.read_csv(OUTPUT, low_memory=False)

# Clean a bit (strip spaces)
types = df["Disaster Type"].astype(str).str.strip()
subtypes = df["Disaster Subtype"].astype(str).str.strip()

print("✅ Unique Disaster Types:")
print(sorted(types.unique()))

print("\n📊 Counts by Disaster Type:")
print(types.value_counts().sort_values(ascending=False))

print("\n📋 Type × Subtype counts (first 20 columns shown):")
type_sub = pd.crosstab(types, subtypes)
print(type_sub.iloc[:, :20])  # show first 20 subtype columns; remove slice to see all
