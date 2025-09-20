from pathlib import Path
import pandas as pd

FOLDER = Path("nasa-zero-gravity/src/data/trainingdat")
INPUT  = FOLDER / "mergeddat_by_disasterno.csv"      # or mergeddat_cleaned.csv if that's your latest
OUTPUT = FOLDER / "mergeddat_pruned.csv"

# 1) Load
df = pd.read_csv(INPUT, low_memory=False)
print(f"Loaded {INPUT}  rows={len(df):,}  cols={len(df.columns):,}")

# 2) Compute % missing
missing_pct = df.isna().mean() * 100

# 3) Drop columns with ≥95% missing
keep = missing_pct[missing_pct < 80].index
dropped = missing_pct[missing_pct >= 80].sort_values(ascending=False)

df_clean = df[keep]

# 4) Save
df_clean.to_csv(OUTPUT, index=False)
print(f"Saved {OUTPUT}  rows={len(df_clean):,}  cols={len(df_clean.columns):,}")
print("\nDropped columns (≥95% missing):")
print(dropped.round(2).to_string())
