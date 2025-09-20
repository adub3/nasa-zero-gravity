from pathlib import Path
import pandas as pd

# Paths
FOLDER = Path(r"nasa-zero-gravity\src\data\trainingdat\output_fullid")
MERGED = FOLDER / "merged_eventtime_with_loc.LEFT_fullID_20250920-142002.csv"
OUTPUT = FOLDER / "predictor_only_complete_filled.csv"

# Load
df = pd.read_csv(MERGED, low_memory=False)

# Mandatory predictor columns (must be present & non-null)
mandatory_cols = [
    "Disaster Type",
    "Disaster Subtype",
    "Latitude",
    "Longitude",
    "Start Year",
    "Start Month",
    "Start Day",
    "End Year",
    "End Month",
    "End Day",
    "Magnitude",
    "Magnitude Scale",
]

# Optional severity columns (include if they exist, but may be blank)
severity_candidates = [
    "Total Damage, Adjusted ('000 US$)",
    "Total Damage ('000 US$)",
    "Total Affected",
    "No. Affected",
    "Total Deaths",
]
optional_severity_cols = [c for c in severity_candidates if c in df.columns]

# Check mandatory columns exist
missing = [c for c in mandatory_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing mandatory columns: {missing}")

# Filter rows where all mandatory predictors are present
cols_to_keep = mandatory_cols + optional_severity_cols
filtered = df.loc[df[mandatory_cols].notna().all(axis=1), cols_to_keep].copy()

# Fill ALL remaining NaNs (including severity blanks) with 0
filtered = filtered.fillna(0)

# Save
filtered.to_csv(OUTPUT, index=False)
print(f"[DONE] wrote {OUTPUT} | rows={len(filtered):,} cols={len(filtered.columns):,}")
