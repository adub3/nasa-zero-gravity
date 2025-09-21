import pandas as pd

# EDIT THIS ONE LINE ONLY (use relative or absolute — your call)
CSV_IN  = r"src\data\trainingdat\output_fullid\mergeddata.csv"
CSV_OUT = r"src\data\trainingdat\output_fullid\predictor_only_complete_filled.csv"

EXCLUDE_TYPES = {"earthquake"}  # case-insensitive

# load
df = pd.read_csv(CSV_IN)

# required columns
mandatory = [
    "Disaster Type","Disaster Subtype","Latitude","Longitude",
    "Start Year","Start Month","Start Day","End Year","End Month","End Day",
    "Magnitude","Magnitude Scale",
]

# optional severity columns (kept if present)
severity_candidates = [
    "Total Damage, Adjusted ('000 US$)",
    "Total Damage ('000 US$)",
    "Total Affected",
    "No. Affected",
    "Total Deaths",
]
optional = [c for c in severity_candidates if c in df.columns]

# exclude earthquakes (case-insensitive)
df["Disaster Type"] = df["Disaster Type"].astype(str)
mask_keep_type = ~df["Disaster Type"].str.strip().str.casefold().isin(EXCLUDE_TYPES)
df = df[mask_keep_type]

# keep only rows with all mandatory fields present
filtered = df.loc[df[mandatory].notna().all(axis=1), mandatory + optional].copy()

# fill blanks with zeros
filtered = filtered.fillna(0)

# save
filtered.to_csv(CSV_OUT, index=False)

# distribution
print("📊 disaster type distribution (after removing Earthquake):")
print(filtered["Disaster Type"].value_counts())
print(f"\n✅ wrote: {CSV_OUT} | rows={len(filtered):,}, cols={len(filtered.columns):,}")
