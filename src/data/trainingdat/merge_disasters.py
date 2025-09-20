import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime

# ========= USER PATHS =========
# Excel source to convert -> CSV
XLSX_EVENTTIME = Path(r"C:\Users\zirun\code\Zero Gravity\nasa-zero-gravity\src\data\trainingdat\eventtime.xlsx")

# Data folder containing eventloc.csv (has disasterno, iso3)
ROOT = Path(r"C:\Users\zirun\code\Zero Gravity\nasa-zero-gravity\src\data\trainingdat")
PATH_EVENTLOC = ROOT / "eventloc.csv"   # must contain columns: disasterno, iso3

# Output folder
OUTDIR = ROOT / "output_fullid"
OUTDIR.mkdir(parents=True, exist_ok=True)
# ==============================


def normalize_key(s: pd.Series) -> pd.Series:
    """Uppercase, trim, collapse spaces, and map common empties to NA."""
    s = s.astype("string")
    s = s.str.strip().str.upper()
    s = s.str.replace(r"\s+", "", regex=True)
    s = s.replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "NONE": pd.NA})
    return s


def normalize_iso3(s: pd.Series) -> pd.Series:
    """Uppercase ISO3; keep only letters A-Z (drop digits/punct); map empties to NA."""
    s = s.astype("string").str.strip().str.upper()
    s = s.str.replace(r"[^A-Z]", "", regex=True)
    s = s.replace({"": pd.NA})
    return s


def make_full_id(disno_core: pd.Series, iso3: pd.Series) -> pd.Series:
    """
    Build full ID like '2009-0631-ALB' from:
      disno_core == '2009-0631'
      iso3       == 'ALB'
    Only create when BOTH parts are non-null; otherwise NA.
    """
    core = normalize_key(disno_core)
    code = normalize_iso3(iso3)
    full = pd.Series(pd.NA, index=core.index, dtype="string")
    mask = core.notna() & code.notna()
    full.loc[mask] = core[mask] + "-" + code[mask]
    return full


def pct_empty(df: pd.DataFrame) -> float:
    total = df.size
    empties = df.isna().sum().sum()
    return 100.0 * empties / total if total else 0.0


def pct_empty_per_col(df: pd.DataFrame) -> pd.Series:
    return df.isna().mean().mul(100.0).sort_values(ascending=False)


def main():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ---- 1) Convert XLSX -> CSV (timestamped) ----
    eventtime_csv_path = ROOT / f"eventtime_from_xlsx_{stamp}.csv"
    df_time = pd.read_excel(XLSX_EVENTTIME, dtype="object")  # first sheet by default
    df_time.to_csv(eventtime_csv_path, index=False)
    print(f"Converted XLSX to CSV: {eventtime_csv_path}")

    # ---- 2) Load CSVs ----
    df_loc  = pd.read_csv(PATH_EVENTLOC, dtype="object", keep_default_na=True, na_values=["", "NA", "N/A"])
    df_time = pd.read_csv(eventtime_csv_path, dtype="object", keep_default_na=True, na_values=["", "NA", "N/A"])

    # Preserve original eventtime for metrics
    df_time_orig = df_time.copy()

    # ---- 3) Validate columns ----
    required_loc_cols = {"disasterno", "iso3"}
    missing = required_loc_cols - set(map(str.lower, df_loc.columns))
    # Try case-insensitive recovery
    cols_lower = {c.lower(): c for c in df_loc.columns}
    if missing:
        raise KeyError(f"Missing required columns in eventloc.csv: {missing}. Found: {list(df_loc.columns)}")
    col_disasterno = cols_lower["disasterno"]
    col_iso3 = cols_lower["iso3"]

    # eventtime join col (DisNo. or DisNo)
    join_col_time_candidates = [c for c in df_time.columns if c.lower().startswith("disno")]
    if not join_col_time_candidates:
        raise KeyError(f"No join key like 'DisNo.' / 'DisNo' found in {eventtime_csv_path}. Found: {list(df_time.columns)}")
    col_disno_time = join_col_time_candidates[0]

    # ---- 4) Build FULL IDs on both sides ----
    # eventloc: full id = normalize(disasterno) + "-" + normalize(iso3)
    df_loc["__disasterno_norm__"] = normalize_key(df_loc[col_disasterno])
    df_loc["__iso3_norm__"]       = normalize_iso3(df_loc[col_iso3])
    df_loc["__full_id__"]         = make_full_id(df_loc["__disasterno_norm__"], df_loc["__iso3_norm__"])
    df_loc_filtered = df_loc[~df_loc["__full_id__"].isna()].copy()

    # eventtime: normalize DisNo (already includes '-ISO3' on your side)
    df_time["__disno_norm__"] = normalize_key(df_time[col_disno_time])

    # ---- 5) De-duplicate eventloc on FULL ID to prevent one-to-many joins ----
    # Duplicate diagnostics
    dup_counts = (
        df_loc_filtered.groupby("__full_id__", dropna=True)
        .size()
        .rename("rows_per_key")
        .reset_index()
    )
    dups = dup_counts[dup_counts["rows_per_key"] > 1]
    n_dup_keys = len(dups)
    n_dup_rows = int(dups["rows_per_key"].sum())

    dup_diag_path = OUTDIR / f"eventloc_duplicate_fullids_{stamp}.csv"
    if n_dup_keys > 0:
        dup_rows = df_loc_filtered.merge(dups[["__full_id__"]], on="__full_id__", how="inner")
        dup_rows.to_csv(dup_diag_path, index=False)

    # Choose 1 best row per full id (most non-null fields)
    df_loc_filtered["__nonnull_score__"] = df_loc_filtered.notna().sum(axis=1)
    best_idx = (
        df_loc_filtered.groupby("__full_id__", dropna=True)["__nonnull_score__"]
        .idxmax()
        .dropna()
        .astype(int)
    )
    df_loc_best = df_loc_filtered.loc[best_idx].copy()
    df_loc_best.drop(columns=["__nonnull_score__"], errors="ignore", inplace=True)

    # ---- 6) Joins (on FULL ID) ----
    left_merge = df_time.merge(
        df_loc_best,
        left_on="__disno_norm__",
        right_on="__full_id__",
        how="left",
        suffixes=("", "_loc"),
    )
    inner_merge = df_time.merge(
        df_loc_best,
        left_on="__disno_norm__",
        right_on="__full_id__",
        how="inner",
        suffixes=("", "_loc"),
    )

    # Safety: LEFT must preserve row count
    assert left_merge.shape[0] == df_time_orig.shape[0], (
        f"LEFT merge row count {left_merge.shape[0]} != eventtime row count {df_time_orig.shape[0]}"
    )

    # ---- 7) Emptiness metrics ----
    overall_empty_time  = pct_empty(df_time_orig)
    overall_empty_left  = pct_empty(left_merge)
    overall_empty_inner = pct_empty(inner_merge)

    percol_empty_time  = pct_empty_per_col(df_time_orig).rename("eventtime_%empty")
    percol_empty_left  = pct_empty_per_col(left_merge).rename("leftmerge_%empty")
    percol_empty_inner = pct_empty_per_col(inner_merge).rename("innermerge_%empty")
    emptiness_compare = pd.concat([percol_empty_time, percol_empty_left, percol_empty_inner], axis=1)

    # ---- 8) Outputs ----
    OUT_LEFT   = OUTDIR / f"merged_eventtime_with_loc.LEFT_fullID_{stamp}.csv"
    OUT_INNER  = OUTDIR / f"merged_eventtime_with_loc.INNER_fullID_{stamp}.csv"
    OUT_EMPTY  = OUTDIR / f"emptiness_compare_fullID_{stamp}.csv"
    OUT_REPORT = OUTDIR / f"merge_report_fullID_{stamp}.txt"

    left_merge.to_csv(OUT_LEFT, index=False)
    inner_merge.to_csv(OUT_INNER, index=False)
    emptiness_compare.to_csv(OUT_EMPTY)

    # ---- 9) Report ----
    set_loc_full  = set(df_loc_best["__full_id__"].dropna().unique())
    set_time_full = set(df_time["__disno_norm__"].dropna().unique())
    matched_full  = set_loc_full & set_time_full
    only_loc_full = set_loc_full - set_time_full
    only_time_full= set_time_full - set_loc_full

    report_lines = []
    report_lines.append("=== Merge Debug Report (FULL ID: YYYY-####-ISO3) ===")
    report_lines.append(f"eventloc file      : {PATH_EVENTLOC}")
    report_lines.append(f"eventtime (XLSX)   : {XLSX_EVENTTIME}")
    report_lines.append(f"eventtime (CSV out): {eventtime_csv_path}")
    report_lines.append("")
    report_lines.append("--- Key Columns ---")
    report_lines.append(f"eventtime join col   : {col_disno_time} (normalized to __disno_norm__)")
    report_lines.append("eventloc FULL ID     : __full_id__ = normalize(disasterno) + '-' + normalize(iso3)")
    report_lines.append("")
    report_lines.append("--- Shapes (rows, cols) ---")
    report_lines.append(f"eventtime shape : {df_time_orig.shape}")
    report_lines.append(f"LEFT merge shape: {left_merge.shape}  (rows should equal eventtime)")
    report_lines.append(f"INNER merge shape: {inner_merge.shape}")
    report_lines.append("")
    report_lines.append("--- eventloc duplicates before de-dup (by FULL ID) ---")
    report_lines.append(f"Keys with >1 rows: {n_dup_keys}")
    report_lines.append(f"Total duplicate rows across those keys: {n_dup_rows}")
    if n_dup_keys > 0:
        report_lines.append(f"Duplicate detail written to: {dup_diag_path}")
    report_lines.append("")
    report_lines.append("--- FULL ID Overlap ---")
    report_lines.append(f"Unique FULL IDs in eventloc : {len(set_loc_full)}")
    report_lines.append(f"Unique FULL IDs in eventtime: {len(set_time_full)}")
    report_lines.append(f"Matched FULL IDs            : {len(matched_full)}")
    report_lines.append(f"Only in eventloc            : {len(only_loc_full)}")
    report_lines.append(f"Only in eventtime           : {len(only_time_full)}")
    if len(set_time_full):
        report_lines.append(f"% of eventtime FULL IDs matched: {100.0*len(matched_full)/len(set_time_full):0.2f}%")
    if len(set_loc_full):
        report_lines.append(f"% of eventloc FULL IDs matched : {100.0*len(matched_full)/len(set_loc_full):0.2f}%")
    report_lines.append("")
    report_lines.append("--- Overall % Empty Cells ---")
    report_lines.append(f"Original eventtime : {overall_empty_time:0.2f}%")
    report_lines.append(f"LEFT-merged output : {overall_empty_left:0.2f}%")
    report_lines.append(f"INNER-merged output: {overall_empty_inner:0.2f}%")
    report_lines.append("")
    report_lines.append("--- Unmatched samples (first 10 each) ---")
    report_lines.append("Only in eventtime (not in loc): " + ", ".join(sorted(list(only_time_full))[:10]))
    report_lines.append("Only in eventloc  (not in time): " + ", ".join(sorted(list(only_loc_full))[:10]))

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("Wrote files:")
    print(f"- Converted CSV    : {eventtime_csv_path}")
    print(f"- LEFT  merge      : {OUT_LEFT}")
    print(f"- INNER merge      : {OUT_INNER}")
    print(f"- Emptiness report : {OUT_EMPTY}")
    print(f"- Debug report     : {OUT_REPORT}")


if __name__ == "__main__":
    main()
