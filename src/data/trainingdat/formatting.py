# quick_merge_emdat_left_on_primary.py
# Run from repo root:  python quick_merge_emdat_left_on_primary.py

from pathlib import Path
import pandas as pd

# --- CONFIG ---
FOLDER = Path("nasa-zero-gravity/src/data/trainingdat")
XLSX_TO_CONVERT = FOLDER / "eventtime.xlsx"
OUTPUT_CSV = FOLDER / "mergeddat_by_disasterno.csv"
JOIN_KEY = "disasterno"
PRIMARY_HINT = "eventloc"   # set to None to auto-pick the file with most unique ids

DEDUPE_COLUMNS = True       # drop duplicate non-key columns from later files

def read_any(p: Path) -> pd.DataFrame:
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, dtype=str, low_memory=False)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p, dtype=str)
    raise ValueError(f"Unsupported file type: {p.name}")

def standardize_key(df: pd.DataFrame) -> pd.DataFrame:
    if JOIN_KEY in df.columns:
        return df
    if "DisNo." in df.columns:
        return df.rename(columns={"DisNo.": JOIN_KEY})
    raise ValueError("No join key found (need 'disasterno' or 'DisNo.').")

def main():
    FOLDER.mkdir(parents=True, exist_ok=True)

    # Optional: convert a known xlsx to csv
    if XLSX_TO_CONVERT.exists():
        read_any(XLSX_TO_CONVERT).to_csv(XLSX_TO_CONVERT.with_suffix(".csv"), index=False)
        print(f"[CONVERT] {XLSX_TO_CONVERT.name} -> {XLSX_TO_CONVERT.with_suffix('.csv').name}")

    files = sorted(list(FOLDER.glob("*.csv")) + list(FOLDER.glob("*.xlsx")) + list(FOLDER.glob("*.xls")))
    if not files:
        raise SystemExit(f"[ERROR] No CSV/XLSX found in {FOLDER}")

    # Load + standardize key + drop duplicate ids in each file
    loaded = []
    for p in files:
        try:
            df = standardize_key(read_any(p))
        except ValueError as e:
            print(f"[SKIP] {p.name}: {e}")
            continue
        before = len(df)
        df = df.drop_duplicates(subset=[JOIN_KEY])
        if len(df) != before:
            print(f"[INFO] {p.stem}: dropped {before - len(df)} duplicate ids")
        loaded.append((p.stem, df))

    if not loaded:
        raise SystemExit("[ERROR] No usable files with join key.")

    # Choose primary:
    if PRIMARY_HINT:
        # prefer file whose stem contains PRIMARY_HINT; else fall back to auto
        prim_tuple = next(((n, d) for (n, d) in loaded if PRIMARY_HINT.lower() in n.lower()), None)
        if prim_tuple is None:
            print(f"[WARN] PRIMARY_HINT='{PRIMARY_HINT}' not found; picking by largest unique IDs")
    else:
        prim_tuple = None

    if prim_tuple is None:
        # auto-pick file with most unique ids
        prim_tuple = max(loaded, key=lambda t: t[1][JOIN_KEY].nunique())

    primary_name, merged = prim_tuple
    print(f"[PRIMARY] {primary_name}: rows={len(merged):,}, cols={len(merged.columns):,}")

    # Merge all others with LEFT JOIN on primary
    for name, d in loaded:
        if name == primary_name:
            continue
        d2 = d
        if DEDUPE_COLUMNS:
            # Drop non-key columns already present to avoid duplicates
            drop_already = [c for c in d2.columns if c != JOIN_KEY and c in merged.columns]
            if drop_already:
                d2 = d2.drop(columns=drop_already)
        # Compute match rate before merging
        base_ids = set(merged[JOIN_KEY])
        match_ids = base_ids.intersection(set(d2[JOIN_KEY]))
        match_rate = (len(match_ids) / len(base_ids)) * 100 if base_ids else 0.0

        prev_rows, prev_cols = len(merged), len(merged.columns)
        merged = merged.merge(d2, on=JOIN_KEY, how="left")
        print(f"[MERGE] + {name:>20s} | match={match_rate:5.1f}% | "
              f"rows={prev_rows:,}->{len(merged):,} | cols={prev_cols}->{len(merged.columns)}")

    merged = merged.sort_values(JOIN_KEY, kind="stable")
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"[DONE] wrote {OUTPUT_CSV} | rows={len(merged):,} cols={len(merged.columns):,}")

if __name__ == "__main__":
    main()
