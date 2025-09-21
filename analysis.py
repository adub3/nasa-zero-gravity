#!/usr/bin/env python3
"""
Evaluation & checkpoint sweep on real CSV events using your full fetch pipeline.

- Loads each checkpoint, rebuilds MultiTaskUNet3D
- Creates one shared evaluation set from CSV (class-balanced & label-aligned)
- Fetches chips -> builds tensor -> per-sample normalization (cached across checkpoints)
- Predicts type/subtype, aggregates accuracy, and plots accuracy vs. checkpoint step

Update CKPT_GLOB and CSV_PATH at the bottom before running.
"""

import os
import re
import glob
import math
import traceback
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Your data fetchers
from src.data.imagesample import fetch_multisource_chips, to_model_tensor


# ----------------------------
# Small helpers
# ----------------------------
def _int_or_none(x):
    try:
        return int(float(x))
    except Exception:
        return None

def _mk_date_str(y, m, d):
    y = _int_or_none(y); m = _int_or_none(m); d = _int_or_none(d)
    if not (y and m and d):
        return None
    m = max(1, min(12, m))
    d = max(1, min(28, d))  # safe day bound
    return f"{y:04d}-{m:02d}-{d:02d}"

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _norm_label(s):
    return str(s).strip() if s is not None else ""


# ----------------------------
# Model (same as your training)
# ----------------------------
class MultiTaskUNet3D(torch.nn.Module):
    """3D UNet with spatial and categorical outputs."""
    def __init__(self, in_ch: int, n_disaster_types: int, n_disaster_subtypes: int,
                 base: int = 16, dropout: float = 0.1):
        super().__init__()

        def block(cin, cout):
            return torch.nn.Sequential(
                torch.nn.Conv3d(cin, cout, 3, padding=1),
                torch.nn.BatchNorm3d(cout),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv3d(cout, cout, 3, padding=1),
                torch.nn.BatchNorm3d(cout),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout3d(dropout)
            )

        # Encoder
        self.enc1 = block(in_ch, base)
        self.pool1 = torch.nn.MaxPool3d((1, 2, 2))
        self.enc2 = block(base, base * 2)
        self.pool2 = torch.nn.MaxPool3d((1, 2, 2))
        self.enc3 = block(base * 2, base * 4)
        self.pool3 = torch.nn.MaxPool3d((1, 2, 2))

        # Bottleneck
        self.bottleneck = block(base * 4, base * 8)

        # Decoder
        self.up3 = torch.nn.ConvTranspose3d(base * 8, base * 4, (1, 2, 2), stride=(1, 2, 2))
        self.dec3 = block(base * 8, base * 4)
        self.up2 = torch.nn.ConvTranspose3d(base * 4, base * 2, (1, 2, 2), stride=(1, 2, 2))
        self.dec2 = block(base * 4, base * 2)
        self.up1 = torch.nn.ConvTranspose3d(base * 2, base, (1, 2, 2), stride=(1, 2, 2))
        self.dec1 = block(base * 2, base)

        # Output heads
        self.spatial_head = torch.nn.Conv3d(base, 1, 1)

        # Global pooling for categorical classification
        self.global_pool = torch.nn.AdaptiveAvgPool3d(1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(base * 8, base * 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(base * 4, base * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout)
        )
        self.disaster_type_head = torch.nn.Linear(base * 2, n_disaster_types)
        self.disaster_subtype_head = torch.nn.Linear(base * 2, n_disaster_subtypes)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)

        spatial_out = self.spatial_head(d1)
        global_feat = self.global_pool(b).flatten(1)
        cls_feat    = self.classifier(global_feat)
        type_out    = self.disaster_type_head(cls_feat)
        sub_out     = self.disaster_subtype_head(cls_feat)
        return {"spatial": spatial_out, "disaster_type": type_out, "disaster_subtype": sub_out}


# ----------------------------
# Checkpoint loaders & label alignment
# ----------------------------
def read_ckpt_labels_only(ckpt_path):
    """Fast read of label sets from checkpoint (no model build)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    types = ckpt.get("disaster_types", ckpt.get("n_disaster_types"))
    subs  = ckpt.get("disaster_subtypes", ckpt.get("n_disaster_subtypes"))
    if isinstance(types, int):  # old style (counts only)
        types = [f"Type_{i}" for i in range(types)]
    if isinstance(subs, int):
        subs = [f"Subtype_{i}" for i in range(subs)]
    types = [_norm_label(t) for t in (types or [])]
    subs  = [_norm_label(s) for s in (subs or [])]
    # also get input channels to catch channel-mismatch early
    in_ch = None
    for k, v in ckpt["model"].items():
        if "enc1.0.weight" in k and v.ndim == 5:
            in_ch = int(v.shape[1])
            break
    return types, subs, in_ch

def load_checkpoint_and_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"]
    base = ckpt.get("base", 32)
    dropout = ckpt.get("dropout", 0.15)

    # class lists (newer checkpoints) or fallbacks
    disaster_types = ckpt.get("disaster_types", ckpt.get("n_disaster_types"))
    disaster_subtypes = ckpt.get("disaster_subtypes", ckpt.get("n_disaster_subtypes"))
    if isinstance(disaster_types, int):
        disaster_types = [f"Type_{i}" for i in range(disaster_types)]
    if isinstance(disaster_subtypes, int):
        disaster_subtypes = [f"Subtype_{i}" for i in range(disaster_subtypes)]
    disaster_types  = [_norm_label(t) for t in (disaster_types or [])]
    disaster_subtypes = [_norm_label(s) for s in (disaster_subtypes or [])]

    # infer input channels from first conv weight
    in_ch = None
    first_weight_key = None
    for k, v in state.items():
        if "enc1.0.weight" in k and v.ndim == 5:
            in_ch = v.shape[1]
            first_weight_key = k
            break
    if in_ch is None:
        raise RuntimeError("Could not infer input channels from checkpoint.")

    n_type = state["disaster_type_head.weight"].shape[0]
    n_sub  = state["disaster_subtype_head.weight"].shape[0]

    print("📊 Model Information:")
    print(f"  Base channels: {base}")
    print(f"  Dropout: {dropout}")
    print(f"  Disaster types: {disaster_types}")
    print(f"  Disaster subtypes: {disaster_subtypes}")
    print(f"  Found input layer: {first_weight_key} -> in_ch={in_ch}")
    print(f"  Output classes: {n_type} types, {n_sub} subtypes")

    model = MultiTaskUNet3D(in_ch=int(in_ch), n_disaster_types=n_type, n_disaster_subtypes=n_sub,
                            base=base, dropout=dropout).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, disaster_types, disaster_subtypes, int(in_ch)


# ----------------------------
# Evaluation core (rows shared across checkpoints; tensors cached)
# ----------------------------
def build_shared_eval_rows(csv_path, target_types, n_per_class=10, seed=42):
    """Build one shared, balanced eval set from CSV using target_types (label-aligned)."""
    df = pd.read_csv(csv_path, low_memory=False)

    # Normalize labels
    if "Disaster Type" in df.columns:
        df["Disaster Type"] = df["Disaster Type"].map(_norm_label)
    if "Disaster Subtype" in df.columns:
        df["Disaster Subtype"] = df["Disaster Subtype"].map(_norm_label)

    # Coerce ints for dates
    for c in ["Start Year","Start Month","Start Day","End Year","End Month","End Day"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Valid rows
    def _valid_row(r):
        lat = _safe_float(r.get("Latitude"))
        lon = _safe_float(r.get("Longitude"))
        t0  = _mk_date_str(r.get("Start Year"), r.get("Start Month"), r.get("Start Day"))
        return (lat is not None and -90<=lat<=90 and
                lon is not None and -180<=lon<=180 and
                t0 is not None)

    df_valid = df[df.apply(_valid_row, axis=1)]
    if len(df_valid) == 0:
        raise RuntimeError("No valid rows in CSV (lat/lon/date missing).")

    # If target_types is empty, just sample globally
    if not target_types:
        return df_valid.sample(min(n_per_class*2, len(df_valid)), random_state=seed).to_dict("records")

    rows = []
    for cls in sorted(target_types):
        take = df_valid[df_valid["Disaster Type"] == cls]
        if len(take) == 0:
            continue
        rows.extend(take.sample(min(n_per_class, len(take)), random_state=seed).to_dict("records"))

    # If nothing matched, fall back to global sample
    if not rows:
        rows = df_valid.sample(min(n_per_class*2, len(df_valid)), random_state=seed).to_dict("records")
    return rows


def build_tensor_for_row(r, lookbacks, chip_size_m, resolution_m, max_cloud, era_windows, device):
    """Fetch chips, build model tensor, return (C,T,H,W) np array or None on failure/empty."""
    lat = float(r["Latitude"]); lon = float(r["Longitude"])
    t0  = _mk_date_str(r.get("Start Year"), r.get("Start Month"), r.get("Start Day"))

    chips = fetch_multisource_chips(
        lat=lat, lon=lon, t0=t0, lookbacks=list(lookbacks),
        chip_size_m=chip_size_m, resolution_m=resolution_m,
        max_cloud=max_cloud, era_windows=list(era_windows),
        include_s2=True, include_dem=True, include_worldcover=True,
        include_era5=True, include_smap=True
    )
    X_np, _ = to_model_tensor(chips)  # [T,C,H,W]
    if (X_np != 0).sum() == 0:
        return None  # all-zero / cloudy

    # Per-sample z-norm & reshape to [C,T,H,W]
    mu = X_np.mean(axis=(0,2,3), keepdims=True).astype(np.float32)
    sd = X_np.std(axis=(0,2,3), keepdims=True).astype(np.float32) + 1e-6
    Xz = (X_np - mu) / sd
    Xcthw = np.transpose(Xz, (1,0,2,3))
    return Xcthw


def evaluate_one_checkpoint_on_rows(ckpt_path, rows, label_intersection,
                                    lookbacks, chip_size_m, resolution_m, max_cloud, era_windows,
                                    device, tensor_cache, strict_channels=True):
    """
    Evaluate a single checkpoint on a fixed set of rows.
    label_intersection: set of disaster types to be scored.
    tensor_cache: dict key=(lat,lon,t0,lookbacks) -> np.ndarray [C,T,H,W] or None
    """
    try:
        model, types_list, subtypes_list, in_ch = load_checkpoint_and_model(ckpt_path, device)
    except Exception as e:
        return {"ckpt": ckpt_path, "ok": False, "error": f"load model failed: {e}"}

    # Metrics
    type_hits = sub_hits = 0
    type_total = sub_total = 0

    # Evaluate
    for i, r in enumerate(rows, 1):
        lat = float(r["Latitude"]); lon = float(r["Longitude"])
        t0  = _mk_date_str(r.get("Start Year"), r.get("Start Month"), r.get("Start Day"))
        key = (round(lat, 5), round(lon, 5), t0, tuple(lookbacks), chip_size_m, resolution_m, max_cloud, tuple(era_windows))

        # Obtain tensor from cache or build
        if key in tensor_cache:
            Xcthw = tensor_cache[key]
        else:
            try:
                Xcthw = build_tensor_for_row(r, lookbacks, chip_size_m, resolution_m, max_cloud, era_windows, device)
            except Exception:
                Xcthw = None
            tensor_cache[key] = Xcthw

        if Xcthw is None:
            continue  # skip empty/failed

        # Channel alignment check
        C = int(Xcthw.shape[0])
        if C != in_ch:
            if strict_channels:
                # Skip this row for this model (mismatch)
                continue
            # Optional: pad/truncate (not recommended)
            if C < in_ch:
                pad = np.zeros((in_ch - C, *Xcthw.shape[1:]), dtype=Xcthw.dtype)
                Xcthw = np.concatenate([Xcthw, pad], axis=0)
            else:
                Xcthw = Xcthw[:in_ch]

        # Ground truth (normalized)
        true_type = _norm_label(r.get("Disaster Type", "Unknown"))
        true_sub  = _norm_label(r.get("Disaster Subtype", "Unknown"))

        score_type = (true_type in (types_list or [])) and (true_type in label_intersection)
        score_sub  = (true_sub in (subtypes_list or []))  # subtype intersection not enforced globally

        # Forward
        X_t = torch.from_numpy(Xcthw).unsqueeze(0).to(device)  # [1,C,T,H,W]
        with torch.no_grad():
            out = model(X_t)
            p_type = torch.softmax(out["disaster_type"], dim=1)[0].cpu().numpy()
            p_sub  = torch.softmax(out["disaster_subtype"], dim=1)[0].cpu().numpy()
            type_idx = int(np.argmax(p_type))
            sub_idx  = int(np.argmax(p_sub))

        # Accumulate metrics
        if score_type:
            type_total += 1
            if types_list[type_idx] == true_type:
                type_hits += 1
        if score_sub:
            sub_total += 1
            if subtypes_list[sub_idx] == true_sub:
                sub_hits += 1

    type_acc = (type_hits / type_total) if type_total > 0 else float("nan")
    sub_acc  = (sub_hits  / sub_total) if sub_total > 0 else float("nan")
    return {
        "ckpt": ckpt_path,
        "ok": True,
        "type_acc": type_acc, "type_n": type_total,
        "sub_acc":  sub_acc,  "sub_n":  sub_total,
        "in_ch": in_ch
    }


# ----------------------------
# Sweep across checkpoints (shared rows & cache)
# ----------------------------
def sweep_checkpoints_and_plot(ckpt_glob_pattern,
                               csv_path,
                               lookbacks=(1,5,10,20),
                               chip_size_m=2560, resolution_m=10,
                               max_cloud=70, era_windows=(1,3,7,14,30),
                               n_per_class=10,
                               out_png="test_results/accuracy_sweep.png"):
    """
    - Gathers all matching checkpoints
    - Computes the common set of disaster types present in BOTH CSV and ALL checkpoints
    - Builds a shared eval set balanced over those types (up to n_per_class each)
    - Evaluates each checkpoint on the same rows (re-using cached tensors)
    - Plots accuracy vs. checkpoint suffix (e.g., tmp_50, tmp_100, ...)
    """
    ckpts = sorted(glob.glob(ckpt_glob_pattern))
    if not ckpts:
        print(f"No checkpoints matched: {ckpt_glob_pattern}")
        return

    # Label sets from CSV
    df_all = pd.read_csv(csv_path, low_memory=False)
    df_all["Disaster Type"] = df_all["Disaster Type"].map(_norm_label) if "Disaster Type" in df_all.columns else ""
    csv_types = set(df_all["Disaster Type"].dropna().unique().tolist()) if "Disaster Type" in df_all.columns else set()

    # Label sets per checkpoint
    ckpt_types_list = []
    ckpt_inch = {}
    for p in ckpts:
        try:
            tlist, slist, in_ch = read_ckpt_labels_only(p)
            ckpt_types_list.append(set(tlist or []))
            ckpt_inch[p] = in_ch
        except Exception:
            ckpt_types_list.append(set())
            ckpt_inch[p] = None

    # Intersection across CSV and ALL checkpoints
    common_types = set(csv_types)
    for s in ckpt_types_list:
        common_types &= s

    if not common_types:
        # Fall back to first checkpoint ∩ CSV
        common_types = (ckpt_types_list[0] & csv_types) if ckpt_types_list else set()

    print("🔎 CSV types (count={}): {}".format(len(csv_types), sorted(list(csv_types))[:10]))
    print("🔎 Common types used for evaluation:", sorted(list(common_types)) if common_types else "(none)")

    # Build shared evaluation rows
    rows = build_shared_eval_rows(csv_path, target_types=common_types, n_per_class=n_per_class, seed=42)
    print(f"🧪 Using {len(rows)} shared rows for all checkpoints")

    # Simple in-memory tensor cache reused across checkpoints
    tensor_cache = {}

    # Helper to extract numeric suffix
    def _suffix_int(path):
        m = re.search(r"\.tmp_(\d+)$", path)
        return int(m.group(1)) if m else -1

    # Evaluate
    results = []
    for p in sorted(ckpts, key=_suffix_int):
        step = _suffix_int(p)
        print(f"→ Evaluating {os.path.basename(p)} (step={step}) …")
        res = evaluate_one_checkpoint_on_rows(
            ckpt_path=p,
            rows=rows,
            label_intersection=common_types,
            lookbacks=lookbacks,
            chip_size_m=chip_size_m,
            resolution_m=resolution_m,
            max_cloud=max_cloud,
            era_windows=era_windows,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            tensor_cache=tensor_cache,
            strict_channels=True  # skip models whose in_ch ≠ data channels
        )
        if not res.get("ok"):
            print(f"  skipped: {res.get('error')}")
            continue
        results.append({"step": step,
                        "type_acc": res["type_acc"], "type_n": res["type_n"],
                        "sub_acc":  res["sub_acc"],  "sub_n":  res["sub_n"],
                        "ckpt": p, "in_ch": res.get("in_ch")})

    if not results:
        print("No successful results to plot.")
        return

    df = pd.DataFrame(results).sort_values("step")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    # Plot (single figure, default style/colors)
    plt.figure(figsize=(7, 4))
    plt.plot(df["step"], df["type_acc"], marker="o", label="Type accuracy")
    plt.plot(df["step"], df["sub_acc"],  marker="s", label="Subtype accuracy")
    plt.xlabel("Checkpoint step")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Checkpoint")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"📈 Saved plot: {out_png}")

    # Print compact table
    print("\nResults:")
    for _, r in df.iterrows():
        ta = "nan" if (isinstance(r.type_acc, float) and math.isnan(r.type_acc)) else f"{r.type_acc:0.3f}"
        sa = "nan" if (isinstance(r.sub_acc, float) and math.isnan(r.sub_acc)) else f"{r.sub_acc:0.3f}"
        print(f" step={int(r.step):5d} | type_acc={ta} (n={int(r.type_n)})"
              f" | sub_acc={sa} (n={int(r.sub_n)}) | C={int(r.in_ch)} | {os.path.basename(r.ckpt)}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # ==== UPDATE THESE ====
    CKPT_GLOB = "model_3d_multitask_with_negatives.ckpt.tmp_*"
    CSV_PATH  = "src/data/trainingdat/output_fullid/predictor_only_complete_filled.csv"
    # ======================

    sweep_checkpoints_and_plot(
        ckpt_glob_pattern=CKPT_GLOB,
        csv_path=CSV_PATH,
        lookbacks=(1, 5, 10, 20),
        chip_size_m=2560,
        resolution_m=10,
        max_cloud=70,                  # relax to avoid all-zero tiles
        era_windows=(1, 3, 7, 14, 30, 60, 90, 150),           # keep light for speed
        n_per_class=10,
        out_png="test_results/accuracy_sweep.png",
    )
