import os, math, gc, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Union
from collections import defaultdict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    warnings.warn("pyproj not available - geographic transformations may fail")

from src.data.imagesample import fetch_multisource_chips, to_model_tensor

# All the helper functions from your original code
def build_t_dates(t0: str, lookbacks: List[int]) -> List[datetime]:
    """Build list of datetime objects from t0 and lookback days."""
    base_date = datetime.fromisoformat(t0)
    return [(base_date - timedelta(days=int(lb))) for lb in lookbacks]

def _safe_int(x, default: int) -> int:
    """Safely convert to int with fallback."""
    if pd.isna(x):
        return default
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return default

def _safe_float(x, default: float) -> float:
    """Safely convert to float with fallback."""
    if pd.isna(x):
        return default
    try:
        return float(x)
    except (ValueError, TypeError):
        return default

def validate_row_data(row: Dict) -> Tuple[bool, str]:
    """Validate required fields in a row."""
    required_fields = ["Latitude", "Longitude", "Start Year", "Start Month", "Start Day", 
                      "Disaster Type", "Disaster Subtype"]
    
    for field in required_fields:
        if field not in row or pd.isna(row[field]):
            return False, f"Missing or invalid {field}"
    
    # Validate coordinate ranges
    lat = _safe_float(row["Latitude"], None)
    lon = _safe_float(row["Longitude"], None)
    
    if lat is None or not (-90 <= lat <= 90):
        return False, f"Invalid latitude: {lat}"
    if lon is None or not (-180 <= lon <= 180):
        return False, f"Invalid longitude: {lon}"
        
    return True, ""

def t0_from_row(row: Dict) -> str:
    """Extract t0 date string from row with validation."""
    y = _safe_int(row.get("Start Year"), 2000)
    m = max(1, min(12, _safe_int(row.get("Start Month"), 1)))
    d = max(1, min(28, _safe_int(row.get("Start Day"), 1)))
    
    if y < 1980 or y > 2030:
        warnings.warn(f"Year {y} seems unusual, using 2000")
        y = 2000
        
    try:
        return datetime(y, m, d).date().isoformat()
    except ValueError as e:
        warnings.warn(f"Invalid date components Y:{y} M:{m} D:{d}, using 2000-01-01")
        return "2000-01-01"

def nearest_time_index(event_dt: datetime, t_dates: List[datetime], tol_days: int = 15) -> Optional[int]:
    """Find nearest time index within tolerance."""
    if not t_dates:
        return None
        
    diffs = [abs((td - event_dt).days) for td in t_dates]
    idx = int(np.argmin(diffs))
    return idx if diffs[idx] <= tol_days else None

def utm_epsg_for(lat: float, lon: float) -> int:
    """Determine UTM EPSG code for given lat/lon."""
    zone = int((lon + 180) // 6) + 1
    return int(f"{'326' if lat >= 0 else '327'}{zone:02d}")

def latlon_to_pixel(lat: float, lon: float, grid_x: np.ndarray, grid_y: np.ndarray, 
                   epsg: int) -> Tuple[int, int]:
    """Convert lat/lon to pixel coordinates with error handling."""
    if not PYPROJ_AVAILABLE:
        warnings.warn("Using fallback coordinate transformation - less accurate")
        ix = int((lon - grid_x.min()) / (grid_x.max() - grid_x.min()) * (len(grid_x) - 1))
        iy = int((lat - grid_y.min()) / (grid_y.max() - grid_y.min()) * (len(grid_y) - 1))
    else:
        try:
            to_crs = Transformer.from_crs(4326, epsg, always_xy=True)
            x, y = to_crs.transform(lon, lat)
            ix = int(np.argmin(np.abs(grid_x - x)))
            iy = int(np.argmin(np.abs(grid_y - y)))
        except Exception as e:
            warnings.warn(f"Coordinate transformation failed: {e}, using fallback")
            ix = int((lon - grid_x.min()) / (grid_x.max() - grid_x.min()) * (len(grid_x) - 1))
            iy = int((lat - grid_y.min()) / (grid_y.max() - grid_y.min()) * (len(grid_y) - 1))
    
    iy = max(0, min(len(grid_y) - 1, iy))
    ix = max(0, min(len(grid_x) - 1, ix))
    return iy, ix

def draw_disk(yc: int, xc: int, radius: int, H: int, W: int) -> np.ndarray:
    """Create circular mask around center point."""
    yy, xx = np.ogrid[:H, :W]
    return (yy - yc)**2 + (xx - xc)**2 <= radius**2

class LabelEncoder3D:
    """Encode disaster types and subtypes with spatial-temporal labels."""
    
    def __init__(self):
        self.disaster_type_encoder = LabelEncoder()
        self.disaster_subtype_encoder = LabelEncoder()
        self.fitted = False
        
    def fit(self, df: pd.DataFrame):
        """Fit encoders on full dataset."""
        disaster_types = df['Disaster Type'].dropna().unique()
        disaster_subtypes = df['Disaster Subtype'].dropna().unique()
        
        self.disaster_type_encoder.fit(disaster_types)
        self.disaster_subtype_encoder.fit(disaster_subtypes)
        self.fitted = True
        
        print(f"Found {len(disaster_types)} disaster types: {list(disaster_types)}")
        print(f"Found {len(disaster_subtypes)} disaster subtypes: {list(disaster_subtypes)}")
        
    def encode_row_labels(self, row: Dict) -> Dict[str, int]:
        """Encode categorical labels for a row."""
        if not self.fitted:
            raise ValueError("LabelEncoder3D must be fitted before encoding")
            
        disaster_type = row.get('Disaster Type', 'Unknown')
        disaster_subtype = row.get('Disaster Subtype', 'Unknown')
        
        try:
            type_idx = self.disaster_type_encoder.transform([disaster_type])[0]
        except ValueError:
            warnings.warn(f"Unknown disaster type: {disaster_type}")
            type_idx = 0
            
        try:
            subtype_idx = self.disaster_subtype_encoder.transform([disaster_subtype])[0]
        except ValueError:
            warnings.warn(f"Unknown disaster subtype: {disaster_subtype}")
            subtype_idx = 0
            
        return {
            'disaster_type': type_idx,
            'disaster_subtype': subtype_idx,
            'n_disaster_types': len(self.disaster_type_encoder.classes_),
            'n_disaster_subtypes': len(self.disaster_subtype_encoder.classes_)
        }

class MultiTaskUNet3D(nn.Module):
    """3D UNet with spatial and categorical outputs."""
    
    def __init__(self, in_ch: int, n_disaster_types: int, n_disaster_subtypes: int, 
                 base: int = 16, dropout: float = 0.1):
        super().__init__()
        
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv3d(cin, cout, 3, padding=1),
                nn.BatchNorm3d(cout),
                nn.ReLU(inplace=True),
                nn.Conv3d(cout, cout, 3, padding=1),
                nn.BatchNorm3d(cout),
                nn.ReLU(inplace=True),
                nn.Dropout3d(dropout)
            )
        
        # Encoder
        self.enc1 = block(in_ch, base)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.enc2 = block(base, base * 2)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.enc3 = block(base * 2, base * 4)
        self.pool3 = nn.MaxPool3d((1, 2, 2))
        
        # Bottleneck
        self.bottleneck = block(base * 4, base * 8)
        
        # Decoder
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, (1, 2, 2), stride=(1, 2, 2))
        self.dec3 = block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, (1, 2, 2), stride=(1, 2, 2))
        self.dec2 = block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, (1, 2, 2), stride=(1, 2, 2))
        self.dec1 = block(base * 2, base)
        
        # Output heads
        self.spatial_head = nn.Conv3d(base, 1, 1)
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(base * 8, base * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base * 4, base * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.disaster_type_head = nn.Linear(base * 2, n_disaster_types)
        self.disaster_subtype_head = nn.Linear(base * 2, n_disaster_subtypes)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        b = self.bottleneck(self.pool3(e3))
        
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        spatial_out = self.spatial_head(d1)
        
        global_features = self.global_pool(b).flatten(1)
        class_features = self.classifier(global_features)
        
        disaster_type_out = self.disaster_type_head(class_features)
        disaster_subtype_out = self.disaster_subtype_head(class_features)
        
        return {
            'spatial': spatial_out,
            'disaster_type': disaster_type_out,
            'disaster_subtype': disaster_subtype_out
        }

def _to_class_targets(batch_dict, key, batch_size, device):
    """Make a [B]-long int64 target tensor from whatever the DataLoader collated."""
    v = batch_dict[key]
    if isinstance(v, torch.Tensor):
        t = v
    else:
        t = torch.as_tensor(v)
    if t.ndim == 0:
        t = t.expand(batch_size)
    return t.to(device).long().view(-1)

class MultiTaskArray3DTemporalPatchDataset(Dataset):
    """Multi-task 3D patch dataset with spatial and categorical labels."""
    
    def __init__(self, X, Y_spatial, Y_categorical, mu=None, sd=None, 
                 patch=128, stride=128, t_window="all", temporal_stride=1, 
                 augment=True, seed=42):
        assert X.ndim == 4 and Y_spatial.ndim == 4
        self.X, self.Y_spatial = X, Y_spatial
        self.Y_categorical = Y_categorical
        self.T, self.C, self.H, self.W = X.shape
        self.mu, self.sd = mu, sd
        self.patch, self.stride = int(patch), int(stride)
        self.temporal_stride = int(max(1, temporal_stride))
        self.augment = augment
        self.rng = np.random.RandomState(seed)
        
        self.Tw = self.T if t_window in (None, "all") else int(t_window)
        if self.Tw > self.T: 
            raise ValueError(f"t_window ({self.Tw}) > T ({self.T})")

        self.index = []
        for t0 in range(0, self.T - self.Tw + 1, self.temporal_stride):
            for y0 in range(0, self.H - self.patch + 1, self.stride):
                for x0 in range(0, self.W - self.patch + 1, self.stride):
                    self.index.append((t0, y0, x0))

    def __len__(self): 
        return len(self.index)

    def _augment(self, x, y_spatial):
        """Apply spatial augmentations."""
        if self.rng.rand() < 0.5:
            x = x[..., :, ::-1].copy()
            y_spatial = y_spatial[..., :, ::-1].copy()
        if self.rng.rand() < 0.5:
            x = x[..., ::-1, :].copy()
            y_spatial = y_spatial[..., ::-1, :].copy()
        return x, y_spatial

    def __getitem__(self, i):
        t0, y0, x0 = self.index[i]
        p = self.patch
        
        x = self.X[t0:t0+self.Tw, :, y0:y0+p, x0:x0+p].astype(np.float32, copy=False)
        y_spatial = self.Y_spatial[t0:t0+self.Tw, :, y0:y0+p, x0:x0+p].astype(np.float32, copy=False)
        
        x = np.transpose(x, (1, 0, 2, 3))
        y_spatial = np.transpose(y_spatial, (1, 0, 2, 3))

        if (self.mu is not None) and (self.sd is not None):
            mu = self.mu[0, :, 0, 0][:, None, None, None]
            sd = self.sd[0, :, 0, 0][:, None, None, None] + 1e-6
            x = (x - mu) / sd

        if self.augment:
            x, y_spatial = self._augment(x, y_spatial)

        return (torch.from_numpy(x), 
                torch.from_numpy(y_spatial),
                self.Y_categorical)

class NegativeSampleGenerator:
    """Generate negative samples for training."""
    
    def __init__(self, positive_df: pd.DataFrame, negative_ratio: float = 1.0, 
                 temporal_offset_days: int = 365, spatial_offset_km: float = 50.0):
        """
        Args:
            positive_df: DataFrame with disaster samples
            negative_ratio: Ratio of negative to positive samples (1.0 = equal amounts)
            temporal_offset_days: Days to offset for temporal negatives
            spatial_offset_km: Kilometers to offset for spatial negatives
        """
        self.positive_df = positive_df
        self.negative_ratio = negative_ratio
        self.temporal_offset_days = temporal_offset_days
        self.spatial_offset_km = spatial_offset_km
        
        # Get bounds for random sampling
        self.lat_min, self.lat_max = positive_df['Latitude'].min(), positive_df['Latitude'].max()
        self.lon_min, self.lon_max = positive_df['Longitude'].min(), positive_df['Longitude'].max()
        self.year_min = positive_df['Start Year'].min()
        self.year_max = positive_df['Start Year'].max()
        
        print(f"Negative sample bounds: lat [{self.lat_min:.2f}, {self.lat_max:.2f}], "
              f"lon [{self.lon_min:.2f}, {self.lon_max:.2f}], years [{self.year_min}, {self.year_max}]")
    
    def generate_temporal_negatives(self, n_samples: int) -> pd.DataFrame:
        """Generate negative samples by shifting disaster locations in time."""
        negatives = []
        
        for _ in range(n_samples):
            # Pick a random positive sample
            pos_sample = self.positive_df.sample(1).iloc[0]
            
            # Create temporal offset
            original_date = datetime(
                int(pos_sample['Start Year']),
                int(pos_sample['Start Month']),
                int(pos_sample['Start Day'])
            )
            
            # Randomly offset by ±temporal_offset_days
            offset_days = random.randint(-self.temporal_offset_days, self.temporal_offset_days)
            new_date = original_date + timedelta(days=offset_days)
            
            # Ensure we stay within reasonable bounds
            if new_date.year < 2000 or new_date.year > 2023:
                new_date = datetime(
                    random.randint(2000, 2023),
                    random.randint(1, 12),
                    random.randint(1, 28)
                )
            
            negative = {
                'Latitude': pos_sample['Latitude'],
                'Longitude': pos_sample['Longitude'],
                'Start Year': new_date.year,
                'Start Month': new_date.month,
                'Start Day': new_date.day,
                'End Year': np.nan,
                'End Month': np.nan,
                'End Day': np.nan,
                'Disaster Type': 'No Disaster',
                'Disaster Subtype': 'No Disaster',
                'sample_type': 'temporal_negative'
            }
            negatives.append(negative)
        
        return pd.DataFrame(negatives)
    
    def generate_spatial_negatives(self, n_samples: int) -> pd.DataFrame:
        """Generate negative samples by shifting disaster times to different locations."""
        negatives = []
        
        for _ in range(n_samples):
            # Pick a random positive sample  
            pos_sample = self.positive_df.sample(1).iloc[0]
            
            # Generate new location (offset from original)
            lat_offset = random.uniform(-self.spatial_offset_km/111.0, self.spatial_offset_km/111.0)  # ~111km per degree
            lon_offset = random.uniform(-self.spatial_offset_km/111.0, self.spatial_offset_km/111.0)
            
            new_lat = pos_sample['Latitude'] + lat_offset
            new_lon = pos_sample['Longitude'] + lon_offset
            
            # Clamp to reasonable bounds
            new_lat = max(-85, min(85, new_lat))
            new_lon = max(-180, min(180, new_lon))
            
            negative = {
                'Latitude': new_lat,
                'Longitude': new_lon,
                'Start Year': pos_sample['Start Year'],
                'Start Month': pos_sample['Start Month'],
                'Start Day': pos_sample['Start Day'],
                'End Year': np.nan,
                'End Month': np.nan,
                'End Day': np.nan,
                'Disaster Type': 'No Disaster',
                'Disaster Subtype': 'No Disaster',
                'sample_type': 'spatial_negative'
            }
            negatives.append(negative)
        
        return pd.DataFrame(negatives)
    
    def generate_random_negatives(self, n_samples: int) -> pd.DataFrame:
        """Generate completely random negative samples."""
        negatives = []
        
        for _ in range(n_samples):
            # Random location within bounds (expanded slightly)
            lat_range = self.lat_max - self.lat_min
            lon_range = self.lon_max - self.lon_min
            
            new_lat = random.uniform(
                self.lat_min - lat_range * 0.1, 
                self.lat_max + lat_range * 0.1
            )
            new_lon = random.uniform(
                self.lon_min - lon_range * 0.1, 
                self.lon_max + lon_range * 0.1
            )
            
            # Clamp to global bounds
            new_lat = max(-85, min(85, new_lat))
            new_lon = max(-180, min(180, new_lon))
            
            # Random date
            year = random.randint(max(2000, self.year_min), min(2023, self.year_max))
            month = random.randint(1, 12)
            day = random.randint(1, 28)  # Conservative
            
            negative = {
                'Latitude': new_lat,
                'Longitude': new_lon,
                'Start Year': year,
                'Start Month': month,
                'Start Day': day,
                'End Year': np.nan,
                'End Month': np.nan,
                'End Day': np.nan,
                'Disaster Type': 'No Disaster',
                'Disaster Subtype': 'No Disaster',
                'sample_type': 'random_negative'
            }
            negatives.append(negative)
        
        return pd.DataFrame(negatives)
    
    def generate_negatives(self, n_positives: int) -> pd.DataFrame:
        """Generate a balanced set of negative samples."""
        total_negatives = int(n_positives * self.negative_ratio)
        
        # Split negatives into three types
        n_temporal = total_negatives // 3
        n_spatial = total_negatives // 3
        n_random = total_negatives - n_temporal - n_spatial
        
        print(f"Generating {total_negatives} negative samples:")
        print(f"  - {n_temporal} temporal negatives")
        print(f"  - {n_spatial} spatial negatives") 
        print(f"  - {n_random} random negatives")
        
        negatives = []
        
        if n_temporal > 0:
            temporal_negs = self.generate_temporal_negatives(n_temporal)
            negatives.append(temporal_negs)
        
        if n_spatial > 0:
            spatial_negs = self.generate_spatial_negatives(n_spatial)
            negatives.append(spatial_negs)
        
        if n_random > 0:
            random_negs = self.generate_random_negatives(n_random)
            negatives.append(random_negs)
        
        if negatives:
            return pd.concat(negatives, ignore_index=True)
        else:
            return pd.DataFrame()

def enhanced_train_over_csv(csv_path: str,
                           max_rows: Optional[int] = None,
                           negative_ratio: float = 1.0,  # NEW: ratio of negative samples
                           lookbacks: Optional[List[int]] = None,
                           patch: int = 128, stride: int = 128,
                           t_window: str = "all", epochs_per_row: int = 2, batch: int = 4,
                           base: int = 16, dropout: float = 0.1,
                           spatial_weight: float = 1.0, type_weight: float = 1.0, subtype_weight: float = 1.0,
                           ckpt_path: str = "model_3d_multitask.ckpt",
                           save_every: int = 10):
    """
    Enhanced training with negative sample generation.
    """
    
    # Load and validate CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} positive samples from {csv_path}")
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}")
    
    # Limit positive samples if requested
    if max_rows is not None:
        df = df.head(max_rows)
        print(f"Limited to {len(df)} positive samples")
    
    # Generate negative samples
    if negative_ratio > 0:
        neg_generator = NegativeSampleGenerator(df, negative_ratio=negative_ratio)
        negative_df = neg_generator.generate_negatives(len(df))
        
        # Combine positive and negative samples
        combined_df = pd.concat([df, negative_df], ignore_index=True)
        print(f"Combined dataset: {len(df)} positives + {len(negative_df)} negatives = {len(combined_df)} total")
    else:
        combined_df = df
        print(f"Training only on positive samples: {len(combined_df)}")
    
    # Shuffle the combined dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Check required columns
    required_cols = ["Latitude", "Longitude", "Start Year", "Start Month", "Start Day",
                    "Disaster Type", "Disaster Subtype"]
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Initialize label encoder (now includes "No Disaster")
    label_encoder = LabelEncoder3D()
    label_encoder.fit(combined_df)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = None
    successful_rows = 0
    failed_rows = 0

    # Process rows
    for i, (_, row) in enumerate(combined_df.iterrows()):
        print(f"\n=== Row {i+1}/{len(combined_df)} ===")
        
        row_dict = row.to_dict()
        sample_type = row_dict.get('sample_type', 'positive')
        disaster_type = row_dict.get('Disaster Type', 'Unknown')
        
        print(f"Sample type: {sample_type}, Disaster type: {disaster_type}")
        
        # Initialize model on first successful row
        if model is None:
            print("Initializing model...")
            try:
                # Probe for channel count using a positive sample
                valid_row = None
                for _, probe_row in combined_df.iterrows():
                    probe_dict = probe_row.to_dict()
                    valid, _ = validate_row_data(probe_dict)
                    if valid:
                        valid_row = probe_dict
                        break
                
                if valid_row is None:
                    raise ValueError("No valid rows found for model initialization")
                    
                lat = _safe_float(valid_row["Latitude"], 0.0)
                lon = _safe_float(valid_row["Longitude"], 0.0)
                t0 = t0_from_row(valid_row)
                lbs = lookbacks or [1, 5, 10, 20]
                
                chips_probe = fetch_multisource_chips(
                    lat=lat, lon=lon, t0=t0, lookbacks=lbs,
                    chip_size_m=2560, resolution_m=10, max_cloud=25,
                    era_windows=[1, 3, 7, 14, 30, 60, 90, 150],
                    include_s2=True, include_dem=True, include_worldcover=True,
                    include_era5=True, include_smap=True
                )
                X_probe, channels = to_model_tensor(chips_probe)
                C = X_probe.shape[1]
                
                # Create model
                dummy_labels = label_encoder.encode_row_labels(valid_row)
                model = MultiTaskUNet3D(
                    in_ch=C,
                    n_disaster_types=dummy_labels['n_disaster_types'],
                    n_disaster_subtypes=dummy_labels['n_disaster_subtypes'],
                    base=base,
                    dropout=dropout
                ).to(device)
                
                print(f"Model initialized with {C} input channels")
                print(f"Disaster types: {dummy_labels['n_disaster_types']}")
                print(f"Disaster subtypes: {dummy_labels['n_disaster_subtypes']}")
                print(f"Classes: {label_encoder.disaster_type_encoder.classes_}")
                
                # Cleanup probe
                del chips_probe, X_probe
                gc.collect()
                
            except Exception as e:
                print(f"Failed to initialize model on row {i+1}: {e}")
                failed_rows += 1
                continue

        # Train on this row
        success = train_on_row(
            row_dict, model, device, label_encoder,
            lookbacks=lookbacks, patch=patch, stride=stride,
            t_window=t_window, epochs=epochs_per_row, batch=batch,
            spatial_weight=spatial_weight, type_weight=type_weight, subtype_weight=subtype_weight
        )
        
        if success:
            successful_rows += 1
        else:
            failed_rows += 1
            
        # Periodic saving
        if successful_rows > 0 and successful_rows % save_every == 0:
            temp_ckpt = f"{ckpt_path}.tmp_{successful_rows}"
            torch.save({
                "model": model.state_dict(),
                "base": base,
                "dropout": dropout,
                "n_disaster_types": label_encoder.disaster_type_encoder.classes_.tolist(),
                "n_disaster_subtypes": label_encoder.disaster_subtype_encoder.classes_.tolist(),
                "successful_rows": successful_rows,
                "failed_rows": failed_rows,
                "negative_ratio": negative_ratio,
                "training_info": {
                    "total_samples": len(combined_df),
                    "positive_samples": len(df),
                    "negative_samples": len(combined_df) - len(df)
                }
            }, temp_ckpt)
            print(f"Saved checkpoint: {temp_ckpt}")

    # Final save
    if model is not None:
        final_save_data = {
            "model": model.state_dict(),
            "base": base,
            "dropout": dropout,
            "disaster_types": label_encoder.disaster_type_encoder.classes_.tolist(),
            "disaster_subtypes": label_encoder.disaster_subtype_encoder.classes_.tolist(),
            "successful_rows": successful_rows,
            "failed_rows": failed_rows,
            "lookbacks": lookbacks or [1, 5, 10, 20],
            "negative_ratio": negative_ratio,
            "training_config": {
                "patch": patch,
                "stride": stride,
                "t_window": t_window,
                "epochs_per_row": epochs_per_row,
                "batch": batch,
                "spatial_weight": spatial_weight,
                "type_weight": type_weight,
                "subtype_weight": subtype_weight,
                "total_samples": len(combined_df),
                "positive_samples": len(df),
                "negative_samples": len(combined_df) - len(df)
            }
        }
        torch.save(final_save_data, ckpt_path)
        print(f"\nFinal model saved: {ckpt_path}")
        print(f"Training summary: {successful_rows} successful, {failed_rows} failed")
        print(f"Sample breakdown: {len(df)} positive, {len(combined_df) - len(df)} negative")
    else:
        print("No model was successfully initialized - check your data and dependencies")

# Enhanced make_Y_from_single_row to handle negative samples
def make_Y_from_single_row_enhanced(row: Dict, chips, t0: str, lookbacks: List[int],
                                   label_encoder: LabelEncoder3D,
                                   radius_px: int = 2, tol_days: int = 15) -> Tuple[np.ndarray, Dict]:
    """
    Enhanced version that properly handles negative samples (no spatial signal).
    """
    lat = _safe_float(row["Latitude"], 0.0)
    lon = _safe_float(row["Longitude"], 0.0)
    disaster_type = row.get("Disaster Type", "Unknown")
    
    # Get spatial grid info
    try:
        s2 = chips["S2"]
        grid = s2.isel(time=0, band=0)
        H, W = grid.sizes["y"], grid.sizes["x"]
        grid_x = grid.x.values
        grid_y = grid.y.values
        
        if "crs" in s2.attrs and isinstance(s2.attrs["crs"], str) and s2.attrs["crs"].startswith("EPSG:"):
            epsg = int(s2.attrs["crs"].split(":")[1])
        else:
            epsg = utm_epsg_for(lat, lon)
    except Exception as e:
        warnings.warn(f"Failed to extract grid info: {e}")
        raise

    T = len(lookbacks)
    Y_spatial = np.zeros((T, 1, H, W), dtype=np.float32)
    
    # Only create spatial labels for actual disasters (not "No Disaster")
    if disaster_type != "No Disaster":
        # Original logic for positive samples
        sy, sm, sd = row.get("Start Year"), row.get("Start Month"), row.get("Start Day")
        ey, em, ed = row.get("End Year"), row.get("End Month"), row.get("End Day")
        
        start = datetime(_safe_int(sy, 2000), max(1, _safe_int(sm, 1)), max(1, _safe_int(sd, 1)))
        if pd.isna(ey):
            end = start
        else:
            end = datetime(_safe_int(ey, start.year), max(1, _safe_int(em, start.month)), max(1, _safe_int(ed, start.day)))

        t_dates = build_t_dates(t0, lookbacks)
        days = max(0, (end - start).days)
        event_days = [start + timedelta(days=d) for d in range(days + 1)]

        try:
            iy, ix = latlon_to_pixel(lat, lon, grid_x, grid_y, epsg)
            mask = draw_disk(iy, ix, radius_px, H, W)

            for d in event_days:
                ti = nearest_time_index(d, t_dates, tol_days)
                if ti is not None:
                    Y_spatial[ti, 0][mask] = 1.0
        except Exception as e:
            warnings.warn(f"Failed to create spatial mask: {e}")
    
    # For negative samples, Y_spatial remains all zeros (which is correct)
    
    # Get categorical labels
    Y_categorical = label_encoder.encode_row_labels(row)
    
    return Y_spatial, Y_categorical

# Enhanced train_on_row that handles negative samples properly
def train_on_row(row: Dict, model, device, label_encoder: LabelEncoder3D,
                 lookbacks=None, patch=128, stride=128, t_window="all", 
                 epochs=2, batch=4, spatial_weight=1.0, type_weight=1.0, subtype_weight=1.0):
    """Train model on single row with multi-task learning, handling negative samples."""
    
    # Validate row data (relaxed for negative samples)
    disaster_type = row.get("Disaster Type", "Unknown")
    if disaster_type != "No Disaster":
        valid, error_msg = validate_row_data(row)
        if not valid:
            warnings.warn(f"Skipping row due to validation error: {error_msg}")
            return False
    
    lat = _safe_float(row["Latitude"], 0.0)
    lon = _safe_float(row["Longitude"], 0.0)
    t0 = t0_from_row(row)
    lookbacks = lookbacks or [1, 5, 10, 20]

    print(f"  Training on: {row.get('Disaster Type', 'Unknown')} - {row.get('Disaster Subtype', 'Unknown')}")
    print(f"  Location: ({lat:.3f}, {lon:.3f}), Date: {t0}")

    try:
        # Fetch chips & build X
        chips = fetch_multisource_chips(
            lat=lat, lon=lon, t0=t0,
            lookbacks=lookbacks,
            chip_size_m=2560, resolution_m=10,
            max_cloud=25, era_windows=[1, 3, 7, 14, 30, 60, 90, 150],
            include_s2=True, include_dem=True, include_worldcover=True,
            include_era5=True, include_smap=True
        )
        X, channels = to_model_tensor(chips)
        
    except Exception as e:
        warnings.warn(f"Failed to fetch chips for row: {e}")
        return False

    try:
        # Build labels using enhanced function for negative samples
        Y_spatial, Y_categorical = make_Y_from_single_row_enhanced(
            row, chips, t0, lookbacks, label_encoder, radius_px=2, tol_days=15
        )
    except Exception as e:
        warnings.warn(f"Failed to create labels for row: {e}")
        return False

    # Normalization stats
    mu = X.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    sd = X.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    sd += 1e-6

    # Dataset & loader
    ds = MultiTaskArray3DTemporalPatchDataset(
        X, Y_spatial, Y_categorical, mu=mu, sd=sd,
        patch=patch, stride=stride, t_window=t_window, 
        temporal_stride=1, augment=True
    )
    
    if len(ds) == 0:
        warnings.warn("No patches generated for this row")
        return False
        
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)

    # Multi-task loss setup
    pos_frac = float((Y_spatial > 0.5).mean())
    pos_weight = 1.0 if pos_frac <= 0 or pos_frac >= 1 else (1.0 - pos_frac) / pos_frac
    pos_weight = min(pos_weight, 100.0)
    
    spatial_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    type_criterion = nn.CrossEntropyLoss()
    subtype_criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Training loop
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        spatial_loss_acc = 0.0
        type_loss_acc = 0.0
        subtype_loss_acc = 0.0
        n_batches = 0

        for xb, yb_spatial, yb_categorical in dl:
            xb = xb.to(device)
            yb_spatial = yb_spatial.to(device)
            
            # Categorical targets
            B = xb.size(0)
            disaster_type_target = _to_class_targets(yb_categorical, 'disaster_type', B, device)
            disaster_subtype_target = _to_class_targets(yb_categorical, 'disaster_subtype', B, device)

            # Forward pass
            outputs = model(xb)
            
            # Compute losses
            spatial_loss = spatial_criterion(outputs['spatial'], yb_spatial)
            type_loss = type_criterion(outputs['disaster_type'], disaster_type_target)
            subtype_loss = subtype_criterion(outputs['disaster_subtype'], disaster_subtype_target)
            
            # Combined loss
            total_batch_loss = (spatial_weight * spatial_loss + 
                              type_weight * type_loss + 
                              subtype_weight * subtype_loss)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Accumulate losses
            total_loss += total_batch_loss.item()
            spatial_loss_acc += spatial_loss.item()
            type_loss_acc += type_loss.item()
            subtype_loss_acc += subtype_loss.item()
            n_batches += 1

        # Print epoch stats
        if n_batches > 0:
            avg_total = total_loss / n_batches
            avg_spatial = spatial_loss_acc / n_batches
            avg_type = type_loss_acc / n_batches
            avg_subtype = subtype_loss_acc / n_batches
            
            print(f"  Epoch {ep+1}: total={avg_total:.4f} "
                  f"spatial={avg_spatial:.4f} type={avg_type:.4f} subtype={avg_subtype:.4f}")

    # Cleanup
    del chips, X, Y_spatial, mu, sd, ds, dl
    gc.collect()
    
    return True

if __name__ == "__main__":
    # Example usage with negative samples
    enhanced_train_over_csv(
        csv_path="src/data/trainingdat/output_fullid/predictor_only_complete_filled.csv",
        max_rows=500,  # Limit for testing
        negative_ratio=2.0,  # 2x negative samples (20 negatives for 10 positives)
        ckpt_path="model_3d_multitask_with_negatives.ckpt",
        base=32,
        dropout=0.15,
        epochs_per_row=2,
        save_every=5
    )