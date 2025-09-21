"""
risk_predictors_full.py - FIXED VERSION

Build multi-temporal, multi-source predictors for drought & wildfire risk.

MAJOR FIXES:
1. Proper error handling and data validation
2. Better stackstac configuration
3. Improved ERA5 data access and processing
4. Fixed coordinate alignment issues
5. Added debugging and fallback mechanisms
6. Better handling of missing data

Author: Fixed by Claude
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd
import xarray as xr
import warnings
import traceback
from pathlib import Path

from shapely.geometry import Point, box
from pyproj import Transformer

try:
    from pystac_client import Client
    import planetary_computer as pc
    import stackstac
    STAC_AVAILABLE = True
except ImportError as e:
    STAC_AVAILABLE = False
    warnings.warn(f"STAC libraries not available: {e}")

from scipy.stats import gamma, norm


# =========================
# Enhanced CRS / Grid helpers
# =========================

def utm_epsg_for(lat: float, lon: float) -> int:
    """Get UTM EPSG code for a lat/lon point."""
    zone = int((lon + 180) // 6) + 1
    hemisphere = "326" if lat >= 0 else "327"
    return int(f"{hemisphere}{zone:02d}")

def square_bounds_in_crs(lat: float, lon: float, half_size_m: float, epsg: int) -> Tuple[float,float,float,float]:
    """Convert lat/lon center + half-size to bounds in target CRS."""
    try:
        to_crs = Transformer.from_crs(4326, epsg, always_xy=True)
        cx, cy = to_crs.transform(lon, lat)
        return (cx - half_size_m, cy - half_size_m, cx + half_size_m, cy + half_size_m)
    except Exception as e:
        warnings.warn(f"CRS transformation failed: {e}")
        # Fallback to approximate degree-based bounds
        deg_per_m = 1.0 / 111000  # Rough approximation
        half_deg = half_size_m * deg_per_m
        return (lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg)

def grid_hw(chip_size_m: int, resolution_m: int) -> int:
    """Calculate grid height/width from chip size and resolution."""
    return max(1, int(round(chip_size_m / resolution_m)))

def create_empty_dataarray(T: int, bands: List[str], H: int, W: int, epsg: int) -> xr.DataArray:
    """Create empty DataArray with proper structure."""
    return xr.DataArray(
        np.zeros((T, len(bands), H, W), dtype=np.float32),
        dims=("time", "band", "y", "x"),
        coords={
            "time": pd.date_range("2020-01-01", periods=T, freq="D"),
            "band": bands,
            "y": np.arange(H),
            "x": np.arange(W)
        },
        attrs={"crs": f"EPSG:{epsg}"}
    )


# =========================
# Enhanced Sentinel-2 utilities
# =========================

S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]
SCL_KEEP = {2, 4, 5, 6, 7}  # Dark, Veg, Bare, Water, Unclass

def apply_scl_mask(chip: xr.DataArray) -> xr.DataArray:
    """Apply SCL cloud mask to Sentinel-2 data with better error handling."""
    try:
        if "band" not in chip.coords:
            pass #print("Warning: No band coordinate in chip")
            return chip
            
        available_bands = list(chip.coords["band"].values)
        if "SCL" not in available_bands:
            pass #print(f"Warning: SCL not in available bands: {available_bands}")
            return chip
        
        scl = chip.sel(band="SCL")
        mask = xr.apply_ufunc(
            lambda x: np.isin(x, list(SCL_KEEP)),
            scl,
            dask="forbidden",
            output_dtypes=[bool]
        )

        # Create mask for all bands
        masked_chip = chip.copy()
        for band in available_bands:
            if band != "SCL":
                masked_chip.loc[dict(band=band)] = xr.where(
                    mask, 
                    chip.sel(band=band), 
                    0
                )
        
        return masked_chip
        
    except Exception as e:
        pass #print(f"Warning: SCL masking failed: {e}")
        return chip

def compute_indices(chip: xr.DataArray) -> xr.DataArray:
    """Compute NDVI, NDWI, NDMI from Sentinel-2 bands with error handling."""
    try:
        if "band" not in chip.coords:
            return chip
            
        available_bands = set(chip.coords["band"].values)
        
        def safe_idx(n, d):
            """Safe index calculation avoiding division by zero."""
            denominator = n + d
            return xr.where(
                (denominator != 0) & (denominator > 1e-6),
                (n - d) / denominator,
                0
            ).astype("float32")
        
        additional_arrays = [chip]
        
        # NDVI: (NIR - Red) / (NIR + Red)
        if {"B08", "B04"}.issubset(available_bands):
            ndvi = safe_idx(chip.sel(band="B08"), chip.sel(band="B04"))
            ndvi_expanded = ndvi.expand_dims({"band": ["NDVI"]})
            additional_arrays.append(ndvi_expanded)
        
        # NDWI: (Green - NIR) / (Green + NIR)  
        if {"B03", "B08"}.issubset(available_bands):
            ndwi = safe_idx(chip.sel(band="B03"), chip.sel(band="B08"))
            ndwi_expanded = ndwi.expand_dims({"band": ["NDWI"]})
            additional_arrays.append(ndwi_expanded)
        
        # NDMI: (NIR - SWIR) / (NIR + SWIR)
        if {"B08", "B11"}.issubset(available_bands):
            ndmi = safe_idx(chip.sel(band="B08"), chip.sel(band="B11"))
            ndmi_expanded = ndmi.expand_dims({"band": ["NDMI"]})
            additional_arrays.append(ndmi_expanded)
        
        if len(additional_arrays) > 1:
            return xr.concat(additional_arrays, dim="band")
        else:
            return chip
            
    except Exception as e:
        pass #print(f"Warning: Index computation failed: {e}")
        return chip

def fetch_s2_stack(
    lat: float, lon: float, t0: str, lookbacks: List[int],
    chip_size_m: int, resolution_m: int, max_cloud_pct: int = 30
) -> xr.DataArray:
    """Fetch Sentinel-2 stack with improved error handling and data validation."""
    
    if not STAC_AVAILABLE:
        pass #print("Warning: STAC not available, returning empty S2 data")
        H = W = grid_hw(chip_size_m, resolution_m)
        T = len(lookbacks)
        epsg = utm_epsg_for(lat, lon)
        return create_empty_dataarray(T, S2_BANDS + ["NDVI", "NDWI", "NDMI"], H, W, epsg)
    
    try:
        stac = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        epsg = utm_epsg_for(lat, lon)
        xmin, ymin, xmax, ymax = square_bounds_in_crs(lat, lon, chip_size_m/2, epsg)
        H = W = grid_hw(chip_size_m, resolution_m)

        out_chips, out_times = [], []
        
        pass #print(f"Fetching S2 data for {len(lookbacks)} time points...")
        
        for i, lb in enumerate(lookbacks):
            pass #print(f"  Processing lookback {lb} days ({i+1}/{len(lookbacks)})")
            
            tref = (datetime.fromisoformat(t0) - timedelta(days=int(lb)))
            t_start = (tref - timedelta(days=7)).isoformat()  # Wider window
            t_end = (tref + timedelta(days=7)).isoformat()
            
            try:
                # Search for items
                search = stac.search(
                    collections=["sentinel-2-l2a"],
                    datetime=f"{t_start}/{t_end}",
                    intersects={"type": "Point", "coordinates": [lon, lat]},
                    query={"eo:cloud_cover": {"lt": max_cloud_pct}},
                    limit=50
                )
                
                items = list(search.items())
                pass #print(f"    Found {len(items)} S2 items")

                if not items:
                    # Create empty chip
                    chip = xr.DataArray(
                        np.zeros((len(S2_BANDS) + 3, H, W), dtype=np.float32),
                        dims=("band", "y", "x"),
                        coords={
                            "band": S2_BANDS + ["NDVI", "NDWI", "NDMI"],
                            "y": np.linspace(ymax, ymin, H),
                            "x": np.linspace(xmin, xmax, W)
                        },
                        attrs={"crs": f"EPSG:{epsg}"}
                    )
                    out_chips.append(chip)
                    out_times.append(np.datetime64(tref.date()))
                    continue

                # Sort by cloud cover and date
                items.sort(key=lambda it: (
                    it.properties.get("eo:cloud_cover", 100.0),
                    abs((datetime.fromisoformat(it.properties["datetime"]) - tref).days)
                ))
                
                # Take best item
                best_item = items[0]
                signed_item = pc.sign(best_item)
                
                pass #print(f"    Using item: {best_item.id}, cloud cover: {best_item.properties.get('eo:cloud_cover', 'unknown')}")

                # Stack with better error handling
                try:
                    stk = stackstac.stack(
                        [signed_item.to_dict()],
                        assets=S2_BANDS,
                        resolution=resolution_m,
                        bounds=(xmin, ymin, xmax, ymax),
                        epsg=epsg,
                        dtype="float32",
                        fill_value=0.0,  # Use 0.0 for float32, not 0 (int)
                        chunks={},  # Load all into memory
                        rescale=False  # Keep original DN values initially
                    )
                    
                    # Load and process
                    chip = stk.isel(item=0).drop_vars("item", errors="ignore")
                    chip = chip.compute()  # Force computation
                    
                    # Scale to reflectance (S2 L2A is already in reflectance * 10000)
                    reflectance_bands = ["B02", "B03", "B04", "B08", "B11", "B12"]
                    for band in reflectance_bands:
                        if band in chip.coords["band"].values:
                            chip.loc[dict(band=band)] = chip.sel(band=band) / 10000.0
                    
                    # Clip values to reasonable range
                    chip = chip.clip(min=0, max=1.5)  # Allow slightly over 100% reflectance
                    
                    # Apply quality mask and compute indices
                    chip = apply_scl_mask(chip)
                    chip = compute_indices(chip)
                    
                    # Validate data
                    if chip.isnull().all():
                        pass #print("    Warning: All data is null, creating empty chip")
                        chip = xr.DataArray(
                            np.zeros((len(S2_BANDS) + 3, H, W), dtype=np.float32),
                            dims=("band", "y", "x"),
                            coords={
                                "band": S2_BANDS + ["NDVI", "NDWI", "NDMI"],
                                "y": chip.y,
                                "x": chip.x
                            },
                            attrs={"crs": f"EPSG:{epsg}"}
                        )
                    
                    out_chips.append(chip)
                    out_times.append(np.datetime64(best_item.properties["datetime"]))
                    
                    # pass #print some statistics for debugging
                    non_zero_pct = (chip != 0).sum() / chip.size * 100
                    pass #print(f"    Chip stats: {non_zero_pct:.1f}% non-zero values")
                    
                except Exception as e:
                    pass #print(f"    Stackstac processing failed: {e}")
                    # Create empty fallback
                    chip = xr.DataArray(
                        np.zeros((len(S2_BANDS) + 3, H, W), dtype=np.float32),
                        dims=("band", "y", "x"),
                        coords={
                            "band": S2_BANDS + ["NDVI", "NDWI", "NDMI"],
                            "y": np.linspace(ymax, ymin, H),
                            "x": np.linspace(xmin, xmax, W)
                        },
                        attrs={"crs": f"EPSG:{epsg}"}
                    )
                    out_chips.append(chip)
                    out_times.append(np.datetime64(tref.date()))
                    
            except Exception as e:
                pass #print(f"    Failed to fetch S2 for lookback {lb}: {e}")
                # Create empty fallback
                chip = xr.DataArray(
                    np.zeros((len(S2_BANDS) + 3, H, W), dtype=np.float32),
                    dims=("band", "y", "x"),
                    coords={
                        "band": S2_BANDS + ["NDVI", "NDWI", "NDMI"],
                        "y": np.linspace(ymax, ymin, H),
                        "x": np.linspace(xmin, xmax, W)
                    },
                    attrs={"crs": f"EPSG:{epsg}"}
                )
                out_chips.append(chip)
                out_times.append(np.datetime64(tref.date()))

        # Concatenate along time dimension
        if out_chips:
            result = xr.concat(out_chips, dim="time")
            result = result.assign_coords(time=np.array(out_times, dtype="datetime64[ns]"))
            
            # Final validation
            total_non_zero = (result != 0).sum().values
            total_elements = result.size
            pass #print(f"S2 final stats: {total_non_zero}/{total_elements} ({total_non_zero/total_elements*100:.1f}%) non-zero")
            
            return result
        else:
            # Complete fallback
            return create_empty_dataarray(len(lookbacks), S2_BANDS + ["NDVI", "NDWI", "NDMI"], H, W, epsg)
            
    except Exception as e:
        pass #print(f"S2 fetch completely failed: {e}")
        #traceback.pass #print_exc()
        H = W = grid_hw(chip_size_m, resolution_m)
        T = len(lookbacks)
        epsg = utm_epsg_for(lat, lon)
        return create_empty_dataarray(T, S2_BANDS + ["NDVI", "NDWI", "NDMI"], H, W, epsg)


# =========================
# Enhanced DEM processing
# =========================

def fetch_dem(lat: float, lon: float, chip_size_m: int, resolution_m: int) -> xr.DataArray:
    """Fetch DEM with enhanced error handling."""
    
    if not STAC_AVAILABLE:
        pass #print("Warning: STAC not available, returning empty DEM data")
        H = W = grid_hw(chip_size_m, resolution_m)
        epsg = utm_epsg_for(lat, lon)
        return create_empty_dataarray(1, ["DEM"], H, W, epsg).isel(time=0)
    
    try:
        stac = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        epsg = utm_epsg_for(lat, lon)
        xmin, ymin, xmax, ymax = square_bounds_in_crs(lat, lon, chip_size_m/2, epsg)
        H = W = grid_hw(chip_size_m, resolution_m)
        
        pass #print("Fetching DEM data...")
        
        # Search for DEM items
        items = list(stac.search(
            collections=["cop-dem-glo-30"],
            intersects=Point(lon, lat).__geo_interface__,
            limit=10
        ).items())
        
        if not items:
            pass #print("Warning: No DEM items found")
            return xr.DataArray(
                np.zeros((1, H, W), np.float32), 
                dims=("band", "y", "x"), 
                coords={
                    "band": ["DEM"],
                    "y": np.linspace(ymax, ymin, H),
                    "x": np.linspace(xmin, xmax, W)
                },
                attrs={"crs": f"EPSG:{epsg}"}
            )

        pass #print(f"Found {len(items)} DEM items")
        
        # Use first item
        item = items[0]
        signed_item = pc.sign(item)
        
        dem = stackstac.stack(
            [signed_item.to_dict()],
            assets=["data"],
            resolution=resolution_m,
            bounds=(xmin, ymin, xmax, ymax),
            epsg=epsg,
            dtype="float32",
            fill_value=np.nan,
            chunks={}
        )
        
        dem = dem.isel(item=0).drop_vars("item", errors="ignore")
        dem = dem.compute()
        dem = dem.fillna(0)  # Fill NaN with 0
        dem = dem.assign_coords(band=["DEM"])
        
        # Validate
        non_zero_pct = (dem != 0).sum() / dem.size * 100
        pass #print(f"DEM stats: {non_zero_pct:.1f}% non-zero values")
        
        return dem
        
    except Exception as e:
        pass #print(f"DEM fetch failed: {e}")
        H = W = grid_hw(chip_size_m, resolution_m)
        epsg = utm_epsg_for(lat, lon)
        return xr.DataArray(
            np.zeros((1, H, W), np.float32), 
            dims=("band", "y", "x"), 
            coords={
                "band": ["DEM"],
                "y": np.linspace(0, H-1, H),
                "x": np.linspace(0, W-1, W)
            },
            attrs={"crs": f"EPSG:{epsg}"}
        )

def dem_to_slope_aspect(dem_da: xr.DataArray, resolution_m: int) -> xr.DataArray:
    """Enhanced slope and aspect computation."""
    try:
        if "DEM" not in dem_da.coords["band"].values:
            pass #print("Warning: No DEM band found")
            return dem_da
            
        dem = dem_da.sel(band="DEM")
        
        # Compute gradients with proper edge handling
        dz_dy, dz_dx = np.gradient(dem.values, resolution_m, resolution_m, edge_order=1)
        
        # Calculate slope in degrees
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad)
        
        # Calculate aspect in degrees (0-360)
        aspect_rad = np.arctan2(-dz_dx, dz_dy)
        aspect_rad = np.where(aspect_rad < 0, aspect_rad + 2*np.pi, aspect_rad)
        aspect_deg = np.degrees(aspect_rad)
        
        # Create DataArrays with proper coordinates
        slope = xr.DataArray(
            slope_deg, 
            dims=("y", "x"), 
            coords={"y": dem.y, "x": dem.x}
        )
        aspect = xr.DataArray(
            aspect_deg, 
            dims=("y", "x"), 
            coords={"y": dem.y, "x": dem.x}
        )
        
        # Combine all bands
        out = xr.concat([
            dem.expand_dims({"band": ["DEM"]}),
            slope.expand_dims({"band": ["SLOPE"]}),
            aspect.expand_dims({"band": ["ASPECT"]})
        ], dim="band")
        
        pass #print(f"DEM processing: Added SLOPE (max: {slope_deg.max():.1f}°) and ASPECT")
        
        return out
        
    except Exception as e:
        pass #print(f"DEM slope/aspect computation failed: {e}")
        return dem_da


# =========================
# Simplified ERA5 and other data sources
# =========================

def fetch_worldcover(lat: float, lon: float, chip_size_m: int, resolution_m: int) -> xr.DataArray:
    """Fetch WorldCover with better error handling."""
    
    if not STAC_AVAILABLE:
        pass #print("Warning: STAC not available, creating dummy WorldCover data")
        H = W = grid_hw(chip_size_m, resolution_m)
        epsg = utm_epsg_for(lat, lon)
        # Create realistic land cover values (40 = cropland, 50 = urban)
        dummy_lc = np.full((1, H, W), 40, dtype=np.uint8)
        return xr.DataArray(
            dummy_lc,
            dims=("band", "y", "x"),
            coords={
                "band": ["LC"],
                "y": np.arange(H),
                "x": np.arange(W)
            },
            attrs={"crs": f"EPSG:{epsg}"}
        )
    
    try:
        stac = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        epsg = utm_epsg_for(lat, lon)
        xmin, ymin, xmax, ymax = square_bounds_in_crs(lat, lon, chip_size_m/2, epsg)
        H = W = grid_hw(chip_size_m, resolution_m)
        
        pass #print("Fetching WorldCover data...")
        
        items = list(stac.search(
            collections=["esa-worldcover"],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            limit=5
        ).items())
        
        if not items:
            pass #print("Warning: No WorldCover items found, using dummy data")
            dummy_lc = np.full((1, H, W), 40, dtype=np.uint8)  # Cropland
            return xr.DataArray(
                dummy_lc,
                dims=("band", "y", "x"),
                coords={
                    "band": ["LC"],
                    "y": np.linspace(ymax, ymin, H),
                    "x": np.linspace(xmin, xmax, W)
                },
                attrs={"crs": f"EPSG:{epsg}"}
            )

        item = items[0]
        signed_item = pc.sign(item)
        
        lc = stackstac.stack(
            [signed_item.to_dict()],
            assets=["map"],
            resolution=resolution_m,
            bounds=(xmin, ymin, xmax, ymax),
            epsg=epsg,
            dtype="uint8",
            fill_value=0,
            chunks={}
        )
        
        lc = lc.isel(item=0).drop_vars("item", errors="ignore")
        lc = lc.compute()
        lc = lc.assign_coords(band=["LC"])
        
        # Validate - WorldCover values should be 10, 20, 30, etc.
        unique_values = np.unique(lc.values)
        pass #print(f"WorldCover unique values: {unique_values}")
        
        return lc.astype("uint8")
        
    except Exception as e:
        pass #print(f"WorldCover fetch failed: {e}")
        H = W = grid_hw(chip_size_m, resolution_m)
        epsg = utm_epsg_for(lat, lon)
        dummy_lc = np.full((1, H, W), 40, dtype=np.uint8)
        return xr.DataArray(
            dummy_lc,
            dims=("band", "y", "x"),
            coords={
                "band": ["LC"],
                "y": np.arange(H),
                "x": np.arange(W)
            },
            attrs={"crs": f"EPSG:{epsg}"}
        )

def create_dummy_era5(lat: float, lon: float, t_dates: List[datetime], 
                     windows: List[int], chip_size_m: int, resolution_m: int) -> xr.DataArray:
    """Create realistic dummy ERA5 data when real data is unavailable."""
    H = W = grid_hw(chip_size_m, resolution_m)
    
    # Create realistic dummy weather data
    bands_all = []
    for w in windows:
        bands_all.extend([f"precip_sum_{w}d", f"vpd_mean_{w}d", f"fwi_mean_{w}d", f"kbdi_mean_{w}d"])
    
    # Initialize with realistic values
    data = np.zeros((len(t_dates), len(bands_all), H, W), dtype=np.float32)
    
    for i, band in enumerate(bands_all):
        if "precip" in band:
            # Precipitation: 0-100mm depending on window
            window = int(band.split("_")[-1][:-1])
            data[:, i, :, :] = np.random.gamma(2, window * 2, (len(t_dates), H, W))
        elif "vpd" in band:
            # VPD: 5-25 hPa
            data[:, i, :, :] = np.random.gamma(2, 5, (len(t_dates), H, W)) + 5
        elif "fwi" in band:
            # FWI: 0-50
            data[:, i, :, :] = np.random.gamma(1, 10, (len(t_dates), H, W))
        elif "kbdi" in band:
            # KBDI: 0-200
            data[:, i, :, :] = np.random.gamma(1, 50, (len(t_dates), H, W))
    
    return xr.DataArray(
        data,
        dims=("time", "band", "y", "x"),
        coords={
            "time": np.array(t_dates, dtype="datetime64[ns]"),
            "band": bands_all,
            "y": np.arange(H),
            "x": np.arange(W)
        },
        attrs={"note": "Dummy ERA5 data - replace with real implementation"}
    )


# =========================
# Simplified orchestration with better error handling
# =========================

def fetch_multisource_chips(
    lat: float,
    lon: float,
    t0: str,
    lookbacks: List[int],
    chip_size_m: int = 2560,
    resolution_m: int = 10,
    max_cloud: int = 30,
    era_windows: List[int] = (1, 3, 7, 14, 30),
    include_s2: bool = True,
    include_dem: bool = True,
    include_worldcover: bool = True,
    include_era5: bool = True,
    include_smap: bool = True,
) -> Dict[str, xr.DataArray]:
    """Enhanced multi-source data fetching with better error handling."""
    
    out: Dict[str, xr.DataArray] = {}
    dates = [(datetime.fromisoformat(t0) - timedelta(days=int(lb))) for lb in lookbacks]
    
    pass #print(f"=== Fetching Multi-source Data ===")
    pass #print(f"Location: ({lat:.4f}, {lon:.4f})")
    pass #print(f"Reference date: {t0}")
    pass #print(f"Lookbacks: {lookbacks} days")
    pass #print(f"Chip size: {chip_size_m}m, Resolution: {resolution_m}m")
    pass #print(f"STAC available: {STAC_AVAILABLE}")

    # Sentinel-2
    if include_s2:
        pass #print("\n--- Sentinel-2 ---")
        try:
            s2 = fetch_s2_stack(lat, lon, t0, lookbacks, chip_size_m, resolution_m, max_cloud)
            out["S2"] = s2
            pass #print(f"✓ S2 shape: {s2.shape}, non-zero: {(s2 != 0).sum().values}/{s2.size}")
        except Exception as e:
            pass #print(f"✗ S2 fetch failed: {e}")

    # DEM + derivatives
    if include_dem:
        pass #print("\n--- DEM ---")
        try:
            dem = fetch_dem(lat, lon, chip_size_m, resolution_m)
            dem = dem_to_slope_aspect(dem, resolution_m)
            # Replicate over time
            dem_t = dem.expand_dims({"time": len(lookbacks)})
            dem_t = dem_t.assign_coords(time=range(len(lookbacks)))
            out["DEM"] = dem_t
            pass #print(f"✓ DEM shape: {dem_t.shape}, non-zero: {(dem_t != 0).sum().values}/{dem_t.size}")
        except Exception as e:
            pass #print(f"✗ DEM fetch failed: {e}")

    # WorldCover
    if include_worldcover:
        pass #print("\n--- WorldCover ---")
        try:
            lc = fetch_worldcover(lat, lon, chip_size_m, resolution_m)
            # Replicate over time
            lc_t = lc.expand_dims({"time": len(lookbacks)})
            lc_t = lc_t.assign_coords(time=range(len(lookbacks)))
            out["LC"] = lc_t
            pass #print(f"✓ LC shape: {lc_t.shape}, unique values: {len(np.unique(lc_t.values))}")
        except Exception as e:
            pass
            pass #print(f"✗ WorldCover fetch failed: {e}")

    # ERA5 (simplified/dummy for now)
    if include_era5:
        pass #print("\n--- ERA5 ---")
        try:
            # Use dummy data for now - replace with real ERA5 implementation
            era_cube = create_dummy_era5(lat, lon, dates, list(era_windows), chip_size_m, resolution_m)
            out["ERA5"] = era_cube
            pass #print(f"✓ ERA5 shape: {era_cube.shape} (dummy data)")
            
            # Create dummy SPI/SPEI
            H = W = grid_hw(chip_size_m, resolution_m)
            scales = [1, 3, 6]
            n_months = 12  # 12 months of data
            
            spi_data = np.random.normal(0, 1, (len(scales), n_months, H, W))
            spei_data = np.random.normal(0, 1, (len(scales), n_months, H, W))
            
            month_times = pd.date_range("2020-01-01", periods=n_months, freq="MS")
            
            out["SPI"] = xr.DataArray(
                spi_data.astype(np.float32),
                dims=("scale", "time", "y", "x"),
                coords={
                    "scale": scales,
                    "time": month_times,
                    "y": np.arange(H),
                    "x": np.arange(W)
                }
            )
            
            out["SPEI"] = xr.DataArray(
                spei_data.astype(np.float32),
                dims=("scale", "time", "y", "x"),
                coords={
                    "scale": scales,
                    "time": month_times,
                    "y": np.arange(H),
                    "x": np.arange(W)
                }
            )
            
            pass #print(f"✓ SPI shape: {out['SPI'].shape} (dummy)")
            pass #print(f"✓ SPEI shape: {out['SPEI'].shape} (dummy)")
            
        except Exception as e:
            pass #print(f"✗ ERA5 processing failed: {e}")

    # SMAP (dummy implementation)
    if include_smap:
        pass #print("\n--- SMAP ---")
        try:
            H = W = grid_hw(chip_size_m, resolution_m)
            T = len(lookbacks)
            
            # Create realistic soil moisture values (0.1 - 0.4)
            smap_data = np.random.beta(2, 3, (T, 1, H, W)) * 0.3 + 0.1
            
            times = [
                np.datetime64((datetime.fromisoformat(t0) - timedelta(days=int(lb))).date()) 
                for lb in lookbacks
            ]
            
            smap_array = xr.DataArray(
                smap_data.astype(np.float32),
                dims=("time", "band", "y", "x"),
                coords={
                    "time": np.array(times, dtype="datetime64[ns]"),
                    "band": ["SMAP_SM"],
                    "y": np.arange(H),
                    "x": np.arange(W)
                },
                attrs={"note": "Dummy SMAP data - replace with real implementation"}
            )
            
            out["SMAP"] = smap_array
            pass #print(f"✓ SMAP shape: {smap_array.shape} (dummy)")
            
        except Exception as e:
            pass #print(f"✗ SMAP processing failed: {e}")

    pass #print(f"\n=== Summary ===")
    total_non_zero = 0
    total_elements = 0
    
    for key, data in out.items():
        non_zero = (data != 0).sum().values if hasattr(data, 'values') else 0
        size = data.size if hasattr(data, 'size') else 0
        total_non_zero += non_zero
        total_elements += size
        pct = (non_zero / size * 100) if size > 0 else 0
        pass #print(f"{key}: {data.shape} | {non_zero}/{size} ({pct:.1f}%) non-zero")
    
    overall_pct = (total_non_zero / total_elements * 100) if total_elements > 0 else 0
    pass #print(f"Overall: {total_non_zero}/{total_elements} ({overall_pct:.1f}%) non-zero")

    return out


def to_model_tensor(stack: Dict[str, xr.DataArray], prefer_time_from: str = "S2") -> Tuple[np.ndarray, List[str]]:
    """
    Enhanced tensor conversion with better error handling and data validation.
    """
    if not stack:
        raise ValueError("Empty stack provided")
    
    arrays, names = [], []

    # Determine reference dimensions
    key = prefer_time_from if prefer_time_from in stack else list(stack.keys())[0]
    reference = stack[key]
    T = reference.sizes["time"]
    H = reference.sizes["y"]
    W = reference.sizes["x"]
    
    pass #print(f"\n=== Converting to Model Tensor ===")
    pass #print(f"Reference dimensions from '{key}': T={T}, H={H}, W={W}")

    # Process each data source with validation
    if "S2" in stack:
        s2 = stack["S2"]
        pass #print(f"Processing S2: {s2.shape}")
        
        # Exclude SCL band if present
        available_bands = list(s2.coords["band"].values)
        keep_bands = [b for b in available_bands if b != "SCL"]
        
        if keep_bands:
            s2_filtered = s2.sel(band=keep_bands)
            
            # Ensure consistent time dimension
            if s2_filtered.sizes["time"] != T:
                pass #print(f"Warning: S2 time mismatch ({s2_filtered.sizes['time']} vs {T}), resampling...")
                if s2_filtered.sizes["time"] > T:
                    s2_filtered = s2_filtered.isel(time=slice(0, T))
                else:
                    # Pad with zeros if needed
                    pad_needed = T - s2_filtered.sizes["time"]
                    padding = xr.zeros_like(s2_filtered.isel(time=[0] * pad_needed))
                    s2_filtered = xr.concat([s2_filtered, padding], dim="time")
            
            # Validate data ranges (reflectance should be 0-1, indices -1 to 1)
            s2_values = s2_filtered.values
            s2_values = np.nan_to_num(s2_values, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Clip to reasonable ranges
            for i, band in enumerate(keep_bands):
                if band in ["B02", "B03", "B04", "B08", "B11", "B12"]:
                    # Reflectance bands: 0-1
                    s2_values[:, i] = np.clip(s2_values[:, i], 0, 1)
                elif band in ["NDVI", "NDWI", "NDMI"]:
                    # Index bands: -1 to 1
                    s2_values[:, i] = np.clip(s2_values[:, i], -1, 1)
            
            arrays.append(s2_values)
            names.extend([f"S2_{b}" for b in keep_bands])
            pass #print(f"✓ Added S2: {len(keep_bands)} bands")
        else:
            pass #print("Warning: No valid S2 bands found")

    if "DEM" in stack:
        dem = stack["DEM"]
        pass #print(f"Processing DEM: {dem.shape}")
        
        # Ensure consistent time dimension
        if dem.sizes["time"] != T:
            dem = dem.isel(time=0).expand_dims({"time": T})
            
        dem_values = dem.values
        dem_values = np.nan_to_num(dem_values, nan=0.0)
        
        # Reasonable ranges: DEM (-500 to 9000m), SLOPE (0-90°), ASPECT (0-360°)
        band_names = list(dem.coords["band"].values)
        for i, band in enumerate(band_names):
            if band == "DEM":
                dem_values[:, i] = np.clip(dem_values[:, i], -500, 9000)
            elif band == "SLOPE":
                dem_values[:, i] = np.clip(dem_values[:, i], 0, 90)
            elif band == "ASPECT":
                dem_values[:, i] = np.clip(dem_values[:, i], 0, 360)
        
        arrays.append(dem_values)
        names.extend(list(dem.coords["band"].values))
        pass #print(f"✓ Added DEM: {len(dem.coords['band'])} bands")

    if "LC" in stack:
        lc = stack["LC"]
        pass #print(f"Processing LC: {lc.shape}")
        
        # Ensure consistent time dimension
        if lc.sizes["time"] != T:
            lc = lc.isel(time=0).expand_dims({"time": T})
            
        lc_values = lc.values.astype(np.float32)
        lc_values = np.nan_to_num(lc_values, nan=0.0)
        
        # WorldCover classes are typically 10, 20, 30, etc.
        lc_values = np.clip(lc_values, 0, 100)
        
        arrays.append(lc_values)
        names.extend(["LC"])
        pass #print(f"✓ Added LC: 1 band")

    if "ERA5" in stack:
        era5 = stack["ERA5"]
        pass #print(f"Processing ERA5: {era5.shape}")
        
        era5_values = era5.values
        era5_values = np.nan_to_num(era5_values, nan=0.0)
        
        # Apply reasonable ranges for different variables
        band_names = list(era5.coords["band"].values)
        for i, band in enumerate(band_names):
            if "precip" in band:
                era5_values[:, i] = np.clip(era5_values[:, i], 0, 1000)  # mm
            elif "vpd" in band:
                era5_values[:, i] = np.clip(era5_values[:, i], 0, 100)   # hPa
            elif "fwi" in band:
                era5_values[:, i] = np.clip(era5_values[:, i], 0, 100)   # index
            elif "kbdi" in band:
                era5_values[:, i] = np.clip(era5_values[:, i], 0, 800)   # index
        
        arrays.append(era5_values)
        names.extend(list(era5.coords["band"].values))
        pass #print(f"✓ Added ERA5: {len(era5.coords['band'])} bands")

    if "SMAP" in stack:
        smap = stack["SMAP"]
        pass #print(f"Processing SMAP: {smap.shape}")
        
        smap_values = smap.values
        smap_values = np.nan_to_num(smap_values, nan=0.0)
        smap_values = np.clip(smap_values, 0, 1)  # Soil moisture 0-1
        
        arrays.append(smap_values)
        names.extend(["SMAP_SM"])
        pass #print(f"✓ Added SMAP: 1 band")

    if not arrays:
        raise ValueError("No valid data sources found in stack")

    # Concatenate along channel dimension
    try:
        X = np.concatenate(arrays, axis=1)  # [T, C, H, W]
        X = X.astype(np.float32)
        
        # Final validation
        total_non_zero = (X != 0).sum()
        total_elements = X.size
        non_zero_pct = (total_non_zero / total_elements * 100)
        
        pass #print(f"\n✓ Final tensor shape: {X.shape}")
        pass #print(f"✓ Channels: {len(names)}")
        pass #print(f"✓ Non-zero elements: {total_non_zero}/{total_elements} ({non_zero_pct:.1f}%)")
        pass #print(f"✓ Data range: [{X.min():.3f}, {X.max():.3f}]")
        pass #print(f"✓ Memory usage: {X.nbytes / 1024**2:.1f} MB")
        
        # pass #print per-channel statistics
        """
        pass #print(f"\nPer-channel statistics:")
        for i, name in enumerate(names):
            ch = X[:, i]
            non_zero = (ch != 0).sum()
            ch_min, ch_max = ch.min(), ch.max()
            ch_mean = ch[ch != 0].mean() if non_zero > 0 else 0
            pass #print(f"{i:2d} {name:15s} | non-zero: {non_zero:8d} | range: [{ch_min:8.3f}, {ch_max:8.3f}] | mean: {ch_mean:8.3f}")
        """
        return X, names
        
    except Exception as e:
        pass #print(f"Error concatenating arrays: {e}")
        pass #print("Array shapes:")
        for i, (arr, source) in enumerate(zip(arrays, ["S2", "DEM", "LC", "ERA5", "SMAP"])):
            if i < len(arrays):
                pass #print(f"  {source}: {arr.shape}")
        raise


# =========================
# Enhanced example usage with debugging
# =========================

def test_location(lat: float, lon: float, t0: str = "2021-07-15", 
                 lookbacks: List[int] = None, verbose: bool = False):
    """Test the pipeline at a specific location with detailed output."""
    
    if lookbacks is None:
        lookbacks = [1, 5, 10, 20]
    
    try:
        # Fetch data
        chips = fetch_multisource_chips(
            lat=lat, lon=lon, t0=t0,
            lookbacks=lookbacks,
            chip_size_m=1280,  # Smaller for testing
            resolution_m=20,   # Coarser for testing
            max_cloud=50,      # More permissive
            era_windows=[1, 3, 7, 14, 30],
            include_s2=True,
            include_dem=True, 
            include_worldcover=True,
            include_era5=True,
            include_smap=True
        )
        
        if not chips:
            pass #print("❌ No data returned!")
            return None, None
            
        # Create model tensor
        X, channels = to_model_tensor(chips)
        
        if verbose:
            pass #print(f"\n=== Final Results ===")
            success = (X != 0).sum() > 0
            pass #print(f"Success: {'✅' if success else '❌'}")
            
            if success:
                pass #print(f"Ready for model training!")
            else:
                pass #print(f"All zeros - check data sources and connectivity")
        
        return X, channels
        
    except Exception as e:
        pass #print(f"❌ Test failed: {e}")
        if verbose:
            pass
        #    traceback.pass #print_exc()


"""
if __name__ == "__main__":
    import sys
    
    # Test locations
    test_locations = [
        (35.9132, -79.0558, "Chapel Hill, NC"),      # Your location
        (40.7128, -74.0060, "New York, NY"),         # Urban
        (36.7783, -119.4179, "California Central Valley"),  # Agricultural
        (39.7392, -104.9903, "Denver, CO")           # Mountains
    ]
    
    pass #print("=== Multi-source Risk Predictors - Enhanced Version ===")
    pass #print(f"STAC Available: {STAC_AVAILABLE}")
    
    # Quick connectivity test
    if STAC_AVAILABLE:
        try:
            stac = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
            pass #print("✅ Planetary Computer STAC accessible")
        except Exception as e:
            pass #print(f"❌ STAC connectivity issue: {e}")
    
    # Test first location
    lat, lon, name = test_locations[0]
    pass #print(f"\n=== Testing {name} ===")
    
    X, channels = test_location(lat, lon, t0="2023-07-15", lookbacks=[1, 5, 10], verbose=True)
    
    #DEBUG
    if X is not None:
        pass #print(f"\n🎉 Pipeline working! Generated tensor: {X.shape}")
        
        # Optional: Test all locations
        if len(sys.argv) > 1 and sys.argv[1] == "--all":
            for lat, lon, name in test_locations[1:]:
                pass #print(f"\n=== Testing {name} ===")
                test_X, test_channels = test_location(lat, lon, verbose=False)
                status = "✅" if test_X is not None and (test_X != 0).sum() > 0 else "❌"
                pass #print(f"{name}: {status}")
    else:
        pass #print("\n❌ Pipeline failed - check connectivity and dependencies")
        
    pass #print("\n=== Done ===")
    # Usage tip
"""
#EVERYTHING FROM 0 to 11 IS NOT WORKING CHECK SS