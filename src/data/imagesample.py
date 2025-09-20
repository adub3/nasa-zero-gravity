"""
risk_predictors_full.py

Build multi-temporal, multi-source predictors for drought & wildfire risk.

Includes:
- Sentinel-2 L2A: NDVI/NDWI/NDMI with SCL cloud mask
- DEM: DEM + SLOPE + ASPECT
- ESA WorldCover: land cover (categorical) band "LC"
- ERA5-Land: precip sums (1/3/7/14/30d), VPD means, FWI & KBDI window features,
             monthly SPI and SPEI (Hargreaves PET) at AOI scale
- SMAP surface soil moisture: best-effort placeholder; returns zeros if not available

Returns a dict of xarray.DataArray aligned to a common UTM grid and utilities
to combine into a single model tensor [T, C, H, W].

Author: you 🛰
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd
import xarray as xr
import warnings

from shapely.geometry import Point
from pyproj import Transformer

from pystac_client import Client
import planetary_computer as pc
import stackstac
from scipy.stats import gamma, norm


# =========================
# CRS / Grid helpers
# =========================

def utm_epsg_for(lat: float, lon: float) -> int:
    """Get UTM EPSG code for a lat/lon point."""
    zone = int((lon + 180) // 6) + 1
    hemisphere = "326" if lat >= 0 else "327"  # Northern: 326**, Southern: 327**
    return int(f"{hemisphere}{zone:02d}")

def square_bounds_in_crs(lat: float, lon: float, half_size_m: float, epsg: int) -> Tuple[float,float,float,float]:
    """Convert lat/lon center + half-size to bounds in target CRS."""
    to_crs = Transformer.from_crs(4326, epsg, always_xy=True)
    cx, cy = to_crs.transform(lon, lat)
    return (cx - half_size_m, cy - half_size_m, cx + half_size_m, cy + half_size_m)

def grid_hw(chip_size_m: int, resolution_m: int) -> int:
    """Calculate grid height/width from chip size and resolution."""
    return int(round(chip_size_m / resolution_m))


# =========================
# Sentinel-2 utilities
# =========================

S2_BANDS = ["B02","B03","B04","B08","B11","B12","SCL"]  # SWIR for NDMI
SCL_KEEP = {2,4,5,6,7}  # Basic mask: Dark, Veg, Bare, Water, Unclass

def apply_scl_mask(chip: xr.DataArray) -> xr.DataArray:
    """Apply SCL cloud mask to Sentinel-2 data."""
    if "band" not in chip.coords or "SCL" not in chip.coords["band"].values:
        return chip
    
    scl = chip.sel(band="SCL")
    mask = xr.apply_ufunc(np.isin, scl, np.array(list(SCL_KEEP)), vectorize=True)

    # Broadcast mask to all bands
    mask_expanded = mask.expand_dims({"band": chip.sizes["band"]})
    mask_aligned = mask_expanded.transpose(*chip.dims)
    return xr.where(mask_aligned, chip, 0)

def compute_indices(chip: xr.DataArray) -> xr.DataArray:
    """Compute NDVI, NDWI, NDMI from Sentinel-2 bands."""
    def get_band(b): 
        return chip.sel(band=b) if b in chip.coords["band"] else None
    
    B03 = get_band("B03")
    B04 = get_band("B04")
    B08 = get_band("B08")
    B11 = get_band("B11")

    def safe_idx(n, d): 
        return xr.where((n + d) != 0, (n - d) / (n + d), 0).astype("float32")
    
    additional_bands = []
    
    if B08 is not None and B04 is not None:
        ndvi = safe_idx(B08, B04).assign_coords(band="NDVI").expand_dims("band")
        additional_bands.append(ndvi)
    
    if B03 is not None and B08 is not None:
        ndwi = safe_idx(B03, B08).assign_coords(band="NDWI").expand_dims("band")
        additional_bands.append(ndwi)
    
    if B11 is not None and B08 is not None:
        ndmi = safe_idx(B08, B11).assign_coords(band="NDMI").expand_dims("band")
        additional_bands.append(ndmi)
    
    if additional_bands:
        return xr.concat([chip] + additional_bands, dim="band")
    else:
        return chip

def fetch_s2_stack(
    lat: float, lon: float, t0: str, lookbacks: List[int],
    chip_size_m: int, resolution_m: int, max_cloud_pct: int = 30
) -> xr.DataArray:
    """Fetch Sentinel-2 stack for given lookback periods."""
    stac = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    epsg = utm_epsg_for(lat, lon)
    xmin, ymin, xmax, ymax = square_bounds_in_crs(lat, lon, chip_size_m/2, epsg)
    H = W = grid_hw(chip_size_m, resolution_m)

    out_chips, out_times = [], []
    
    for lb in lookbacks:
        tref = (datetime.fromisoformat(t0) - timedelta(days=int(lb)))
        t_start = (tref - timedelta(days=3)).isoformat()
        t_end = (tref + timedelta(days=3)).isoformat()
        
        try:
            items = list(stac.search(
                collections=["sentinel-2-l2a"],
                datetime=f"{t_start}/{t_end}",
                intersects={"type":"Point","coordinates":[lon,lat]},
                query={"eo:cloud_cover": {"lt": max_cloud_pct}},
                limit=40
            ).items())

            if not items:
                # Create empty chip with correct structure
                chip = xr.DataArray(
                    np.zeros((len(S2_BANDS)+3, H, W), dtype=np.float32),
                    dims=("band","y","x"),
                    coords={"band": S2_BANDS+["NDVI","NDWI","NDMI"]},
                    attrs={"crs": f"EPSG:{epsg}"}
                )
                out_chips.append(chip)
                out_times.append(np.datetime64(tref.date(), 'ns'))
                continue

            # Sort by cloud cover
            items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100.0))
            signed = [pc.sign(it).to_dict() for it in items[:6]]
            
            stk = stackstac.stack(
                signed, assets=S2_BANDS, resolution=resolution_m,
                bounds=(xmin, ymin, xmax, ymax), epsg=epsg,
                dtype="float32", fill_value=0, chunks=None
            )
            
            chip = stk.isel(item=0).drop_vars("item", errors="ignore").compute()
            chip = apply_scl_mask(chip)
            chip = compute_indices(chip)
            
            out_chips.append(chip)
            out_times.append(np.datetime64(items[0].properties["datetime"], 'ns'))
            
        except Exception as e:
            print(f"Warning: Failed to fetch S2 for lookback {lb}: {e}")
            # Create empty chip
            chip = xr.DataArray(
                np.zeros((len(S2_BANDS)+3, H, W), dtype=np.float32),
                dims=("band","y","x"),
                coords={"band": S2_BANDS+["NDVI","NDWI","NDMI"]},
                attrs={"crs": f"EPSG:{epsg}"}
            )
            out_chips.append(chip)
            out_times.append(np.datetime64(tref.date(), 'ns'))

    return xr.concat(out_chips, dim="time").assign_coords(time=np.array(out_times, dtype="datetime64[ns]"))


# =========================
# DEM → Slope & Aspect
# =========================

def fetch_dem(lat: float, lon: float, chip_size_m: int, resolution_m: int) -> xr.DataArray:
    """Fetch DEM data from Copernicus DEM."""
    stac = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    epsg = utm_epsg_for(lat, lon)
    xmin, ymin, xmax, ymax = square_bounds_in_crs(lat, lon, chip_size_m/2, epsg)
    
    try:
        items = list(stac.search(
            collections=["cop-dem-glo-30"],
            intersects=Point(lon, lat).__geo_interface__,
            limit=5
        ).items())
        
        if not items:
            H = W = grid_hw(chip_size_m, resolution_m)
            return xr.DataArray(
                np.zeros((1, H, W), np.float32), 
                dims=("band","y","x"), 
                coords={"band":["DEM"]},
                attrs={"crs": f"EPSG:{epsg}"}
            )
        
        dem = stackstac.stack(
            [pc.sign(items[0]).to_dict()], assets=["data"],
            resolution=resolution_m, bounds=(xmin, ymin, xmax, ymax), epsg=epsg,
            dtype="float32", fill_value=np.nan, chunks=None
        ).isel(item=0).drop_vars("item", errors="ignore")
        
        return dem.assign_coords(band=["DEM"]).fillna(0)
        
    except Exception as e:
        print(f"Warning: Failed to fetch DEM: {e}")
        H = W = grid_hw(chip_size_m, resolution_m)
        return xr.DataArray(
            np.zeros((1, H, W), np.float32), 
            dims=("band","y","x"), 
            coords={"band":["DEM"]},
            attrs={"crs": f"EPSG:{epsg}"}
        )

def dem_to_slope_aspect(dem_da: xr.DataArray, resolution_m: int) -> xr.DataArray:
    """Compute slope and aspect from DEM."""
    dem = dem_da.sel(band="DEM")
    
    # Compute gradients
    dz_dy, dz_dx = np.gradient(dem.values, resolution_m, resolution_m)
    
    # Calculate slope in degrees
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    
    # Calculate aspect in degrees
    aspect_rad = np.arctan2(-dz_dx, dz_dy)
    aspect_rad = np.where(aspect_rad < 0, aspect_rad + 2*np.pi, aspect_rad)
    aspect_deg = np.degrees(aspect_rad)
    
    # Create DataArrays
    slope = xr.DataArray(slope_deg, dims=("y","x"), coords={"y": dem.y, "x": dem.x})
    aspect = xr.DataArray(aspect_deg, dims=("y","x"), coords={"y": dem.y, "x": dem.x})
    
    out = xr.concat([
        dem.assign_coords(band="DEM").expand_dims("band"),
        slope.assign_coords(band="SLOPE").expand_dims("band"),
        aspect.assign_coords(band="ASPECT").expand_dims("band"),
    ], dim="band")
    
    return out


# =========================
# WorldCover (LC)
# =========================

def fetch_worldcover(lat: float, lon: float, chip_size_m: int, resolution_m: int) -> xr.DataArray:
    """Fetch ESA WorldCover land cover data."""
    stac = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    epsg = utm_epsg_for(lat, lon)
    xmin, ymin, xmax, ymax = square_bounds_in_crs(lat, lon, chip_size_m/2, epsg)
    
    try:
        items = list(stac.search(
            collections=["esa-worldcover"],
            intersects={"type":"Point","coordinates":[lon,lat]},
            limit=5
        ).items())
        
        if not items:
            H = W = grid_hw(chip_size_m, resolution_m)
            return xr.DataArray(
                np.zeros((1, H, W), np.uint8), 
                dims=("band","y","x"), 
                coords={"band":["LC"]},
                attrs={"crs": f"EPSG:{epsg}"}
            )
        
        lc = stackstac.stack(
            [pc.sign(items[0]).to_dict()], assets=["map"],
            resolution=resolution_m, bounds=(xmin, ymin, xmax, ymax), epsg=epsg,
            dtype="uint8", fill_value=0, chunks=None
        ).isel(item=0).drop_vars("item", errors="ignore")
        
        return lc.assign_coords(band=["LC"]).astype("uint8")
        
    except Exception as e:
        print(f"Warning: Failed to fetch WorldCover: {e}")
        H = W = grid_hw(chip_size_m, resolution_m)
        return xr.DataArray(
            np.zeros((1, H, W), np.uint8), 
            dims=("band","y","x"), 
            coords={"band":["LC"]},
            attrs={"crs": f"EPSG:{epsg}"}
        )


# =========================
# ERA5-Land: Precip, VPD, SPI/SPEI, FWI, KBDI
# =========================

def _svp_hPa(temp_C: xr.DataArray) -> xr.DataArray:
    """Calculate saturated vapor pressure in hPa."""
    # Replace deprecated xarray.ufuncs with numpy ufuncs
    return 6.112 * np.exp((17.67*temp_C)/(temp_C+243.5))

def vpd_from_t_and_td(temp_K: xr.DataArray, dewpoint_K: xr.DataArray) -> xr.DataArray:
    """Calculate VPD from temperature and dewpoint."""
    T = temp_K - 273.15
    Td = dewpoint_K - 273.15
    es = _svp_hPa(T)
    ea = _svp_hPa(Td)
    return (es - ea).clip(min=0).astype("float32")

def fetch_era5_hourly_cube(lat: float, lon: float, chip_size_m: int, resolution_m: int, 
                          start: datetime, end: datetime) -> xr.DataArray:
    """Fetch ERA5-Land hourly data."""
    stac = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    epsg = utm_epsg_for(lat, lon)
    xmin, ymin, xmax, ymax = square_bounds_in_crs(lat, lon, chip_size_m/2, epsg)
    
    try:
        items = list(stac.search(
            collections=["era5-land"],
            datetime=f"{start.isoformat()}/{end.isoformat()}",
            intersects={"type":"Point","coordinates":[lon,lat]},
            limit=1000
        ).items())
        
        if not items:
            H = W = grid_hw(chip_size_m, resolution_m)
            times = pd.date_range(start, end, freq="h")
            return xr.DataArray(
                np.zeros((len(times), 3, H, W), np.float32),
                dims=("hour","band","y","x"),
                coords={
                    "hour": times.values.astype("datetime64[ns]"),
                    "band": ["total_precipitation","2m_temperature","2m_dewpoint_temperature"]
                },
                attrs={"crs": f"EPSG:{epsg}"}
            )
        
        signed = [pc.sign(it).to_dict() for it in items]
        era = stackstac.stack(
            signed,
            assets=["total_precipitation","2m_temperature","2m_dewpoint_temperature",
                   "10m_u_component_of_wind","10m_v_component_of_wind","surface_pressure","2m_relative_humidity"],
            resolution=resolution_m, bounds=(xmin, ymin, xmax, ymax), epsg=epsg,
            dtype="float32", fill_value=np.nan, chunks=None
        )
        
        times = [np.datetime64(it["properties"]["start_datetime"], 'ns') for it in signed]
        era = era.assign_coords(item=np.array(times, dtype="datetime64[ns]")).rename({"item":"hour"}).sortby("hour")
        return era
        
    except Exception as e:
        print(f"Warning: Failed to fetch ERA5: {e}")
        H = W = grid_hw(chip_size_m, resolution_m)
        times = pd.date_range(start, end, freq="h")
        return xr.DataArray(
            np.zeros((len(times), 3, H, W), np.float32),
            dims=("hour","band","y","x"),
            coords={
                "hour": times.values.astype("datetime64[ns]"),
                "band": ["total_precipitation","2m_temperature","2m_dewpoint_temperature"]
            },
            attrs={"crs": f"EPSG:{epsg}"}
        )

def aggregate_era_features(lat: float, lon: float, t_dates: List[datetime], 
                          windows: List[int], chip_size_m: int, resolution_m: int) -> Tuple[xr.DataArray, xr.Dataset]:
    """Aggregate ERA5 features for drought/fire risk indicators."""
    maxW = max(windows)
    start = (min(t_dates) - timedelta(days=maxW)).replace(hour=0, minute=0, second=0, microsecond=0)
    end = max(t_dates).replace(hour=23, minute=59, second=59)
    
    era = fetch_era5_hourly_cube(lat, lon, chip_size_m, resolution_m, start, end)

    # Extract variables
    tp = era.sel(band="total_precipitation").drop_vars("band", errors="ignore")
    t2m = era.sel(band="2m_temperature").drop_vars("band", errors="ignore")
    d2m = era.sel(band="2m_dewpoint_temperature").drop_vars("band", errors="ignore")
    
    # Wind speed from u,v components
    if "band" in era.coords and "10m_u_component_of_wind" in era.coords["band"].values:
        u = era.sel(band="10m_u_component_of_wind").drop_vars("band", errors="ignore")
        v = era.sel(band="10m_v_component_of_wind").drop_vars("band", errors="ignore")
        wind_ms = np.sqrt(u**2 + v**2)
    else:
        wind_ms = xr.zeros_like(t2m)

    # Calculate VPD
    vpd_h = vpd_from_t_and_td(t2m, d2m)

    # Initialize output cube
    H = W = grid_hw(chip_size_m, resolution_m)
    bands_all = []
    for w in windows:
        bands_all.extend([f"precip_sum_{w}d", f"vpd_mean_{w}d", f"fwi_mean_{w}d", f"kbdi_mean_{w}d"])
    
    cube = xr.DataArray(
        np.zeros((len(t_dates), len(bands_all), H, W), np.float32),
        dims=("time","band","y","x"),
        coords={
            "time": np.array(t_dates, dtype="datetime64[ns]"), 
            "band": bands_all
        }
    )

    # Create daily aggregates
    daily_ds = xr.Dataset({
        "precip_mm": (tp * 1000.0).resample(hour="1D").sum(),
        "t2m_mean": (t2m - 273.15).resample(hour="1D").mean(),
        "t2m_min": (t2m - 273.15).resample(hour="1D").min(),
        "t2m_max": (t2m - 273.15).resample(hour="1D").max(),
        "rh_mean": era.sel(band="2m_relative_humidity", missing_dims="ignore").resample(hour="1D").mean() if ("band" in era.coords and "2m_relative_humidity" in era.coords["band"].values) else xr.full_like((t2m*0).resample(hour="1D").mean(), 50),
        "wind_ms": wind_ms.resample(hour="1D").mean(),
        "vpd_hpa": vpd_h.resample(hour="1D").mean(),
    }).rename({"hour":"day"})

    # Compute FWI and KBDI series
    try:
        fwi_day = compute_fwi_series(daily_ds)
        kbdi_day = compute_kbdi_series(daily_ds)
    except Exception as e:
        print(f"Warning: FWI/KBDI computation failed: {e}")
        fwi_day = xr.zeros_like(daily_ds["precip_mm"])
        kbdi_day = xr.zeros_like(daily_ds["precip_mm"])

    # Aggregate features for each date and window
    for ti, d in enumerate(t_dates):
        for w in windows:
            start_date = d - timedelta(days=w)
            end_date = d
            
            # Get time slice
            time_slice = slice(np.datetime64(start_date, 'ns'), np.datetime64(end_date, 'ns'))
            
            try:
                # Precipitation sum
                p_mm = tp.sel(hour=time_slice).sum("hour") * 1000.0
                p_mm = p_mm.fillna(0)
                
                # VPD mean
                vpd_mean = vpd_h.sel(hour=time_slice).mean("hour")
                vpd_mean = vpd_mean.fillna(0)
                
                # FWI and KBDI means
                day_slice = slice(np.datetime64(start_date.date(), 'ns'), np.datetime64(end_date.date(), 'ns'))
                fwi_mean = fwi_day.sel(day=day_slice).mean("day").fillna(0)
                kbdi_mean = kbdi_day.sel(day=day_slice).mean("day").fillna(0)
                
                # Assign to cube
                cube.loc[dict(time=np.datetime64(d, 'ns'), band=f"precip_sum_{w}d")] = p_mm.values
                cube.loc[dict(time=np.datetime64(d, 'ns'), band=f"vpd_mean_{w}d")] = vpd_mean.values
                cube.loc[dict(time=np.datetime64(d, 'ns'), band=f"fwi_mean_{w}d")] = fwi_mean.values
                cube.loc[dict(time=np.datetime64(d, 'ns'), band=f"kbdi_mean_{w}d")] = kbdi_mean.values
                
            except Exception as e:
                print(f"Warning: Failed to aggregate for date {d}, window {w}: {e}")
                # Fill with zeros
                cube.loc[dict(time=np.datetime64(d, 'ns'), band=f"precip_sum_{w}d")] = 0
                cube.loc[dict(time=np.datetime64(d, 'ns'), band=f"vpd_mean_{w}d")] = 0
                cube.loc[dict(time=np.datetime64(d, 'ns'), band=f"fwi_mean_{w}d")] = 0
                cube.loc[dict(time=np.datetime64(d, 'ns'), band=f"kbdi_mean_{w}d")] = 0

    return cube, daily_ds


# ---------- SPI / SPEI ----------

def spi_from_precip_monthly(p_mm_monthly: xr.DataArray, scale: int = 3) -> xr.DataArray:
    """Compute Standardized Precipitation Index."""
    try:
        acc = p_mm_monthly.rolling(time=scale, min_periods=scale).sum()
        
        # Get valid (positive) values for fitting
        valid_data = acc.where(acc > 0).dropna("time")
        if valid_data.size < 24:  # Need at least 2 years of data
            return xr.full_like(acc, fill_value=np.nan)
        
        # Fit gamma distribution
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Use method of moments for more stable fitting
                valid_values = valid_data.values.flatten()
                valid_values = valid_values[valid_values > 0]
                
                if len(valid_values) < 10:
                    return xr.full_like(acc, fill_value=np.nan)
                
                # Method of moments for gamma parameters
                mean_val = np.mean(valid_values)
                var_val = np.var(valid_values)
                
                if var_val <= 0 or mean_val <= 0:
                    return xr.full_like(acc, fill_value=np.nan)
                
                scale_param = var_val / mean_val
                shape_param = mean_val / scale_param
                
                # Compute CDF and convert to standard normal
                cdf_vals = gamma.cdf(acc.fillna(0).clip(min=0).values, 
                                   a=shape_param, scale=scale_param)
                cdf_vals = np.clip(cdf_vals, 1e-6, 1-1e-6)
                spi_vals = norm.ppf(cdf_vals)
                
                return xr.DataArray(spi_vals, coords=acc.coords, dims=acc.dims)
                
            except Exception as e:
                print(f"Warning: SPI computation failed: {e}")
                return xr.full_like(acc, fill_value=np.nan)
                
    except Exception as e:
        print(f"Warning: SPI computation failed: {e}")
        return xr.full_like(p_mm_monthly, fill_value=np.nan)

def extraterrestrial_radiation_MJm2day(lat_deg: float, day_of_year: int) -> float:
    """Calculate extraterrestrial radiation using FAO-56 method."""
    try:
        Gsc = 0.0820  # Solar constant MJ m-2 min-1
        phi = math.radians(lat_deg)
        dr = 1 + 0.033 * math.cos(2*math.pi*day_of_year/365)
        delta = 0.409 * math.sin(2*math.pi*day_of_year/365 - 1.39)
        ws = math.acos(-math.tan(phi)*math.tan(delta))
        
        Ra = (24*60/math.pi) * Gsc * dr * (
            ws*math.sin(phi)*math.sin(delta) + 
            math.cos(phi)*math.cos(delta)*math.sin(ws)
        )
        return max(Ra, 0)  # Ensure non-negative
    except Exception:
        return 15.0  # Default fallback value

def hargreaves_pet_mm_day(tmean_C: float, tmin_C: float, tmax_C: float, 
                         lat_deg: float, doy: int) -> float:
    """Calculate PET using Hargreaves method."""
    try:
        Ra = extraterrestrial_radiation_MJm2day(lat_deg, doy)
        temp_range = max(tmax_C - tmin_C, 0)
        pet = 0.0023 * (tmean_C + 17.8) * math.sqrt(temp_range) * Ra
        return max(pet, 0)  # Ensure non-negative
    except Exception:
        return 0.0

def spei_from_precip_pet_monthly(p_mm_daily: xr.DataArray, tmean_C: xr.DataArray, 
                                tmin_C: xr.DataArray, tmax_C: xr.DataArray, 
                                lat_deg: float, scale: int = 3) -> xr.DataArray:
    """Compute Standardized Precipitation-Evapotranspiration Index."""
    try:
        # Calculate daily PET
        days = pd.to_datetime(p_mm_daily["day"].values)
        doy_array = np.array([d.timetuple().tm_yday for d in days])
        
        # Vectorized PET calculation
        pet_values = []
        for i, doy in enumerate(doy_array):
            tmean_val = float(tmean_C.isel(day=i).values)
            tmin_val = float(tmin_C.isel(day=i).values)  
            tmax_val = float(tmax_C.isel(day=i).values)
            pet_val = hargreaves_pet_mm_day(tmean_val, tmin_val, tmax_val, lat_deg, doy)
            pet_values.append(pet_val)
        
        pet = xr.DataArray(pet_values, dims=("day",), coords={"day": p_mm_daily["day"]})
        
        # Calculate water balance (P - PET) and monthly sums
        water_balance = p_mm_daily - pet
        monthly_balance = water_balance.resample(day="MS").sum().rename({"day":"time"})
        
        # Apply SPI method to water balance
        return spi_from_precip_monthly(monthly_balance, scale)
        
    except Exception as e:
        print(f"Warning: SPEI computation failed: {e}")
        return xr.full_like(p_mm_daily.resample(day="MS").sum().rename({"day":"time"}), 
                          fill_value=np.nan)


# ---------- FWI & KBDI (simplified implementations) ----------

def compute_fwi_series(ds_daily: xr.Dataset) -> xr.DataArray:
    """
    Simplified Fire Weather Index computation.
    Returns daily FWI values based on temperature, humidity, wind, and precipitation.
    """
    try:
        P = ds_daily["precip_mm"].fillna(0)
        T = ds_daily["t2m_mean"].fillna(20)
        H = ds_daily["rh_mean"].fillna(50)
        W_kmh = (ds_daily["wind_ms"].fillna(2) * 3.6)  # Convert m/s to km/h

        # Initialize with standard overwintered codes
        num_days = P.sizes["day"]
        fwi_values = np.zeros(num_days, dtype=np.float32)
        
        # Simple FWI approximation based on meteorological conditions
        for i in range(num_days):
            try:
                p_val = float(P.isel(day=i).values)
                t_val = float(T.isel(day=i).values)
                h_val = float(H.isel(day=i).values)
                w_val = float(W_kmh.isel(day=i).values)
                
                # Simplified FWI calculation
                drought_factor = max(0, (t_val - 10) / 30) * max(0, (100 - h_val) / 100)
                wind_factor = min(w_val / 30, 2.0)  # Cap wind effect
                precip_factor = max(0, 1 - (p_val / 10))  # Significant rain reduces FWI
                fwi = drought_factor * wind_factor * precip_factor * 10
                fwi_values[i] = max(0, min(fwi, 100))  # Cap between 0-100
                
            except Exception:
                fwi_values[i] = 0.0

        fwi_da = xr.DataArray(
            fwi_values, 
            coords={"day": ds_daily["precip_mm"]["day"]}, 
            dims=("day",)
        )
        return fwi_da.broadcast_like(ds_daily["precip_mm"])
        
    except Exception as e:
        print(f"Warning: FWI computation failed: {e}")
        return xr.zeros_like(ds_daily["precip_mm"])

def compute_kbdi_series(ds_daily: xr.Dataset) -> xr.DataArray:
    """
    Simplified Keetch-Byram Drought Index computation.
    Returns daily KBDI values based on precipitation and temperature.
    """
    try:
        P = ds_daily["precip_mm"].fillna(0)
        T = ds_daily["t2m_mean"].fillna(20)
        
        num_days = P.sizes["day"]
        kbdi_values = np.zeros(num_days, dtype=np.float32)
        
        kbdi_current = 0.0
        for i in range(num_days):
            try:
                p_val = float(P.isel(day=i).values)
                t_val = float(T.isel(day=i).values)
                
                if p_val > 5:  # Significant rainfall threshold
                    kbdi_current = max(0, kbdi_current - (p_val - 5) * 0.8)
                
                if t_val > 15:  # Above threshold temperature
                    drying_factor = (t_val - 15) * 0.5
                    kbdi_current = min(kbdi_current + drying_factor, 203.2)  # Cap at ~8 inches
                
                kbdi_values[i] = kbdi_current
                
            except Exception:
                kbdi_values[i] = kbdi_current
        
        kbdi_da = xr.DataArray(
            kbdi_values,
            coords={"day": ds_daily["precip_mm"]["day"]},
            dims=("day",)
        )
        return kbdi_da.broadcast_like(ds_daily["precip_mm"])
        
    except Exception as e:
        print(f"Warning: KBDI computation failed: {e}")
        return xr.zeros_like(ds_daily["precip_mm"])


# =========================
# SMAP (placeholder best-effort)
# =========================

def fetch_smap_surface_sm(lat: float, lon: float, t0: str, lookbacks: List[int], 
                         chip_size_m: int, resolution_m: int) -> xr.DataArray:
    """
    Placeholder that returns zeros with the correct shape.
    Integrate real SMAP access (e.g., via GEE or STAC) when available.
    """
    H = W = grid_hw(chip_size_m, resolution_m)
    T = len(lookbacks)
    times = [
        np.datetime64((datetime.fromisoformat(t0) - timedelta(days=int(lb))).date(), 'ns') 
        for lb in lookbacks
    ]
    return xr.DataArray(
        np.zeros((T, 1, H, W), np.float32),
        dims=("time", "band", "y", "x"),
        coords={"time": np.array(times, dtype="datetime64[ns]"), "band": ["SMAP_SM"]},
        attrs={"note": "Placeholder - zeros returned"}
    )


# =========================
# Orchestration
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
    """
    Build dict of aligned DataArrays:
      - "S2":    [T, bands, H, W] (incl. NDVI/NDWI/NDMI; SCL retained but you can drop before modeling)
      - "DEM":   [T, 3, H, W]     bands DEM/SLOPE/ASPECT replicated over T
      - "LC":    [T, 1, H, W]     WorldCover categorical band replicated over T
      - "ERA5":  [T, bands, H, W] precip/vpd/fwi/kbdi per window
      - "SMAP":  [T, 1, H, W]     surface soil moisture (zeros if not available)
      - "SPI":   [scales, time, H, W] monthly SPI (1,3,6) at AOI (replicated over grid)
      - "SPEI":  [scales, time, H, W] monthly SPEI (1,3,6)
    """
    out: Dict[str, xr.DataArray] = {}
    dates = [(datetime.fromisoformat(t0) - timedelta(days=int(lb))) for lb in lookbacks]
    
    print(f"Fetching data for location ({lat:.4f}, {lon:.4f}) with {len(lookbacks)} time steps")

    # Sentinel-2
    if include_s2:
        print("Fetching Sentinel-2 data...")
        try:
            s2 = fetch_s2_stack(lat, lon, t0, lookbacks, chip_size_m, resolution_m, max_cloud)
            out["S2"] = s2
            print(f"✓ S2 shape: {s2.shape}")
        except Exception as e:
            print(f"✗ S2 fetch failed: {e}")

    # DEM + derivatives
    if include_dem:
        print("Fetching DEM data...")
        try:
            dem = fetch_dem(lat, lon, chip_size_m, resolution_m)
            dem = dem_to_slope_aspect(dem, resolution_m)
            dem_t = dem.expand_dims({"time": len(lookbacks)}).assign_coords(time=range(len(lookbacks)))
            out["DEM"] = dem_t
            print(f"✓ DEM shape: {dem_t.shape}")
        except Exception as e:
            print(f"✗ DEM fetch failed: {e}")

    # WorldCover
    if include_worldcover:
        print("Fetching WorldCover data...")
        try:
            lc = fetch_worldcover(lat, lon, chip_size_m, resolution_m)
            lc_t = lc.expand_dims({"time": len(lookbacks)}).assign_coords(time=range(len(lookbacks)))
            out["LC"] = lc_t
            print(f"✓ LC shape: {lc_t.shape}")
        except Exception as e:
            print(f"✗ WorldCover fetch failed: {e}")

    # ERA5 (precip/vpd/fwi/kbdi) + SPI/SPEI
    if include_era5:
        print("Fetching ERA5 data and computing drought indices...")
        try:
            era_cube, daily_ds = aggregate_era_features(
                lat, lon, dates, list(era_windows), chip_size_m, resolution_m
            )
            out["ERA5"] = era_cube
            print(f"✓ ERA5 shape: {era_cube.shape}")

            # Monthly SPI/SPEI (scales 1/3/6)
            print("Computing SPI/SPEI indices...")
            try:
                precip_monthly = daily_ds["precip_mm"].resample(day="MS").sum().rename({"day": "time"})
                tmean = daily_ds["t2m_mean"]
                tmin = daily_ds["t2m_min"] 
                tmax = daily_ds["t2m_max"]

                spi_list, spei_list, scales = [], [], [1, 3, 6]
                
                for sc in scales:
                    try:
                        spi_sc = spi_from_precip_monthly(precip_monthly, sc)
                        spei_sc = spei_from_precip_pet_monthly(
                            daily_ds["precip_mm"], tmean, tmin, tmax, lat, sc
                        )
                        spi_list.append(spi_sc.expand_dims("scale"))
                        spei_list.append(spei_sc.expand_dims("scale"))
                    except Exception as e:
                        print(f"Warning: Failed to compute SPI/SPEI for scale {sc}: {e}")
                        dummy_spi = xr.full_like(precip_monthly, np.nan).expand_dims("scale")
                        dummy_spei = xr.full_like(precip_monthly, np.nan).expand_dims("scale")
                        spi_list.append(dummy_spi)
                        spei_list.append(dummy_spei)
                
                if spi_list and spei_list:
                    SPI = xr.concat(spi_list, dim="scale").assign_coords(scale=scales)
                    SPEI = xr.concat(spei_list, dim="scale").assign_coords(scale=scales)

                    H = W = grid_hw(chip_size_m, resolution_m)
                    SPI_grid = SPI.expand_dims({"y": H, "x": W}).fillna(0)
                    SPEI_grid = SPEI.expand_dims({"y": H, "x": W}).fillna(0)
                    
                    out["SPI"] = SPI_grid
                    out["SPEI"] = SPEI_grid
                    print(f"✓ SPI shape: {SPI_grid.shape}")
                    print(f"✓ SPEI shape: {SPEI_grid.shape}")
                
            except Exception as e:
                print(f"✗ SPI/SPEI computation failed: {e}")
                
        except Exception as e:
            print(f"✗ ERA5 fetch failed: {e}")

    # SMAP
    if include_smap:
        print("Fetching SMAP data (placeholder)...")
        try:
            smap_data = fetch_smap_surface_sm(lat, lon, t0, lookbacks, chip_size_m, resolution_m)
            out["SMAP"] = smap_data
            print(f"✓ SMAP shape: {smap_data.shape}")
        except Exception as e:
            print(f"✗ SMAP fetch failed: {e}")

    return out


def to_model_tensor(stack: Dict[str, xr.DataArray], prefer_time_from: str = "S2") -> Tuple[np.ndarray, List[str]]:
    """
    Concatenate channels from available sources into a single numpy tensor [T, C, H, W].
    Channel order: S2 bands (excluding SCL), indices, DEM triplet, LC (categorical id), ERA5 window feats, SMAP.
    SPI/SPEI are time series at monthly step — not merged here by default (different time axis).
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
    
    print(f"Reference dimensions from '{key}': T={T}, H={H}, W={W}")

    # Process each data source
    if "S2" in stack:
        s2 = stack["S2"]
        # Exclude SCL band if present
        keep_bands = [b for b in s2.coords["band"].values if b != "SCL"]
        s2_filtered = s2.sel(band=keep_bands)
        
        if s2_filtered.sizes["time"] != T:
            print(f"Warning: S2 time dimension mismatch ({s2_filtered.sizes['time']} vs {T})")
        
        arrays.append(s2_filtered.values)
        names.extend([f"S2_{b}" for b in keep_bands])
        print(f"Added S2: {len(keep_bands)} bands")

    if "DEM" in stack:
        dem = stack["DEM"]
        if dem.sizes["time"] != T:
            dem = dem.isel(time=0).expand_dims({"time": T})
        arrays.append(dem.values)
        names.extend(list(dem.coords["band"].values))
        print(f"Added DEM: {len(dem.coords['band'])} bands")

    if "LC" in stack:
        lc = stack["LC"]
        if lc.sizes["time"] != T:
            lc = lc.isel(time=0).expand_dims({"time": T})
        arrays.append(lc.values)
        names.extend(["LC"])
        print(f"Added LC: 1 band")

    if "ERA5" in stack:
        era5 = stack["ERA5"]
        arrays.append(era5.values)
        names.extend(list(era5.coords["band"].values))
        print(f"Added ERA5: {len(era5.coords['band'])} bands")

    if "SMAP" in stack:
        smap = stack["SMAP"]
        arrays.append(smap.values)
        names.extend(["SMAP_SM"])
        print(f"Added SMAP: 1 band")

    if not arrays:
        raise ValueError("No valid data sources found in stack")

    # Concatenate along channel dimension
    try:
        X = np.concatenate(arrays, axis=1)  # [T, C, H, W]
        print(f"Final tensor shape: {X.shape} with {len(names)} channels")
        return X.astype("float32"), names
    except Exception as e:
        print(f"Error concatenating arrays: {e}")
        for i, arr in enumerate(arrays):
            print(f"Array {i} shape: {arr.shape}")
        raise


# =========================
# Example usage
# =========================

if __name__ == "__main__":
    # Example: Chapel Hill, NC
    lat, lon = 35.9132, -79.0558
    
    print("=== Multi-source Risk Predictors Example ===")
    print(f"Location: Chapel Hill, NC ({lat:.4f}, {lon:.4f})")
    
    try:
        chips = fetch_multisource_chips(
            lat=lat, lon=lon, t0="2021-07-15",
            lookbacks=[1, 5, 10, 20],
            chip_size_m=2560, resolution_m=10,
            max_cloud=25, era_windows=[1, 3, 7, 14, 30],
            include_s2=True, include_dem=True, include_worldcover=True, 
            include_era5=True, include_smap=True
        )
        
        print(f"\n=== Data Summary ===")
        for key, data in chips.items():
            print(f"{key}: {data.shape} | dims: {data.dims}")
            if hasattr(data, 'coords') and 'band' in data.coords:
                print(f"  Bands: {list(data.coords['band'].values)}")
        
        # Create model tensor
        print("\n=== Creating Model Tensor ===")
        X, channels = to_model_tensor(chips)
        print(f"✓ Final tensor shape [T,C,H,W]: {X.shape}")
        print(f"✓ Channels ({len(channels)}): {channels}")
        
        # SPI/SPEI example shapes (if available)
        if "SPI" in chips:
            spi = chips["SPI"]
            print(f"✓ SPI shape [scale,time,y,x]: {spi.shape}")
            print(f"  Scales: {list(spi.coords['scale'].values)}")
        
        if "SPEI" in chips:
            spei = chips["SPEI"]
            print(f"✓ SPEI shape [scale,time,y,x]: {spei.shape}")
            print(f"  Scales: {list(spei.coords['scale'].values)}")
            
        print("\n=== Success! ===")
        
    except Exception as e:
        print(f"\n=== Error ===")
        print(f"Failed to fetch data: {e}")
        import traceback
        traceback.print_exc()
