"""Data: OISST download and crop, climatology and MHW thresholds, region split, synthetic clouds."""
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.ndimage import gaussian_filter

from utils import load_config

# The U-Net downsamples 3 times, so the 51 x 85 grid is padded to a multiple of 8.
PAD_H, PAD_W = 56, 88


# ---------------------------------------------------------------- download

def _year_url(cfg, year):
    d = cfg["data"]
    (la0, la1), (lo0, lo1) = d["lat"], d["lon"]
    sel = (f"[({year}-01-01T12:00:00Z):1:({year}-12-31T12:00:00Z)][(0.0)]"
           f"[({la0}):1:({la1})][({lo0}):1:({lo1})]")
    query = ",".join(v + sel for v in ("sst", "ice"))
    return d["erddap_url"] + "?" + urllib.parse.quote(query, safe=",:()")


def _download_year(cfg, year):
    path = os.path.join(cfg["data"]["raw_dir"], f"oisst_{year}.nc")
    if os.path.exists(path):
        return path
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(_year_url(cfg, year), path + ".part")
            os.replace(path + ".part", path)
            print(f"  downloaded {year}", flush=True)
            return path
        except Exception as e:  # ERDDAP occasionally times out under load
            print(f"  {year}: attempt {attempt + 1} failed ({e}); retrying", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"could not download {year}")


NCEI_DAY = ("https://www.ncei.noaa.gov/thredds/dodsC/OisstBase/NetCDF/V2.1/AVHRR/"
            "{d:%Y%m}/oisst-avhrr-v02r01.{d:%Y%m%d}.nc.ascii?")


def _parse_opendap_ascii(text, name, shape):
    body = text.split(f"{name}.{name}[", 1)[1].split("\n", 1)[1]
    vals = [float(v) for line in body.splitlines()[:shape[0]] for v in line.split(",")[1:]]
    a = np.array(vals, dtype=np.float32).reshape(shape)
    return np.where(a == -999, np.nan, a * 0.01)  # OISST stores int16 with scale_factor 0.01


def _fetch_day_ncei(day, lat, lon):
    """The ERDDAP aggregation is missing ~1200 days (mostly 1992-1998); NCEI's per-day files are
    complete. Uses the OPeNDAP ASCII response over plain HTTP so requests can run in parallel."""
    i0, i1 = (int(round((v + 89.875) / 0.25)) for v in (lat[0], lat[-1]))
    j0, j1 = (int(round((v - 0.125) / 0.25)) for v in (lon[0], lon[-1]))
    sel = f"[0:0][0:0][{i0}:{i1}][{j0}:{j1}]"
    url = NCEI_DAY.format(d=day) + urllib.parse.quote(f"sst{sel},ice{sel}", safe=",:")
    shape = (len(lat), len(lon))
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                text = r.read().decode()
            data = {v: (("time", "latitude", "longitude"), _parse_opendap_ascii(text, v, shape)[None])
                    for v in ("sst", "ice")}
            return xr.Dataset(data, coords=dict(time=[day], latitude=lat, longitude=lon))
        except Exception as e:
            print(f"  {day:%Y-%m-%d}: attempt {attempt + 1} failed ({e})", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not fetch {day}")


def download(cfg):
    os.makedirs(cfg["data"]["raw_dir"], exist_ok=True)
    y0, y1 = cfg["data"]["years"]
    with ThreadPoolExecutor(max_workers=4) as ex:
        paths = list(ex.map(lambda y: _download_year(cfg, y), range(y0, y1 + 1)))
    ds = xr.open_mfdataset(paths, combine="by_coords").isel(zlev=0, drop=True).load()
    ds = ds.assign_coords(time=ds.time.dt.floor("D")).sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
    ds = ds.drop_duplicates("time")

    full = pd.date_range(f"{y0}-01-01", f"{y1}-12-31", freq="D")
    missing = full.difference(pd.DatetimeIndex(ds.time.values))
    fill_path = os.path.join(cfg["data"]["raw_dir"], "ncei_missing_days.nc")
    if len(missing) and not os.path.exists(fill_path):
        print(f"filling {len(missing)} days missing from ERDDAP via NCEI THREDDS ...", flush=True)
        lat, lon = ds.latitude.values, ds.longitude.values
        with ThreadPoolExecutor(max_workers=8) as ex:
            days = list(ex.map(lambda d: _fetch_day_ncei(d, lat, lon), missing))
        fill = xr.concat(days, "time")
        fill = fill.assign_coords(time=fill.time.dt.floor("D"))
        fill.to_netcdf(fill_path)
    if len(missing):
        ds = xr.concat([ds, xr.open_dataset(fill_path)], "time").sortby("time")
    ds = ds.reindex(time=full)  # guarantees one entry per calendar day (NaN if still missing)
    print(f"days still missing: {int(ds.sst.isnull().all(('latitude', 'longitude')).sum())}")
    ds.to_netcdf(cfg["data"]["processed"])
    print(f"wrote {cfg['data']['processed']}: {dict(ds.sizes)}")


# ---------------------------------------------------------------- climatology & regions

def day_of_year_366(times):
    """Day of year on a 366-day calendar, so 1 March is always day 61 (as in Hobday et al. 2016)."""
    t = pd.DatetimeIndex(times)
    doy = t.dayofyear.values.copy()
    doy[(~t.is_leap_year) & (t.month > 2)] += 1
    return doy


def _pad(a, fill=np.nan):
    """Pad the trailing (lat, lon) dims to PAD_H x PAD_W."""
    h, w = a.shape[-2:]
    pad = [(0, 0)] * (a.ndim - 2) + [(0, PAD_H - h), (0, PAD_W - w)]
    return np.pad(a, pad, constant_values=fill)


def _circular_smooth(a, width=31):
    """Moving average along axis 0 of a (366, H, W) array, wrapping around the year."""
    k = width // 2
    ext = np.concatenate([a[-k:], a, a[:k]])
    c = np.nancumsum(ext, axis=0)
    n = np.cumsum(np.isfinite(ext), axis=0)
    c = np.concatenate([np.zeros_like(c[:1]), c])
    n = np.concatenate([np.zeros_like(n[:1]), n])
    s = (c[width:] - c[:-width]) / np.maximum(n[width:] - n[:-width], 1)
    s[(n[width:] - n[:-width]) == 0] = np.nan
    return s


def climatology(sst, doy, years, clim_years, pct):
    """Hobday-style daily climatology: pool +-5 days over the baseline years, then 31-day smoothing.
    Returns mean, std and percentile threshold, each (366, H, W)."""
    in_base = (years >= clim_years[0]) & (years <= clim_years[1])
    base, base_doy = sst[in_base], doy[in_base]
    out = np.full((3, 366) + sst.shape[1:], np.nan, dtype=np.float32)
    for d in range(1, 367):
        dist = np.abs(base_doy - d)
        dist = np.minimum(dist, 366 - dist)
        x = base[dist <= 5]
        with np.errstate(all="ignore"), np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            out[0, d - 1] = np.nanmean(x, 0)
            out[1, d - 1] = np.nanstd(x, 0)
            out[2, d - 1] = np.nanpercentile(x, pct, axis=0)
    return [_circular_smooth(o) for o in out]


def region_masks(lat, lon, cfg):
    s = cfg["split"]
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    gof = (LON >= s["gulf_of_finland_min_lon"]) & (LAT >= 59.0) & (LAT < 61.0)
    test = (LAT >= s["test_region_min_lat"]) & (LON >= s["test_region_min_lon"]) & ~gof
    buffer = ((LAT >= s["buffer_min_lat"]) | gof) & ~test
    train = ~(test | buffer)
    return {k: _pad(v, False) for k, v in dict(train=train, test=test, buffer=buffer).items()}


class Baltic:
    """Everything derived from the processed file, padded to the model grid."""

    def __init__(self, cfg):
        self.cfg = cfg
        ds = xr.open_dataset(cfg["data"]["processed"])
        self.times = pd.DatetimeIndex(ds.time.values)
        self.lat, self.lon = ds.latitude.values, ds.longitude.values
        self.H, self.W = len(self.lat), len(self.lon)
        sst = ds.sst.values.astype(np.float32)
        ice = ds.ice.fillna(0).values
        # Ice-covered pixels have no meaningful SST (OISST pins them near freezing): treat as invalid.
        self.valid = _pad(np.isfinite(sst) & (ice <= cfg["data"]["ice_threshold"]), False)
        self.sst = _pad(np.where(self.valid[:, :self.H, :self.W], sst, np.nan))
        self.doy = day_of_year_366(self.times)
        self.years = self.times.year.values
        self.ocean = _pad(np.isfinite(sst).any(0), False)

        cache = cfg["data"]["processed"].replace(".nc", "_clim.npz")
        if os.path.exists(cache):
            c = np.load(cache)
            self.clim_mean, self.clim_std, self.thresh = c["mean"], c["std"], c["thresh"]
        else:
            print("computing climatology and MHW threshold ...", flush=True)
            self.clim_mean, self.clim_std, self.thresh = climatology(
                self.sst, self.doy, self.years, cfg["data"]["clim_years"], cfg["eval"]["mhw_percentile"])
            np.savez(cache, mean=self.clim_mean, std=self.clim_std, thresh=self.thresh)
        # Anomaly relative to the day-of-year climatology is what the model reconstructs.
        self.anom = self.sst - self.clim_mean[self.doy - 1]
        # A few Bothnian pixels were ice-covered on that calendar day in every baseline year, so
        # they have no climatology; when they are ice-free later (2016-17) they cannot be scored.
        self.valid &= np.isfinite(self.anom)
        self.regions = region_masks(self.lat, self.lon, cfg)

    def day_indices(self, years):
        k = self.cfg["gaps"]["half_window"]
        idx = np.where((self.years >= years[0]) & (self.years <= years[1]))[0]
        return idx[(idx >= k) & (idx < len(self.times) - k)]


# ---------------------------------------------------------------- synthetic clouds

def cloud_mask(rng, ocean, frac, sigma):
    """Boolean (H, W) mask, True = hidden by cloud. Smoothed noise thresholded so that
    `frac` of the ocean is hidden, giving spatially correlated blobs."""
    noise = gaussian_filter(rng.standard_normal(ocean.shape), sigma, mode="wrap")
    q = np.quantile(noise[ocean], frac)
    return noise <= q


class GapDataset(torch.utils.data.Dataset):
    """One sample = a 7-day window centred on day t, every day partly hidden by clouds and
    some days missing entirely. The loss/metrics are computed on the hidden pixels of day t.

    region: ocean pixels the model is allowed to see as input; everything else is treated as land.
    seed:   None -> fresh random clouds every epoch (training); int -> fixed clouds (evaluation).
    """

    def __init__(self, bal, days, region, seed=None, frac_range=None):
        self.b, self.days, self.region, self.seed = bal, days, region, seed
        g = bal.cfg["gaps"]
        self.k, self.sigma, self.p_drop = g["half_window"], g["blob_sigma"], g["p_drop_day"]
        self.frac_range = frac_range or g["frac_range"]
        self.ocean = bal.ocean & region

    def __len__(self):
        return len(self.days)

    def __getitem__(self, i):
        t = self.days[i]
        rng = np.random.default_rng(None if self.seed is None else (self.seed, int(t)))
        b, k = self.b, self.k
        vals, obs = [], []
        for dt in range(-k, k + 1):
            hidden = cloud_mask(rng, self.ocean, rng.uniform(*self.frac_range), self.sigma)
            if dt == 0:
                target_hidden = hidden
            elif rng.random() < self.p_drop:
                hidden = np.ones_like(hidden)  # whole day missing: irregular sampling in time
            o = b.valid[t + dt] & self.ocean & ~hidden
            obs.append(o)
            vals.append(np.where(o, b.anom[t + dt], 0.0))
        doy = 2 * np.pi * b.doy[t] / 366
        cm = b.clim_mean[b.doy[t] - 1]
        static = [
            self.ocean.astype(np.float32),
            np.full((PAD_H, PAD_W), np.sin(doy)),
            np.full((PAD_H, PAD_W), np.cos(doy)),
            np.nan_to_num((cm - 8.0) / 6.0),
            np.nan_to_num(b.clim_std[b.doy[t] - 1]),
        ]
        x = np.stack(vals + [o.astype(np.float32) for o in obs] + static).astype(np.float32)
        y = np.nan_to_num(b.anom[t]).astype(np.float32)
        loss_mask = b.valid[t] & self.ocean & target_hidden
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(loss_mask), int(t)


def n_input_channels(cfg):
    return 2 * (2 * cfg["gaps"]["half_window"] + 1) + 5


if __name__ == "__main__":
    download(load_config())
