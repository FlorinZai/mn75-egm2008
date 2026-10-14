#!/usr/bin/env python3
"""Simple EGM2008 <-> MN75 (Marea Neagra 1975) height transform.

Chain (via ETRS89 ellipsoid heights):
    H_egm2008 = H_mn75 + C,   C = zeta_RO - N_egm2008

Uses the two grids next to this script (override with env vars or args):
    rom_grid3d_25.09.spg   ANCPI ROMGEO quasigeoid (zeta_RO), sampled with
                           romgeo's own B-spline interpolator -> matches TransDatRO
    us_nga_egm08_25.tif    EGM2008 undulation N (tide-free, as distributed)

Python API:
    from transform_simple import egm_to_mn75, mn75_to_egm
    H = egm_to_mn75(44.895, 29.60, 2.34)        # scalars or numpy arrays

CLI:
    python transform_simple.py egm-to-mn75 44.895 29.60 2.34
    python transform_simple.py mn75-to-egm --csv in.csv out.csv   # cols lat,lon,h

Note: N is used tide-free (self-consistent). For SWOT mean-tide work add
--tide-mean, which applies dN = 1.30*(0.099 - 0.296 sin^2 phi) to N.
"""
import os
import sys
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
SPG_PATH = os.environ.get("RO_SPG", os.path.join(_DIR, "rom_grid3d_25.09.spg"))
EGM_PATH = os.environ.get("EGM_TIF", os.path.join(_DIR, "us_nga_egm08_25.tif"))

_cache = {}


def _zeta_ro(lat, lon):
    """Romanian quasigeoid anomaly (m) from the ANCPI SPG grid."""
    if "spg" not in _cache:
        import pickle
        from romgeo.transformations import _doBSInterpolation
        with open(SPG_PATH, "rb") as f:
            gh = pickle.load(f)["grids"]["geoid_heights"]
        md = gh["metadata"]
        grid = np.asarray(gh["grid"], float)[0]
        _cache["spg"] = (md, grid, _doBSInterpolation)
    md, grid, interp = _cache["spg"]
    lat = np.atleast_1d(np.asarray(lat, float))
    lon = np.atleast_1d(np.asarray(lon, float))
    out = np.full(lat.shape, np.nan)
    for i in np.ndindex(lat.shape):
        try:
            out[i] = interp(lon[i], lat[i], md["minla"], md["minphi"],
                            md["stepla"], md["stepphi"], grid)
        except Exception:
            pass
    return out


def _n_egm2008(lat, lon):
    """EGM2008 undulation N (m), bilinear from the NGA 2.5' GeoTIFF (0-360 lon)."""
    if "egm" not in _cache:
        import rasterio
        ds = rasterio.open(EGM_PATH)
        _cache["egm"] = (ds.read(1).astype(float), ds.transform, ds.shape)
        ds.close()
    band, T, (H, W) = _cache["egm"]
    lat = np.asarray(lat, float)
    lon = np.where(np.asarray(lon, float) < 0, np.asarray(lon, float) + 360, lon)
    col, row = (~T) * (lon, lat)
    col, row = np.asarray(col) - 0.5, np.asarray(row) - 0.5
    c0, r0 = np.floor(col).astype(int), np.floor(row).astype(int)
    fc, fr = col - c0, row - r0
    c0, r0 = np.clip(c0, 0, W - 2), np.clip(r0, 0, H - 2)
    return (band[r0, c0] * (1 - fc) * (1 - fr) + band[r0, c0 + 1] * fc * (1 - fr)
            + band[r0 + 1, c0] * (1 - fc) * fr + band[r0 + 1, c0 + 1] * fc * fr)


def correction(lat, lon, tide_mean=False):
    """C (m) such that H_egm2008 = H_mn75 + C."""
    n = _n_egm2008(lat, lon)
    if tide_mean:
        s2 = np.sin(np.radians(np.asarray(lat, float))) ** 2
        n = n + 1.30 * (0.099 - 0.296 * s2)
    return _zeta_ro(lat, lon) - n


def mn75_to_egm(lat, lon, h, tide_mean=False):
    return np.asarray(h, float) + correction(lat, lon, tide_mean)


def egm_to_mn75(lat, lon, h, tide_mean=False):
    return np.asarray(h, float) - correction(lat, lon, tide_mean)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("direction", choices=["mn75-to-egm", "egm-to-mn75"])
    p.add_argument("args", nargs="*", help="lat lon h   (single point)")
    p.add_argument("--csv", nargs=2, metavar=("IN", "OUT"),
                   help="CSV with columns lat,lon,h")
    p.add_argument("--tide-mean", action="store_true")
    a = p.parse_args()
    fn = mn75_to_egm if a.direction == "mn75-to-egm" else egm_to_mn75
    if a.csv:
        import pandas as pd
        df = pd.read_csv(a.csv[0])
        out = fn(df["lat"].values, df["lon"].values, df["h"].values, a.tide_mean)
        df["h_out"] = np.round(out, 4)
        df.to_csv(a.csv[1], index=False)
        print(f"{len(df)} points -> {a.csv[1]}  "
              f"(mean C = {np.nanmean(out - df['h'].values):+.3f} m)")
    elif len(a.args) == 3:
        lat, lon, h = map(float, a.args)
        out = fn(lat, lon, h, a.tide_mean)
        C = correction(lat, lon, a.tide_mean)
        print(f"h_in={h:.4f}  zeta_RO={_zeta_ro(lat, lon)[0]:.4f}  "
              f"N_egm={float(np.atleast_1d(_n_egm2008(lat, lon))[0]):.4f}  "
              f"C={float(np.atleast_1d(C)[0]):+.4f}  ->  h_out={float(np.atleast_1d(out)[0]):.4f}")
    else:
        p.error("give 'lat lon h' or --csv IN OUT")


if __name__ == "__main__":
    main()
