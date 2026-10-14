#!/usr/bin/env python3
"""Add MN75 daily heights to the cleaned SWOT WSE NetCDF (separate output file).

Applies the per-pixel correction C = zeta_RO - N_egm2008(mean-tide) from
transform_simple.py (grids auto-found next to that script):

    H_mn75 = H_egm2008 - C(lat, lon)

C is computed once per (y, x) pixel — it is date-independent — then applied
to every date slice. The output is a SEPARATE, self-contained NetCDF holding:

    height_final_daily_merged        original EGM2008 heights (unchanged copy)
    height_final_daily_merged_mn75   NEW: heights in MN75 (Marea Neagra 1975)
    C_mn75_to_egm_m                  the correction surface, for inspection
    date / y / x / spatial_ref       coords + CRS as in the input

so the validation pipeline can point at it and pick either datum by variable
name. The input file is never modified.

Implementation notes (sandbox-hardened):
- Streams date-by-date via netCDF4 (never loads a full (date,y,x) cube).
- RESUMABLE: progress is kept in the output attr `n_dates_done`; rerun the
  same command after a timeout and it continues where it stopped.
- C is cached next to the output as <out>.C.npy (delete to force recompute).
- The ROMGEO B-spline returns NaN within ~2 grid cells of the grid edge; the
  SWOT domain pokes ~5 km past the eastern edge (offshore Black Sea). Query
  points are clamped to the safe interior — nearest-edge extrapolation, error
  ~cm there — and the affected pixel count is recorded in the attrs.

Usage (defaults match the DeltaTwin validation conventions):
    python correct_swot_to_mn75.py D:\\DeltaSwot\\SWOT_PIXC_clean_height_water_only_100m_strict.nc
    # -> writes ..._mn75.nc next to it
Options:
    --out OUT.nc      output path
    --var NAME        height variable (default height_final_daily_merged)
    --epsg CODE       CRS of x/y coords (default 32635)
    --tide-free       use tide-free EGM2008 instead of mean-tide
                      (default mean-tide: SWOT applies the geoid mean-tide;
                      verify against JPL D-109532 before production use)
"""
import argparse
import os
import pickle
import sys

import numpy as np
import netCDF4
from pyproj import Transformer

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
from transform_simple import correction, SPG_PATH


def spg_safe_bounds(margin_cells=2.5):
    """Lon/lat box inside which the ROMGEO B-spline is NaN-free.

    NB: a half-integer margin is deliberate — clamping exactly ONTO a grid
    node (integer margin) makes _doBSInterpolation's edge window overrun and
    raise for a subset of query points; between nodes it is reliable."""
    with open(SPG_PATH, "rb") as f:
        md = pickle.load(f)["grids"]["geoid_heights"]["metadata"]
    nla = int(md["ncols"]) if "ncols" in md else None
    # metadata keys vary; derive extents from the grid shape if needed
    with open(SPG_PATH, "rb") as f:
        gh = pickle.load(f)["grids"]["geoid_heights"]
    grid = np.asarray(gh["grid"], float)
    if grid.ndim == 3:
        grid = grid[0]
    nphi, nla = grid.shape
    minla, minphi = float(md["minla"]), float(md["minphi"])
    stepla, stepphi = float(md["stepla"]), float(md["stepphi"])
    return (minla + margin_cells * stepla,
            minla + (nla - 1 - margin_cells) * stepla,
            minphi + margin_cells * stepphi,
            minphi + (nphi - 1 - margin_cells) * stepphi)


def compute_C(xs, ys, epsg, tide_mean, cache_path):
    """Per-pixel correction C(y, x); cached to cache_path."""
    if os.path.exists(cache_path):
        C = np.load(cache_path)
        print(f"[C  ] loaded cache {cache_path}")
        return C
    X, Y = np.meshgrid(xs, ys)
    t = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon, lat = t.transform(X.ravel(), Y.ravel())
    lon = np.asarray(lon); lat = np.asarray(lat)
    lo_min, lo_max, la_min, la_max = spg_safe_bounds()
    clamped = ((lon < lo_min) | (lon > lo_max) |
               (lat < la_min) | (lat > la_max))
    n_clamped = int(clamped.sum())
    if n_clamped:
        print(f"[C  ] {n_clamped} pixels outside ROMGEO safe box "
              f"(lon {lo_min:.3f}..{lo_max:.3f}, lat {la_min:.3f}..{la_max:.3f})"
              " -> nearest-edge extrapolation")
    lon_q = np.clip(lon, lo_min, lo_max)
    lat_q = np.clip(lat, la_min, la_max)
    C = correction(lat_q, lon_q, tide_mean=tide_mean).reshape(Y.shape)
    np.save(cache_path, C)
    return C


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("--out")
    p.add_argument("--var", default="height_final_daily_merged")
    p.add_argument("--epsg", type=int, default=32635)
    p.add_argument("--tide-free", action="store_true")
    a = p.parse_args()
    out_path = a.out or a.input.replace(".nc", "_mn75.nc")
    var_mn75 = a.var + "_mn75"
    tide_mean = not a.tide_free

    src = netCDF4.Dataset(a.input)
    if a.var not in src.variables:
        raise SystemExit(f"variable '{a.var}' not in file")
    v = src.variables[a.var]
    ddim, ydim, xdim = v.dimensions
    nd, ny, nx = v.shape
    xs = src.variables[xdim][:].astype(float)
    ys = src.variables[ydim][:].astype(float)
    print(f"[in ] {a.input}: {a.var} {v.shape}, EPSG:{a.epsg}")

    C = compute_C(xs, ys, a.epsg, tide_mean, out_path + ".C.npy")
    print(f"[C  ] tide={'mean' if tide_mean else 'free'}  "
          f"mean {np.nanmean(C):+.3f} m  "
          f"range [{np.nanmin(C):+.3f}, {np.nanmax(C):+.3f}]")

    # --- create or resume the output file ------------------------------------
    new = not os.path.exists(out_path)
    dst = netCDF4.Dataset(out_path, "w" if new else "a")
    if new:
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
        dst.vertical_datum_note = (
            f"{var_mn75}: MN75 (Marea Neagra 1975), converted from EGM2008 "
            f"({'mean-tide' if tide_mean else 'tide-free'}) via "
            "H_mn75 = H - (zeta_RO - N_egm2008); zeta_RO: ANCPI "
            "rom_grid3d_25.09.spg (romgeo B-spline); N: us_nga_egm08_25.tif; "
            "pixels east/off the ROMGEO grid use nearest-edge extrapolated C.")
        dst.history = (getattr(src, "history", "") +
                       " | correct_swot_to_mn75.py: added MN75 daily heights"
                       ).strip(" |")
        for dim, n in ((ddim, nd), (ydim, ny), (xdim, nx)):
            dst.createDimension(dim, n)
        # coords + grid mapping, copied verbatim
        copy_vars = [ddim, ydim, xdim]
        if "spatial_ref" in src.variables:
            copy_vars.append("spatial_ref")
        for name in copy_vars:
            sv = src.variables[name]
            ov = dst.createVariable(name, sv.dtype, sv.dimensions)
            ov.setncatts({k: sv.getncattr(k) for k in sv.ncattrs()})
            if sv.shape:
                ov[:] = sv[:]
        kw = dict(zlib=True, complevel=4, chunksizes=(1, ny, nx),
                  fill_value=np.float32(np.nan))
        for name, ln in ((a.var, "EGM2008 (as input)"), (var_mn75, "MN75")):
            ov = dst.createVariable(name, "f4", (ddim, ydim, xdim), **kw)
            ov.setncatts({k: v.getncattr(k) for k in v.ncattrs()
                          if k != "_FillValue"})
            ov.vertical_datum = ln
        cv = dst.createVariable("C_mn75_to_egm_m", "f4", (ydim, xdim),
                                zlib=True, complevel=4,
                                fill_value=np.float32(np.nan))
        cv.long_name = "correction C such that H_egm2008 = H_mn75 + C"
        cv.units = "m"
        cv[:] = C.astype("f4")
        dst.n_dates_done = np.int32(0)
        dst.sync()
    done = int(getattr(dst, "n_dates_done", 0))
    if done:
        print(f"[resume] {done}/{nd} dates already written")

    ov_e = dst.variables[a.var]
    ov_m = dst.variables[var_mn75]
    Cf = C.astype("f4")
    for d in range(done, nd):
        sl = np.ma.filled(v[d], np.nan).astype("f4")
        ov_e[d] = sl
        ov_m[d] = sl - Cf
        dst.n_dates_done = np.int32(d + 1)
        if (d + 1) % 10 == 0 or d == nd - 1:
            dst.sync()
            print(f"[{d + 1}/{nd}]", flush=True)
    dst.conversion_complete = "yes"
    dst.close()
    src.close()
    print(f"[out] {out_path}")


if __name__ == "__main__":
    main()
