#!/usr/bin/env python3
"""Convert SWOT water-surface heights from EGM2008 to MN75 (Marea Neagra 1975) in a gridded NetCDF.

Writes a SEPARATE, self-contained copy of the input: every variable of the input is copied unchanged (heights in
EGM2008, classes, flags, sources, 2-D layers, coordinates, CRS), and each height variable H gets a twin H_mn75:

    H_mn75 = H_egm2008 - C(lat, lon),     C = zeta_RO - N_egm2008          (transform_simple.correction)

  zeta_RO   ANCPI ROMGEO quasigeoid, rom_grid3d_25.09.spg, romgeo's own B-spline (matches TransDatRO)
  N         EGM2008 undulation, us_nga_egm08_25.tif, bilinear; mean-tide by default (SWOT applies the geoid in the
            mean-tide system; --tide-free for tide-free). Verify against JPL D-109532 before relying on cm.
C is also written (C_mn75_to_egm_m, y x) so the conversion can be inspected. The input file is never modified.

What it does well (2026-10 rewrite, for danube_swot_final_60m_mlfill.nc; v1 is correct_swot_to_mn75_v1_2026-08-01.py):
  - Keeps ALL variables (v1 copied only the one height variable, so flags, classes and sources were lost).
  - Finds the height variables itself (3-D float, units m, name contains "height"), or --vars a,b.
  - C exactly per pixel (romgeo is compiled: ~4 M pixels in seconds), in row blocks.
  - ROMGEO B-spline is NaN within ~2 grid cells of its edge (the SWOT box reaches ~5 km east of it, offshore):
    query points are clamped to the safe interior (nearest-edge extrapolation, ~cm error); any C still NaN is
    filled from the nearest valid pixel. Both counts are written to the attributes.
  - Self-checks: C range, and C at random pixels against the independent implementation in mn75_egm2008.py
    (SpgSampler + GridSampler); stops if they differ by more than 1 mm.
  - Resumable and safe on a time limit: dates are written in order, progress is in the attribute
    n_dates_done, and with --max-seconds the run closes the file cleanly before the limit; rerun the same
    command to continue until "conversion_complete = yes".
  - Same chunking and compression as the input variables.

Usage:
    python correct_swot_to_mn75.py D:\\NeXus\\DeltaTwin\\swot\\danube_swot_final_60m_mlfill.nc
        -> danube_swot_final_60m_mlfill_mn75.nc next to it (or --out)
    python correct_swot_to_mn75.py D:\\DeltaSwot\\SWOT_PIXC_clean_height_water_only_100m_strict.nc
        (old 100 m product: still works; gives height_final_daily_merged_mn75 as before)
Options: --out, --vars, --epsg (default: from spatial_ref, else 32635), --tide-free, --suffix (_mn75),
         --max-seconds, --no-check
"""
import argparse
import os
import pickle
import sys
import time

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import netCDF4
import numpy as np
from pyproj import CRS, Transformer

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
from transform_simple import EGM_PATH, SPG_PATH, correction  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def spg_safe_bounds(margin_cells=2.5):
    """Lon/lat box inside which the ROMGEO B-spline is NaN-free (half-integer margin on purpose: clamping exactly
    onto a grid node makes the edge window of _doBSInterpolation overrun)."""
    with open(SPG_PATH, "rb") as f:
        gh = pickle.load(f)["grids"]["geoid_heights"]
    md = gh["metadata"]
    grid = np.asarray(gh["grid"], float)
    if grid.ndim == 3:
        grid = grid[0]
    nphi, nla = grid.shape
    minla, minphi = float(md["minla"]), float(md["minphi"])
    stepla, stepphi = float(md["stepla"]), float(md["stepphi"])
    return (minla + margin_cells * stepla, minla + (nla - 1 - margin_cells) * stepla,
            minphi + margin_cells * stepphi, minphi + (nphi - 1 - margin_cells) * stepphi)


def epsg_of(src, default=32635):
    if "spatial_ref" in src.variables:
        wkt = getattr(src.variables["spatial_ref"], "crs_wkt", None) or getattr(src.variables["spatial_ref"], "spatial_ref", None)
        if wkt:
            e = CRS.from_wkt(wkt).to_epsg()
            if e:
                return e
    return default


def compute_C(xs, ys, epsg, tide_mean, block=256):
    """C(y, x) per pixel, exact. Returns C, number of clamped pixels, number of NaN pixels filled."""
    t = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lo0, lo1, la0, la1 = spg_safe_bounds()
    C = np.empty((len(ys), len(xs)), np.float64)
    n_clamp = 0
    for j0 in range(0, len(ys), block):
        X, Y = np.meshgrid(xs, ys[j0:j0 + block])
        lon, lat = t.transform(X.ravel(), Y.ravel())
        lon, lat = np.asarray(lon), np.asarray(lat)
        out = (lon < lo0) | (lon > lo1) | (lat < la0) | (lat > la1)
        n_clamp += int(out.sum())
        C[j0:j0 + block] = correction(np.clip(lat, la0, la1), np.clip(lon, lo0, lo1), tide_mean=tide_mean).reshape(X.shape)
    bad = ~np.isfinite(C)
    n_fill = int(bad.sum())
    if n_fill:
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        C = C[tuple(idx)]
    return C, n_clamp, n_fill


def check_C(C, xs, ys, epsg, tide_mean, n=400, seed=0):
    """C at random pixels against mn75_egm2008.py (independent samplers). Returns max |diff| (m)."""
    import mn75_egm2008 as ref
    lo0, lo1, la0, la1 = spg_safe_bounds()
    rng = np.random.default_rng(seed)
    jj, ii = rng.integers(0, len(ys), n), rng.integers(0, len(xs), n)
    lon, lat = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform(xs[ii], ys[jj])
    lon, lat = np.clip(lon, lo0, lo1), np.clip(lat, la0, la1)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        z = ref.SpgSampler(SPG_PATH).sample(lat, lon)
    nn = ref.GridSampler(EGM_PATH).sample(lat, lon)
    if tide_mean:
        nn = nn + ref.tide_free_to_mean_dN(lat)
    d = np.abs((z - nn) - C[jj, ii])
    return float(np.nanmax(d)), int(np.isfinite(d).sum())


def var_kwargs(sv):
    f = sv.filters() or {}
    kw = dict(zlib=bool(f.get("zlib")), complevel=int(f.get("complevel") or 0), shuffle=bool(f.get("shuffle")))
    ch = sv.chunking()
    if isinstance(ch, list):
        kw["chunksizes"] = tuple(ch)
    if "_FillValue" in sv.ncattrs():
        kw["fill_value"] = sv.getncattr("_FillValue")
    return kw


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("--out")
    p.add_argument("--vars", default=None, help="height variables to convert (default: detected)")
    p.add_argument("--var", default=None, help=argparse.SUPPRESS)          # v1 option, same as --vars
    p.add_argument("--epsg", type=int, default=None)
    p.add_argument("--tide-free", action="store_true")
    p.add_argument("--suffix", default="_mn75")
    p.add_argument("--max-seconds", type=float, default=0, help="stop cleanly after this long (rerun to resume)")
    p.add_argument("--no-check", action="store_true")
    a = p.parse_args()
    out_path = a.out or a.input[:-3] + "_mn75.nc"
    tide_mean = not a.tide_free
    src = netCDF4.Dataset(a.input)
    epsg = a.epsg or epsg_of(src)

    names = a.vars or a.var
    if names:
        hv = [s.strip() for s in names.split(",")]
        tdim = src.variables[hv[0]].dimensions[0]
    else:
        cand = [v for v in src.variables.values() if v.ndim == 3 and v.dtype.kind == "f"
                and getattr(v, "units", "") == "m" and "height" in v.name.lower()]
        if not cand:
            raise SystemExit("no height variable found; use --vars")
        tdim = cand[0].dimensions[0]
        hv = [v.name for v in cand if v.dimensions[0] == tdim]
    if not hv:
        raise SystemExit("no height variable found; use --vars")
    v0 = src.variables[hv[0]]
    _, ydim, xdim = v0.dimensions
    nt = len(src.dimensions[tdim])
    xs = src.variables[xdim][:].astype(float)
    ys = src.variables[ydim][:].astype(float)
    per_date = [n for n, v in src.variables.items() if v.ndim >= 2 and v.dimensions[0] == tdim]
    log(f"{a.input}: {nt} {tdim} x {len(ys)} x {len(xs)}, EPSG:{epsg}; heights {hv}; "
        f"{len(per_date)} per-date variables; tide {'mean' if tide_mean else 'free'}")

    new = not os.path.exists(out_path)
    if new:
        C, n_clamp, n_fill = compute_C(xs, ys, epsg, tide_mean)
        log(f"C computed in {time.time() - T0:.0f} s: {n_clamp} pixels clamped to the ROMGEO safe box, "
            f"{n_fill} filled from the nearest pixel")
        log(f"C (H_egm2008 = H_mn75 + C): mean {C.mean():+.4f} m, range [{C.min():+.4f}, {C.max():+.4f}]")
        if not a.no_check:
            dmax, nchk = check_C(C, xs, ys, epsg, tide_mean)
            log(f"check against mn75_egm2008.py at {nchk} random pixels: max |diff| {dmax * 1000:.3f} mm")
            if not dmax < 1e-3:
                raise SystemExit("C does not match the reference implementation")

        dst = netCDF4.Dataset(out_path, "w")
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
        for dname, dd in src.dimensions.items():
            dst.createDimension(dname, None if dd.isunlimited() else len(dd))
        for n, sv in src.variables.items():
            ov = dst.createVariable(n, sv.dtype, sv.dimensions, **var_kwargs(sv))
            ov.setncatts({k: sv.getncattr(k) for k in sv.ncattrs() if k != "_FillValue"})
            if n in hv:
                ov.vertical_datum = "EGM2008 (as in the input)"
            if n not in per_date:
                ov[:] = sv[:]
        for n in hv:
            sv = src.variables[n]
            ov = dst.createVariable(n + a.suffix, sv.dtype, sv.dimensions, **var_kwargs(sv))
            ov.setncatts({k: sv.getncattr(k) for k in sv.ncattrs() if k != "_FillValue"})
            ln = getattr(sv, "long_name", n)
            ov.long_name = ln.replace("EGM2008", "MN75") if "EGM2008" in ln else ln + " [MN75]"
            ov.vertical_datum = "MN75 (Marea Neagra 1975)"
        cv = dst.createVariable("C_mn75_to_egm_m", "f4", (ydim, xdim), zlib=True, complevel=4)
        cv.long_name = "correction C such that H_egm2008 = H_mn75 + C"
        cv.units = "m"
        if "spatial_ref" in src.variables:
            cv.grid_mapping = "spatial_ref"
        cv[:] = C.astype("f4")
        dst.vertical_datum_note = (
            f"{', '.join(n + a.suffix for n in hv)}: MN75 (Marea Neagra 1975) from EGM2008 "
            f"({'mean-tide' if tide_mean else 'tide-free'}) as H_mn75 = H - C, C = zeta_RO - N_egm2008; zeta_RO: ANCPI "
            "rom_grid3d_25.09.spg (romgeo B-spline); N: us_nga_egm08_25.tif (bilinear). The other variables are copied "
            "unchanged (heights in EGM2008).")
        dst.mn75_pixels_clamped = np.int64(n_clamp)
        dst.mn75_pixels_nan_filled = np.int64(n_fill)
        dst.history = (getattr(src, "history", "") + f" | {time.strftime('%Y-%m-%d')} correct_swot_to_mn75.py: "
                       "copy + MN75 heights").strip(" |")
        dst.n_dates_done = np.int32(0)
        dst.conversion_complete = "no"
        dst.close()
        log(f"created {out_path}")

    dst = netCDF4.Dataset(out_path, "a")
    C = dst.variables["C_mn75_to_egm_m"][:].astype(np.float32)
    done = int(dst.n_dates_done)
    if done:
        log(f"resume at {done}/{nt}")
    for d in range(done, nt):
        if a.max_seconds and time.time() - T0 > a.max_seconds:
            break
        for n in per_date:
            dst.variables[n][d] = src.variables[n][d]
        for n in hv:
            h = np.ma.filled(src.variables[n][d], np.nan).astype(np.float32)
            dst.variables[n + a.suffix][d] = h - C
        dst.n_dates_done = np.int32(d + 1)
        if (d + 1) % 5 == 0:
            dst.sync()
            log(f"{d + 1}/{nt}")
    done = int(dst.n_dates_done)
    if done == nt:
        dst.conversion_complete = "yes"
    dst.close()
    src.close()
    log(f"{out_path}: {done}/{nt} dates" + (" - done" if done == nt else " - rerun to continue"))


if __name__ == "__main__":
    main()
