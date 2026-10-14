#!/usr/bin/env python3
"""
mn75_egm2008.py — Convert heights between MN75 (Marea Neagră 1975) and EGM2008,
via ellipsoid heights (ETRS89/GRS80 ~ WGS84 at mm level for this purpose).

Chain:
    h_ell      = H_MN75    + zeta_RO(phi, lam)          (RO quasigeoid anomaly)
    H_EGM2008  = h_ell     - N_EGM2008(phi, lam)        (EGM2008 undulation)
=>  H_EGM2008  = H_MN75    + [zeta_RO - N_EGM2008]      (single correction surface C)

zeta_RO sources (pick one):
  --zeta-grid  GeoTIFF of quasigeoid anomaly zeta (m), EPSG:4326 axes
  --zeta-const single value (m) for small areas (e.g. delta-wide constant)
  --benchmarks CSV with columns lat,lon,h_ell,H_mn75  -> fits constant or plane
               (zeta = h_ell - H_mn75 at each benchmark)

N_EGM2008 sources (pick one):
  --egm-grid   GeoTIFF of EGM2008 undulation (m), e.g. us_nga_egm08_25.tif
               from https://cdn.proj.org/us_nga_egm08_25.tif  (recommended)
  --egm-pyproj use pyproj EPSG:4979 -> EPSG:3855 (requires PROJ grids installed;
               the script REFUSES silent no-op transforms)
  --egm-const  single value (m) (testing / tiny areas only)

Tide system:
  EGM2008 grids are distributed TIDE-FREE; SWOT applies the geoid in MEAN-TIDE.
  --tide mean  converts the geoid undulation tide-free -> mean-tide via
               dN = (1+k2) * (0.099 - 0.296 sin^2 phi) [m], k2 = 0.30 (Ekman 1989).
               At 45 N this is about -0.064 m on N, i.e. +0.064 m on H_EGM2008.
               VERIFY the sign/formula against the SWOT User Handbook
               (JPL D-109532) before production use; conventions differ between
               documents. Default: none (stay tide-free, self-consistent).

I/O:
  Points: CSV with columns lat,lon,h (names configurable). Adds converted column.
  Raster: GeoTIFF in any projected CRS; band 1 heights; converted in place-of-copy.

Examples:
  # CSV of MN75 gauge zeros -> EGM2008, using grids
  python mn75_egm2008.py points in.csv out.csv --direction mn75-to-egm \
      --zeta-grid ro_geoid.tif --egm-grid us_nga_egm08_25.tif --tide mean

  # Calibrate constant C from RTK benchmarks, then convert a DEM back to MN75
  python mn75_egm2008.py raster dem_egm08.tif dem_mn75.tif \
      --direction egm-to-mn75 --benchmarks bench.csv --egm-grid us_nga_egm08_25.tif

  # Just report the correction surface at points (no conversion)
  python mn75_egm2008.py points in.csv out.csv --direction report \
      --zeta-const 30.85 --egm-grid us_nga_egm08_25.tif
"""

import argparse
import sys
import numpy as np

K2_LOVE = 0.30  # degree-2 potential Love number used in tide-free geoid definition


# ----------------------------------------------------------------------------- 
# Geoid / quasigeoid samplers
# -----------------------------------------------------------------------------

class GridSampler:
    """Bilinear sampler for a lat/lon GeoTIFF (EPSG:4326 axes)."""

    def __init__(self, path):
        import rasterio
        self.ds = rasterio.open(path)
        if self.ds.crs is not None and self.ds.crs.to_epsg() not in (4326, 4979, None):
            # Reprojection of the *query points* is handled by caller supplying
            # lat/lon; the grid itself must be geographic.
            raise SystemExit(
                f"{path}: grid CRS is {self.ds.crs}; supply a geographic "
                "(EPSG:4326) geoid grid."
            )
        self.band = self.ds.read(1).astype(float)
        nod = self.ds.nodata
        if nod is not None:
            self.band[self.band == nod] = np.nan
        self.transform = self.ds.transform

    def sample(self, lat, lon):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        # Handle 0-360 grids (the NGA EGM08 grid is 0..360)
        lon_q = lon.copy()
        left = self.transform.c
        right = left + self.transform.a * self.ds.width
        if min(left, right) >= 0.0 and max(left, right) > 180.0:
            lon_q = np.where(lon_q < 0.0, lon_q + 360.0, lon_q)
        # fractional pixel coords (col, row), pixel-center convention
        inv = ~self.transform
        col, row = inv * (lon_q, lat)
        col = col - 0.5
        row = row - 0.5
        c0 = np.floor(col).astype(int)
        r0 = np.floor(row).astype(int)
        fc = col - c0
        fr = row - r0
        H, W = self.band.shape
        c0c = np.clip(c0, 0, W - 2)
        r0c = np.clip(r0, 0, H - 2)
        z00 = self.band[r0c, c0c]
        z01 = self.band[r0c, c0c + 1]
        z10 = self.band[r0c + 1, c0c]
        z11 = self.band[r0c + 1, c0c + 1]
        out = (z00 * (1 - fc) * (1 - fr) + z01 * fc * (1 - fr)
               + z10 * (1 - fc) * fr + z11 * fc * fr)
        oob = (c0 < 0) | (c0 > W - 2) | (r0 < 0) | (r0 > H - 2)
        out = np.where(oob, np.nan, out)
        return out


class SpgSampler:
    """zeta_RO from an official ANCPI ROMGEO grid file (rom_grid3d_*.spg,
    from the 'romgeo-grid' download on rompos.ro). Uses romgeo's own
    B-spline interpolation so results match TransDatRO exactly.
    Note: .spg is a Python pickle — only load files obtained from ANCPI."""

    def __init__(self, path):
        import pickle
        try:
            from romgeo.transformations import _doBSInterpolation
        except ImportError:
            raise SystemExit("--zeta-spg requires the romgeo package: "
                             "pip install romgeo")
        with open(path, "rb") as f:
            data = pickle.load(f)
        gh = data["grids"]["geoid_heights"]
        md = gh["metadata"]
        grid = np.asarray(gh["grid"], dtype=float)
        if grid.ndim == 3:
            grid = grid[0]
        self.grid = grid
        self.minla = float(md["minla"]); self.minphi = float(md["minphi"])
        self.stepla = float(md["stepla"]); self.stepphi = float(md["stepphi"])
        self.interp = _doBSInterpolation
        ver = data.get("params", {}).get("version", "?")
        print(f"[spg] {path}: version {ver}, geoid grid '{gh.get('name','?')}' "
              f"{gh.get('source','?')} -> {gh.get('target','?')}, "
              f"shape {grid.shape}, "
              f"lon {self.minla:.2f}+{self.stepla}*{grid.shape[-1]}, "
              f"lat {self.minphi:.2f}+{self.stepphi}*{grid.shape[-2]}")

    def sample(self, lat, lon):
        lat = np.atleast_1d(np.asarray(lat, dtype=float))
        lon = np.atleast_1d(np.asarray(lon, dtype=float))
        out = np.empty(lat.shape, dtype=float)
        it = np.nditer([lat, lon], flags=["multi_index"])
        for la, lo in it:
            try:
                out[it.multi_index] = self.interp(
                    float(lo), float(la), self.minla, self.minphi,
                    self.stepla, self.stepphi, self.grid)
            except Exception:
                out[it.multi_index] = np.nan
        return out


class ConstSampler:
    def __init__(self, value):
        self.value = float(value)

    def sample(self, lat, lon):
        return np.full(np.broadcast(np.asarray(lat), np.asarray(lon)).shape,
                       self.value, dtype=float)


class PlaneSampler:
    """zeta = a + b*(lat-lat0) + c*(lon-lon0), fitted from benchmarks."""

    def __init__(self, a, b, c, lat0, lon0):
        self.a, self.b, self.c, self.lat0, self.lon0 = a, b, c, lat0, lon0

    def sample(self, lat, lon):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        return self.a + self.b * (lat - self.lat0) + self.c * (lon - self.lon0)


class PyprojEGM:
    """EGM2008 via PROJ (EPSG:4979 -> EPSG:3855). Refuses silent no-ops."""

    def __init__(self):
        from pyproj import Transformer
        self.t = Transformer.from_crs("EPSG:4979", "EPSG:3855", always_xy=True)
        # sanity probe: over open ocean near the delta N ~ +29..+35 m; if the
        # vertical grid is missing PROJ returns the input height unchanged.
        probe = self.t.transform(29.6, 44.9, 100.0)[2]
        if abs(probe - 100.0) < 1e-9:
            raise SystemExit(
                "pyproj/PROJ has no EGM2008 grid installed (transform was a "
                "no-op). Install grids (`projsync --file us_nga_egm08_25`) or "
                "use --egm-grid with us_nga_egm08_25.tif from cdn.proj.org."
            )

    def sample(self, lat, lon):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        zeros = np.zeros_like(lat, dtype=float)
        _, _, h_out = self.t.transform(lon, lat, zeros)
        # h_out = orthometric height of the ellipsoidal-0 point => N = -h_out
        return -np.asarray(h_out, dtype=float)


# -----------------------------------------------------------------------------
# Benchmarks -> zeta model
# -----------------------------------------------------------------------------

def fit_zeta_from_benchmarks(path, mode="auto"):
    import pandas as pd
    df = pd.read_csv(path)
    need = {"lat", "lon", "h_ell", "H_mn75"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"benchmarks CSV missing columns: {sorted(missing)}")
    zeta = (df["h_ell"] - df["H_mn75"]).to_numpy(dtype=float)
    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)
    n = len(df)
    const = float(np.mean(zeta))
    resid_c = zeta - const
    rms_c = float(np.sqrt(np.mean(resid_c ** 2)))
    print(f"[benchmarks] n={n}  zeta mean={const:+.4f} m  "
          f"rms about constant={rms_c*100:.1f} cm  "
          f"spread={ (zeta.max()-zeta.min())*100:.1f} cm")
    if mode == "const" or (mode == "auto" and (n < 4 or rms_c < 0.01)):
        print(f"[benchmarks] using CONSTANT zeta = {const:+.4f} m")
        return ConstSampler(const)
    lat0, lon0 = float(lat.mean()), float(lon.mean())
    A = np.column_stack([np.ones(n), lat - lat0, lon - lon0])
    coef, *_ = np.linalg.lstsq(A, zeta, rcond=None)
    resid_p = zeta - A @ coef
    rms_p = float(np.sqrt(np.mean(resid_p ** 2)))
    print(f"[benchmarks] plane fit: a={coef[0]:+.4f} m, "
          f"dz/dlat={coef[1]*100:+.2f} cm/deg, dz/dlon={coef[2]*100:+.2f} cm/deg, "
          f"rms={rms_p*100:.1f} cm -> using PLANE")
    return PlaneSampler(coef[0], coef[1], coef[2], lat0, lon0)


# -----------------------------------------------------------------------------
# Tide system
# -----------------------------------------------------------------------------

def tide_free_to_mean_dN(lat):
    """dN (m) to ADD to a tide-free geoid undulation to obtain mean-tide.
    Ekman (1989): N_mean = N_zero + (0.099 - 0.296 sin^2 phi);
                  N_zero = N_free + k2*(0.099 - 0.296 sin^2 phi).
    => N_mean - N_free = (1 + k2)(0.099 - 0.296 sin^2 phi).
    At 45 deg: (1.30)(0.099-0.148) = -0.064 m.
    Heights H = h_ell - N shift by the OPPOSITE sign.
    Verify convention against JPL D-109532 before SWOT production use."""
    s2 = np.sin(np.radians(np.asarray(lat, dtype=float))) ** 2
    return (1.0 + K2_LOVE) * (0.099 - 0.296 * s2)


# -----------------------------------------------------------------------------
# Core conversion
# -----------------------------------------------------------------------------

def build_samplers(args):
    # zeta_RO
    if args.zeta_spg:
        zeta = SpgSampler(args.zeta_spg)
        zsrc = f"spg:{args.zeta_spg}"
    elif args.zeta_grid:
        zeta = GridSampler(args.zeta_grid)
        zsrc = f"grid:{args.zeta_grid}"
    elif args.benchmarks:
        zeta = fit_zeta_from_benchmarks(args.benchmarks, args.benchmark_fit)
        zsrc = f"benchmarks:{args.benchmarks}"
    elif args.zeta_const is not None:
        zeta = ConstSampler(args.zeta_const)
        zsrc = f"const:{args.zeta_const}"
    else:
        raise SystemExit("Provide a zeta_RO source: --zeta-spg | --zeta-grid | --benchmarks | --zeta-const")
    # N_EGM2008
    if args.egm_grid:
        egm = GridSampler(args.egm_grid)
        esrc = f"grid:{args.egm_grid}"
    elif args.egm_pyproj:
        egm = PyprojEGM()
        esrc = "pyproj:EPSG3855"
    elif args.egm_const is not None:
        egm = ConstSampler(args.egm_const)
        esrc = f"const:{args.egm_const}"
    else:
        raise SystemExit("Provide an EGM2008 source: --egm-grid | --egm-pyproj | --egm-const")
    print(f"[sources] zeta_RO = {zsrc}   N_EGM2008 = {esrc}   tide = {args.tide}")
    return zeta, egm


def correction_C(lat, lon, zeta, egm, tide):
    """C such that H_EGM2008 = H_MN75 + C."""
    z = zeta.sample(lat, lon)
    n = egm.sample(lat, lon)
    if tide == "mean":
        n = n + tide_free_to_mean_dN(lat)
    return z - n, z, n


def convert_values(h, lat, lon, direction, zeta, egm, tide):
    C, z, n = correction_C(lat, lon, zeta, egm, tide)
    if direction == "mn75-to-egm":
        out = h + C
    elif direction == "egm-to-mn75":
        out = h - C
    else:  # report
        out = C
    return out, C, z, n


# -----------------------------------------------------------------------------
# Points / raster drivers
# -----------------------------------------------------------------------------

def run_points(args, zeta, egm):
    import pandas as pd
    df = pd.read_csv(args.input)
    for col in (args.lat_col, args.lon_col):
        if col not in df.columns:
            raise SystemExit(f"{args.input}: missing column '{col}'")
    lat = df[args.lat_col].to_numpy(float)
    lon = df[args.lon_col].to_numpy(float)
    if args.direction == "report":
        h = np.zeros_like(lat)
    else:
        if args.h_col not in df.columns:
            raise SystemExit(f"{args.input}: missing height column '{args.h_col}'")
        h = df[args.h_col].to_numpy(float)
    out, C, z, n = convert_values(h, lat, lon, args.direction, zeta, egm, args.tide)
    df["zeta_RO_m"] = np.round(z, 4)
    df["N_egm2008_m"] = np.round(n, 4)
    df["C_mn75_to_egm_m"] = np.round(C, 4)
    if args.direction == "mn75-to-egm":
        df["H_egm2008_m"] = np.round(out, 4)
    elif args.direction == "egm-to-mn75":
        df["H_mn75_m"] = np.round(out, 4)
    df.to_csv(args.output, index=False)
    ok = np.isfinite(C)
    print(f"[points] {ok.sum()}/{len(df)} converted; "
          f"C = {np.nanmean(C):+.3f} m (min {np.nanmin(C):+.3f}, max {np.nanmax(C):+.3f})")
    print(f"[points] wrote {args.output}")


def run_raster(args, zeta, egm):
    import rasterio
    from rasterio.warp import transform as rio_transform
    with rasterio.open(args.input) as src:
        prof = src.profile.copy()
        data = src.read(1).astype(float)
        nod = src.nodata
        mask = np.isfinite(data)
        if nod is not None:
            mask &= data != nod
        # pixel-center coords of every valid cell -> lat/lon
        rows, cols = np.where(mask)
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        xs = np.asarray(xs, float); ys = np.asarray(ys, float)
        if src.crs and src.crs.to_epsg() not in (4326, 4979):
            lon, lat = rio_transform(src.crs, "EPSG:4326", xs, ys)
            lon = np.asarray(lon, float); lat = np.asarray(lat, float)
        else:
            lon, lat = xs, ys
        out_vals, C, _, _ = convert_values(
            data[rows, cols], lat, lon, args.direction, zeta, egm, args.tide)
        out = np.full_like(data, np.nan if nod is None else nod, dtype=float)
        result = out_vals if args.direction != "report" else C
        out[rows, cols] = result
        bad = ~np.isfinite(result)
        if bad.any():
            print(f"[raster] WARNING: {bad.sum()} cells fell outside a geoid grid -> nodata")
            out[rows[bad], cols[bad]] = np.nan if nod is None else nod
        prof.update(dtype="float32", nodata=np.nan if nod is None else nod)
        with rasterio.open(args.output, "w", **prof) as dst:
            dst.write(out.astype("float32"), 1)
    print(f"[raster] C = {np.nanmean(C):+.3f} m over {np.isfinite(C).sum()} cells; "
          f"wrote {args.output}")


# -----------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["points", "raster"])
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--direction", required=True,
                   choices=["mn75-to-egm", "egm-to-mn75", "report"])
    p.add_argument("--zeta-spg", help="ANCPI rom_grid3d_*.spg (romgeo-grid download)")
    p.add_argument("--zeta-grid")
    p.add_argument("--zeta-const", type=float)
    p.add_argument("--benchmarks")
    p.add_argument("--benchmark-fit", choices=["auto", "const", "plane"], default="auto")
    p.add_argument("--egm-grid")
    p.add_argument("--egm-pyproj", action="store_true")
    p.add_argument("--egm-const", type=float)
    p.add_argument("--tide", choices=["none", "mean"], default="none")
    p.add_argument("--lat-col", default="lat")
    p.add_argument("--lon-col", default="lon")
    p.add_argument("--h-col", default="h")
    args = p.parse_args(argv)
    zeta, egm = build_samplers(args)
    if args.mode == "points":
        run_points(args, zeta, egm)
    else:
        run_raster(args, zeta, egm)


if __name__ == "__main__":
    main()
