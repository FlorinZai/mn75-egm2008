# MN75 ⇄ EGM2008 height conversion (Romania)

Converts heights between the Romanian normal-height datum **MN75 (Marea Neagră 1975)** and **EGM2008**, via ETRS89 ellipsoidal heights:

```
h_ell     = H_MN75 + ζ_RO(φ, λ)        (Romanian quasigeoid, RomGEO grid)
H_EGM2008 = h_ell  − N_EGM2008(φ, λ)    (EGM2008 undulation)
⇒ H_EGM2008 = H_MN75 + C,   C = ζ_RO − N_EGM2008
```

ζ_RO is sampled with romgeo's own B-spline interpolator, so it matches TransDatRO.

## Contents

| File | Purpose |
|---|---|
| `transform_simple.py` | Minimal Python API + CLI (`egm_to_mn75`, `mn75_to_egm`, `correction`) |
| `mn75_egm2008.py` | Full tool: points (CSV) and rasters (GeoTIFF), several ζ / N sources, benchmark calibration |
| `correct_swot_to_mn75.py` | Adds an MN75 height variable to a SWOT WSE NetCDF (streamed, resumable) |
| `rom_grid3d_25.09.spg` | RomGEO grid release 25.09 (third-party, see *Data licences*) |
| `rom_grid3d_25.09.json` | Same RomGEO grid release 25.09 in JSON form (third-party, see *Data licences*) |
| `metadata.json` | Licence and release metadata for the RomGEO grid, as distributed by CNC |
| `us_nga_egm08_25.tif` | EGM2008 2.5′ undulation grid (tide-free), from https://cdn.proj.org/us_nga_egm08_25.tif |

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# single point: lat lon h
python transform_simple.py egm-to-mn75 44.895 29.60 2.34

# CSV with columns lat,lon,h
python transform_simple.py mn75-to-egm --csv in.csv out.csv

# SWOT NetCDF -> adds height_final_daily_merged_mn75
python correct_swot_to_mn75.py SWOT_PIXC_clean.nc
```

```python
from transform_simple import egm_to_mn75, mn75_to_egm
H = egm_to_mn75(44.895, 29.60, 2.34)   # scalars or numpy arrays
```

**Tide system:** the EGM2008 grid is tide-free. For SWOT (mean-tide geoid) add `--tide-mean`. Check the convention against the SWOT User Handbook (JPL D-109532) before production use.

**Security note:** `.spg` files are Python pickles. Load only grids obtained from the official RomGEO source.

## Data licences

- **`rom_grid3d_25.09.spg`** and **`rom_grid3d_25.09.json`**: *RomGEO Densified transformation Grid* © Centrul Național de Cartografie (CNC), https://romgeo.ro/grids, licensed under **CC BY-ND 4.0** (https://creativecommons.org/licenses/by-nd/4.0/). Included **unmodified**, with the original `metadata.json`. Do not redistribute modified, cropped or resampled versions of them.
- **`us_nga_egm08_25.tif`**: EGM2008 geoid model by the US National Geospatial-Intelligence Agency (NGA), as packaged by the PROJ project (https://cdn.proj.org).
- **`romgeo`** (Python dependency, installed via pip): © CNC, CC BY-SA 4.0. Not redistributed here.

## Code licence

The **MIT licence in `LICENSE` covers only the Python scripts (`*.py`)** in this repository. It does **not** apply to the data files above, which keep their own licences.

## Author

Iulian-Florin Zăinescu, University of Bucharest (ICUB / Faculty of Geography)
