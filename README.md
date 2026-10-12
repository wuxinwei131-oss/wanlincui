# Frozen crop forecast core reproduction

Reproduction code and the selected frozen inputs for *Satellite augmentation does not ensure beneficial adaptive selection in corn and soybean yield forecasting*. The scientific scope retains the validated 59-file core; four additional files provide the software license, data license scope, author credit and per-member rights schedule. Code is covered by the MIT license in LICENSE. Author-owned rights in the 33 selected data members are covered by CC BY 4.0 as scoped in DATA_LICENSE.md; public-domain and upstream elements retain their original status. See AUTHORS.md and DATA_RIGHTS.csv.

Level A independently evaluates retained county-year outcomes, forecasts, selection actions and weights. It does not refit, tune or select models. Level B reconstruction from original raw sources and training is **NOT_RUN**. R3 tested Python 3.13.14; the R4 core retest used a newly created Python 3.12.14 environment. These verification environments are not historical execution evidence.

From this directory on Windows, using Python 3.12 or 3.13:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt -c requirements-lock.txt
.venv\Scripts\python -B run_reproduction.py
```

On Unix, use `.venv/bin/python`; that platform was not separately tested. The entry verifies the core file hashes, extracts `data/frozen_core.zip` into `results/inputs`, and writes results under `results/`. Run from a writable directory. It needs no account, external service or original research directory. Installation of dependencies needs access to their package provider; the numerical run is offline.

Coverage includes six-condition sample/block counts, PB3 and augmentation/selection gains; S1 eligibility and flow; S5's 13 variants; S6 groups and weighting; S7 common samples and environment omissions; S8 action losses; S9 recoverable information sets; S10 interval reliability; and descriptive paired resampling. Methods for S2/S3 are disclosed in `methods/` and `config/`, without a training claim.

Expected checks: core 2792/2792; S5 106/106; paired ranges 56/56. S9 is **256/262**: six raw-Parquet versus ten-significant-digit CSV endpoint comparisons exceed the unchanged 1e-10 threshold. Successful command completion means this documented evidence pattern is reproduced, not that those six comparisons passed. Workbook reporting precision, stored Parquet values, evaluation CSVs and archived statistics remain distinct. See `SCIENTIFIC_LIMITATIONS.md`.

This core does **not** contain the 20-file optional drawing package and does **not** redraw all twelve paper figures. The frozen numerical tables support core claims. The complete twelve-figure redraw was verified separately in R3 using the privately retained optional package. `--figures` is intentionally unavailable here and must not be reported as successful. Figure 2 remains the author-approved current manuscript version; Figures 4/5 use the approved 20261011 versions in that optional package. The publication manuscript and its figures are not bundled here.

Actual scientific method implementations and their original source hashes are provided in `methods/SOURCE_VERSIONS.json`; retained execution settings are in `config/method_parameters.json`. For optional source-import and no-fit checks, install the broader method dependencies:

```powershell
.venv\Scripts\python -m pip install -r requirements-methods.txt -c requirements-lock.txt
.venv\Scripts\python -B check_method_sources.py
```

Run the numerical entry first. The method check was verified in R3; R4 reruns the numerical core and checks dependency resolution. It does not fit models or authenticate to Earth Engine. Original cache locking is Windows-specific. The method requirements include geospatial libraries used by actual data-construction source; those dependencies do not supply the omitted figure package. Level A never calls the original training entry points.

`requirements.txt` contains only numpy and pandas. The lock file and dependency-license table also retain the broader R3 verification environment as version constraints and provenance; do not install every locked package for numerical use. PyMuPDF is not required. Third-party raw raster collections and geographic boundaries are obtained from official providers linked in `DATA_SOURCES.md`, rather than included in the core.
