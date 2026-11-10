---
name: lumens-landuse-planning
description: Use when running or testing ICRAF LUMENS (R/Shiny).
---

# LUMENS — install, run, test

LUMENS (Land Use Planning for Multiple Environmental Services, CIFOR-ICRAF) is an R
land-use planning toolkit: spatially explicit, semi-agent-based modelling for carbon,
biodiversity, hydrology, profitability and regional-economics trade-off analysis.

## 1. Know which of the three repos you need

| Repo | What it is |
|---|---|
| `icraf-indonesia/LUMENSR` | The **R package** = the analysis engine. Only the Pre-QuES module is packaged. 32 exported fns, 13 example datasets in `inst/extdata`. |
| `icraf-indonesia/lumens-shiny` | The **GUI**: `app.R` renders a 13-button launcher and shells out to `callNN.R`, each of which runs one module's own Shiny app under `<NN>_<module>/rscript/`. ~400 MB, mostly help videos. |
| `icraf-indonesia/lumensbook` | Quarto **user guide** published at https://help.lumens.or.id |

**Dependency direction matters:** modules `04_quesc`, `11_sciendo-train` and
`11_sciendo-simulate` call `library(LUMENSR)`. Install the package first or those
modules cannot start. Verify with `grep -rl "library(LUMENSR)" **/rscript/*.R`.

Module map: PUR build/reconcile (01/02), Pre-QuES (03), QUES-C/B/H (04/05/06),
LU Profitability (07), Regional Economics descriptive/projection (08/09),
Scenario Builder (10), DINAMICA train/simulate (11/12), LASEM (12_lasem).

## 2. Install (Linux)

1. System layer: CRAN apt repo `noble-cran40`, then `r-base-dev libgdal-dev gdal-bin
   libgeos-dev libproj-dev libudunits2-dev libsqlite3-dev libcurl4-openssl-dev
   libssl-dev libxml2-dev libgit2-dev libmagick++-dev imagemagick libfontconfig1-dev
   libfreetype6-dev libharfbuzz-dev libfribidi-dev libpng-dev libtiff5-dev libjpeg-dev
   libwebp-dev pandoc cmake build-essential zlib1g-dev` **plus `libuv1-dev`** — the
   non-obvious one; without it `fs` fails to configure and takes 40+ packages with it.
   Quarto CLI deb for report rendering.
2. R layer: **use `pak::pkg_install()`, not hand-ordered `install.packages()`**. pak
   resolves the dependency graph; a manual list that puts dependents first
   (gt/networkD3/textclean/shiny before htmlwidgets/sass/reactable/bslib/data.tree)
   fails with "dependencies X are not available for package Y".
3. `remotes::install_github("icraf-indonesia/LUMENSR")`.

See `references/install-runbook.md` for the exact scripts.

## 3. Run and test

- Engine test: run the Pre-QuES pipeline on the bundled NTT dataset and render the
  shipped Quarto report. Use `references/install-runbook.md` as the template.
- GUI: `LUMENS_OPEN_BROWSER=false LUMENS_PORT=875 ./run_lumens_launcher.sh`, then open
  the URL. Patch `app.R` first (§5).
- Guide (lumensbook): plain `quarto render` builds 21 pages into `docs/`. The book has
  no `_freeze` but is nearly all prose plus pre-rendered media — only `13-rice_ID.qmd`
  (3 chunks) and `index.qmd` (1 chunk) contain R, and the CSA chapter reads
  `images/CSA/*.tif|csv` that ship in the repo. Verify in two passes: `--no-execute`
  first to prove structure/cross-refs/bibliography resolve, then a real render. The
  knitr step log looks **identical** either way, so treat generated
  `<chapter>_files/figure-html/*.png` plus `cell-output` blocks in the HTML as the only
  proof that code actually executed.

## 4. Upstream bugs to expect (all reproduced)

1. **`rasterise_multipolygon()` output is unusable as a crosstab zone.** It inherits
   the polygon bbox and a non-integer grid (≈99.995 × 99.986 m), so
   `terra::compareGeom(stopOnError = TRUE)` inside `create_crosstab()` aborts with
   `[compareGeom] extents do not match`. CRS and dimensions match; only the extent/res
   are off. **Workaround: use the shipped `inst/extdata/ntt_admin_spatraster.tif` as
   the planning-unit input**, or `terra::resample(z, lc_t1, method = "near")` before
   passing a rasterised shapefile. The shapefile planning-unit path is broken.
2. **The upstream testthat suite has 2 hard errors** — `test-harmonise_raster.R` calls
   `st_crs()` without attaching sf, and `test-create_crosstab.R` hits bug 1. When
   auditing test results, count the `error` column: `as.data.frame(res)$error`. A check
   that only looks at `failed` reports green over real errors.
3. **`app.R` hardcodes `rscript.exe`** — Windows only. No module launches on Linux.
4. **All 13 `callNN.R` pin `port = 875`, which is a PRIVILEGED port (<1024).** Nothing
   starts as a non-root Linux user: `bind 127.0.0.1:875` returns `EACCES`, which httpuv
   reports as the misleading `createTcpServer: permission denied / Failed to create
   server`. Windows has no privileged-port concept, which is why upstream never saw it.
   Always run with a port >= 1024 (`LUMENS_PORT=3838`); that also lets two modules run
   at once, since they all shared 875. `shell.exec()` (open output folder / open report)
   appears in ~15 module files and is Windows-only.
5. **`10_sciendo-scenario` installs an unpinned package from a personal GitHub account at
   app startup.** `10_sciendo-scenario/rscript/global.R` runs
   `install_github("degi/abacuslib")` when `abacuslib` is absent, then
   `library(abacuslib)`. The repo exists, but the install is `@HEAD` (no pinned commit,
   no PAT), so startup is non-deterministic: it fails on an offline box, behind a proxy,
   or once the unauthenticated GitHub API rate limit (60/hr) is hit, and the error is a
   bare `cannot open URL .../tarball/HEAD`. Do **not** install it as a side effect of
   testing — unpinned third-party code execution is the user's call. In a sandboxed or
   locked-down environment, expect this module to fail while the other 12 load fine.
6. **Four modules run `install.packages()` at app startup** via
   `check_and_install_packages()` — observed in `02_pur2` (rosm, ggspatial), `04_quesc`
   (reshape), `10_sciendo-scenario` (excelR), `11_sciendo-simulate` (openxlsx2). Opening
   a module therefore needs network plus a writable R library, and can kick off source
   compilation before the user clicks anything. Pre-install these to make startup
   deterministic and offline-capable.

## 5. The Linux port of the GUI

Patch, don't fork; keep originals as `*.orig`.

**Port rules — get both right or nothing starts.** Modules need a port that is
(a) >= 1024, because 875 is below it and non-root users get `EACCES / permission denied`,
and (b) **different from the launcher's own port**, because the launcher holds its port
for as long as it runs and children otherwise die with `createTcpServer: address already
in use`. Upstream's 875 exists precisely to dodge the launcher's default 3838 — and
inheriting one shared `LUMENS_PORT` for both roles reintroduces the collision, which is a
mistake worth avoiding (it happened: modules loaded every package, then failed on bind).
Use a separate `LUMENS_MODULE_PORT`, or have the launcher allocate a free port per launch
with `httpuv::randomPort(min = 1024L)` and pass it to the child as `LUMENS_PORT`.

- In `app.R`: `R_BIN <- if (.Platform$OS.type == "windows") "rscript.exe" else "Rscript"`;
  launcher port from `LUMENS_PORT` (default >= 1024, e.g. 3838); browser behaviour from
  `LUMENS_OPEN_BROWSER` (default false, works headless/over SSH); bind `0.0.0.0`.
- Spawn modules with `system2(R_BIN, args = shQuote(script), stdout = log, stderr = log,
  env = c(paste0("LUMENS_PORT=", port), ...))` — `system2(env=)` is cross-platform and
  avoids shell quoting. Do NOT copy `LUMENS_PORT` into the child unchanged.
- Show the module URL in a notification before the slow load, and on non-zero exit
  report the log path plus the last lines. Upstream's bare `system()` discards output, so
  a startup crash just looks like a frozen launcher.
- In every `callNN.R`: `launch.browser = tolower(Sys.getenv("LUMENS_OPEN_BROWSER",
  "false")) %in% c("true","1","yes")` and
  `port = as.integer(Sys.getenv("LUMENS_PORT", "3838"))`.
- Guard every `shell.exec(x)` as `if (.Platform$OS.type == "windows") shell.exec(x) else
  message("Path (open manually on Linux): ", x)`.
- `Rscript -e 'invisible(parse("app.R"))'` after patching.
- Proving it works means starting the launcher AND a module at the same time and probing
  both ports; a sequential smoke test passes even with the collision present. See
  `13_test_dual_ports.sh` in the project for the pattern.

## 6. Rendering the shipped Pre-QuES report

`inst/extdata/PreQUES_report.qmd` is parameterised and its params default to `"NA"`, so
rendering it bare fails with "Error running quarto CLI from R". Supply all of:
`dir_lc_t1_`, `dir_lc_t2_`, `dir_admin_` (paths to **annotated** rasters, written with
`writeRaster` so categories and time survive), `dir_ques_pre` (an `.rds` of the full
`ques_pre()` return value) and the two cutoffs. **Also copy the child document
`quespre_by_pu.qmd` next to it** — the report `knit_child()`s it by filename from the
working directory, and it is easy to miss because the error names only the parent.

## 7. When R says `createTcpServer: permission denied`

This is almost always a **privileged port**, not a sandbox or a firewall. An httpuv app
that logs `Listening on http://0.0.0.0:875` and then `createTcpServer: permission denied
/ Error in initialize(...) : Failed to create server` is failing because ports below
1024 require root, and `EACCES` gets reported as "permission denied".

Diagnose in three cheap steps, as the same unprivileged user:

```bash
sysctl net.ipv4.ip_unprivileged_port_start        # 1024 = ports <1024 need root
python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',875));s.listen(1);print('OK')"
python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',18875));s.listen(1);print('OK')"
grep Seccomp /proc/self/status                    # 0 = no seccomp filter to blame
```

Low port fails + high port succeeds = privileged port, full stop. Fix by choosing a port
>= 1024 (`LUMENS_PORT`), or by granting `CAP_NET_BIND_SERVICE` if the low port is truly
required. Do **not** reach for sandbox / AppArmor / firewall theories first — that cost a
wrong diagnosis once.

A useful side effect: each module prints `Listening on` *after* loading all of its
packages, so a smoke test that probes every module and greps its log for `Listening on`
separates "the app is broken" from "the bind failed". Record per-module results to a TSV
and keep each process's log so any pass/fail claim is justified per module.

## 8. Pitfalls

- R's default 60 s `download.file` timeout silently drops large packages
  (data.table/stringi/igraph) and cascades into dozens of failures. Set
  `options(timeout = 900)`.
- Install into a **user library**: R only auto-adds
  `~/R/x86_64-pc-linux-gnu-library/<major.minor>` if that directory already exists.
  Create it and no sudo is needed for packages.
- `pandoc-citeproc` does not exist on Ubuntu noble (merged into pandoc).
- `spatial.tools` is archived on CRAN; only a module 07 test script references it.
- `pak` can stall on a hung socket; re-run it — the download cache makes passes cheap.
