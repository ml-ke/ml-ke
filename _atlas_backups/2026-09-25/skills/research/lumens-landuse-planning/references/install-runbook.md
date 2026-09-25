# Exact scripts used to install and verify LUMENS

Numbered so the order is unambiguous. All live in the working directory (e.g. `~/Dev/lumens`).

## 00_install_sys_deps.sh — system layer

```bash
#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# CRAN apt repo (gives a newer R than noble's)
sudo install -d -m 0755 /etc/apt/keyrings
curl -fsSL https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
  | sudo tee /etc/apt/keyrings/cran_r.asc >/dev/null
echo "deb [signed-by=/etc/apt/keyrings/cran_r.asc] https://cloud.r-project.org/bin/linux/ubuntu noble-cran40/" \
  | sudo tee /etc/apt/sources.list.d/cran_r.list >/dev/null
sudo apt-get update -qq

sudo apt-get install -y -qq --no-install-recommends \
  r-base r-base-dev libgdal-dev gdal-bin libgeos-dev libproj-dev proj-data proj-bin \
  libudunits2-dev libsqlite3-dev libuv1-dev \
  libcurl4-openssl-dev libssl-dev libxml2-dev libgit2-dev \
  libmagick++-dev imagemagick \
  libfontconfig1-dev libfreetype6-dev libharfbuzz-dev libfribidi-dev \
  libpng-dev libtiff5-dev libjpeg-dev libwebp-dev \
  pandoc cmake pkg-config build-essential zlib1g-dev
# NB: no pandoc-citeproc on noble

# Quarto CLI (for report rendering)
QVER=1.6.42
TMP=$(mktemp -d)
curl -fsSL -o "$TMP/quarto.deb" \
  "https://github.com/quarto-dev/quarto-cli/releases/download/v${QVER}/quarto-${QVER}-linux-amd64.deb"
sudo dpkg -i "$TMP/quarto.deb" || sudo apt-get install -y -f -qq
```

## 10_finalize_deps.R — R layer (run under a retry loop)

```r
options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/noble/latest"),
        timeout = 1800L, Ncpus = 4L)
user_lib <- file.path(Sys.getenv("HOME"), "R", "x86_64-pc-linux-gnu-library", "4.6")
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))
library(pak)
need <- c(  # LUMENSR Imports, then the Shiny module stack
  "ggplot2","tidyterra","scales","sf","terra","magrittr","dplyr","purrr","gt",
  "networkD3","textclean","stringr","ggrepel","tidyr","forcats","viridis",
  "cowplot","rlang","magick","tibble","quarto","knitr","testthat",
  "shiny","shinyjs","shinydashboard","shinyFiles","shinyalert","bslib",
  "rmarkdown","DT","leaflet","mapview","htmlwidgets","sass","reactable",
  "data.tree","lexicon","qdapRegex","textshape","htmlTable","kableExtra",
  "writexl","openxlsx","readr","readxl","reshape2","plotly","patchwork",
  "caTools","DBI","RSQLite","conflicted","rasterVis","splitstackshape","tiff",
  "ggthemes","corrplot","hexbin","latticeExtra","plyr","gridExtra","randomForest"
)
pak::pkg_install(need, ask = FALSE, upgrade = FALSE, dependencies = TRUE)
```

Wrap it, because pak can hang on a stalled socket:

```bash
for i in $(seq 1 8); do
  timeout 1500 Rscript 10_finalize_deps.R >> /tmp/pak.log 2>&1
  grep -aq DEPS_COMPLETE /tmp/pak.log && break
done
```

## 02_install_lumensr.R

```r
remotes::install_github("icraf-indonesia/LUMENSR", dependencies = TRUE,
                        upgrade = "never", build_vignettes = FALSE, force = TRUE)
# fall back to remotes::install_local("<clone path>") if the ~90 MB tarball drops
```

## Pre-QuES functional test — the parts that are easy to get wrong

```r
lk <- lc_lookup_klhk_sequence
lc_t1 <- add_legend_to_categorical_raster(rast(LUMENSR_example("NTT_LC90.tif")), lk, year = 1990)
lc_t2 <- add_legend_to_categorical_raster(rast(LUMENSR_example("NTT_LC20.tif")), lk, year = 2020)
adm_r <- rast(LUMENSR_example("ntt_admin_spatraster.tif"))   # NOT rasterise_multipolygon(ntt_admin)
preq  <- ques_pre(lc_t1, lc_t2, adm_r, cutoff_landscape = 5000, cutoff_pu = 500, convert_to_Ha = TRUE)
```

Then save the annotated rasters and the `ques_pre()` result, stage the child qmd, and
render with params:

```r
work <- "outputs/report"
for (f in c("PreQUES_report.qmd", "quespre_by_pu.qmd")) file.copy(LUMENSR_example(f), file.path(work, f))
writeRaster(lc_t1, file.path(work, "lc_t1.tif"), overwrite = TRUE)
writeRaster(lc_t2, file.path(work, "lc_t2.tif"), overwrite = TRUE)
writeRaster(adm_r, file.path(work, "admin.tif"), overwrite = TRUE)
saveRDS(preq, file.path(work, "ques_pre_output.rds"))
setwd(work)
quarto::quarto_render("PreQUES_report.qmd", output_file = "PreQUES_report.html",
  execute_params = list(dir_lc_t1_ = "lc_t1.tif", dir_lc_t2_ = "lc_t2.tif",
                        dir_admin_ = "admin.tif", dir_ques_pre = "ques_pre_output.rds",
                        cutoff_landscape = 5000, cutoff_pu = 500))
```

## Counting test results honestly

```r
res <- testthat::test_dir("tests/testthat", reporter = "summary", stop_on_failure = FALSE)
df  <- as.data.frame(res)
sum(df$failed); sum(df$error)   # count BOTH; errors hide in `error`
```
