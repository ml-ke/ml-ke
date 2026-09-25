#!/usr/bin/env bash
# Install R + the system C libraries that the common R geospatial/image stack links
# against, plus the Quarto CLI. Idempotent: safe to re-run.
#
# Target: Ubuntu/Debian. For another distro swap the CRAN repo suite (noble-cran40
# -> jammy-cran40 / bookworm-cran40) and the lib package names.
#
# Run with sudo available via $SUDO so it works with a wrapper:
#   SUDO=$HOME/.local/bin/sudox bash install-r-system-deps.sh
set -euo pipefail
SUDO="${SUDO:-sudo}"
export DEBIAN_FRONTEND=noninteractive

SUITE="noble-cran40"   # adjust for your release

# --- 1. CRAN apt repo (apt's own r-base is usually several releases behind) ---
$SUDO install -d -m 0755 /etc/apt/keyrings
curl -fsSL https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
  | $SUDO tee /etc/apt/keyrings/cran_r.asc >/dev/null
echo "deb [signed-by=/etc/apt/keyrings/cran_r.asc] https://cloud.r-project.org/bin/linux/ubuntu ${SUITE}/" \
  | $SUDO tee /etc/apt/sources.list.d/cran_r.list >/dev/null
$SUDO apt-get update -qq

# --- 2. R + the libraries the R packages actually link ---
# gdal/geos/proj  -> terra, sf
# libmagick++-dev -> magick
# libuv1-dev      -> fs        (absent => configure fails on uv.h, cascades widely)
# libudunits2-dev -> units (terra/sf time handling)
# harfbuzz/fribidi/fontconfig/freetype -> systemfonts/ragg/textshaping (plot text)
$SUDO apt-get install -y -qq --no-install-recommends \
  r-base r-base-dev \
  libgdal-dev gdal-bin libgeos-dev libproj-dev proj-data proj-bin \
  libudunits2-dev libsqlite3-dev \
  libcurl4-openssl-dev libssl-dev libxml2-dev libgit2-dev libuv1-dev \
  libmagick++-dev imagemagick \
  libfontconfig1-dev libfreetype6-dev libharfbuzz-dev libfribidi-dev \
  libpng-dev libtiff5-dev libjpeg-dev libwebp-dev \
  pandoc cmake pkg-config build-essential zlib1g-dev
# NOTE: do NOT add pandoc-citeproc here - it no longer exists on noble-class
# releases (merged into pandoc >= 2.19) and a single unavailable package aborts
# the entire apt-get install list.

# --- 3. Quarto CLI (needed to render .qmd books/reports) ---
if ! command -v quarto >/dev/null 2>&1; then
  QVER="${QUARTO_VERSION:-1.6.42}"
  TMP=$(mktemp -d)
  if curl -fsSL -o "$TMP/quarto.deb" \
      "https://github.com/quarto-dev/quarto-cli/releases/download/v${QVER}/quarto-${QVER}-linux-amd64.deb"; then
    $SUDO dpkg -i "$TMP/quarto.deb" >/dev/null 2>&1 || $SUDO apt-get install -y -f -qq
  else
    echo "WARN: quarto download failed; .qmd rendering will not work" >&2
  fi
  rm -rf "$TMP"
fi

# --- 4. verify ---
R --version | head -2
command -v quarto >/dev/null && quarto --version || echo "quarto: NOT installed"
gdalinfo --version || echo "gdalinfo: NOT installed"
echo "SYSTEM_DEPS_OK"
