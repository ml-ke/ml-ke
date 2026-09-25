---
name: r-project-bringup
description: "Use when setting up or verifying an R project on Linux."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [r, rstats, shiny, quarto, geospatial, terra, sf, verification, package-install]
    related_skills: [codebase-inspection, systematic-debugging]
---

# R Project Bring-Up & Verification

Covers R **packages**, **Shiny apps**, and **Quarto books/reports** brought up from
source on Linux: installing R plus the geospatial/Shiny dependency stack without
sudo, then proving the thing works by real execution rather than a green install log.

## When to use

- "clone this R repo and run it", "install this R package and test it", "get this Shiny app running"
- Reproducing a research tool / analysis package that must actually execute, not merely install
- Rendering a Quarto book or report template that ships inside a package

## Order of operations

Do these in order; each one de-risks the next.

1. **Recon the repo** — read the real dependency list and the entry points before installing anything.
2. **System layer** (apt) — R itself, the geo/image C libraries, Quarto CLI.
3. **R package layer** — user library, precompiled binaries, a resolver for the dependency graph.
4. **Install the target** — from GitHub, with a local-clone fallback.
5. **Functional verification** — the project's own test suite + its README example.
6. **Interactive verification** — Shiny apps and rendered documents.

## 1. Recon before installing

- `DESCRIPTION` (or `README.md` for non-packages) IS the dependency list — fetch it raw rather than reading it in a browser:
  `curl -sL https://raw.githubusercontent.com/<org>/<repo>/main/DESCRIPTION`
- For apps and books, grep the imports the README never mentions. Shiny templates often carry a `required_packages <- c(...)` vector, and each module is its own app:
  `grep -rhoE "^\s*(library|require)\(['\"]?[A-Za-z][A-Za-z0-9._]*" --include='*.R' . | sed "s/.*(//;s/['\"]//" | sort | uniq -c | sort -rn`
- Grep for Windows-isms early — they change the run plan, not just the polish:
  `grep -rn "rscript\.exe\|shell\.exec" --include='*.R' .`
- Map the entry points: `app.R` vs `ui.R`+`server.R`, `call*.R` sub-app launchers, `tests/testthat/`, `inst/extdata/*.qmd`, bundled example rasters.
- Check the repo's own CI config (`R CMD check` workflow) — it names the Suggests the authors actually exercise.

## 2. System layer

Run `scripts/install-r-system-deps.sh` (Ubuntu/Debian; adjust the CRAN repo suite for other distros). It covers the CRAN apt repo for current R, GDAL+GEOS+PROJ (terra/sf), `libmagick++-dev` (magick), `libuv1-dev` (fs), pandoc, and the Quarto CLI.

Pitfalls:
- **One unavailable package aborts the whole `apt-get install -y` list.** `pandoc-citeproc` no longer exists on Noble-class releases (merged into pandoc ≥2.19) and killed an install halfway through. Drop or existence-check extras rather than appending them hopefully.
- **`fs` is a silent blocker** — without `libuv1-dev` it fails at configure with `uv.h: No such file or directory`, then everything depending on it cascades.
- Verify the layer before moving on: `R --version`, `quarto --version`, `gdalinfo --version`.

## 3. R package layer

Full recipe with the failure transcripts distilled: `references/linux-r-stack-setup.md`. The load-bearing rules:

- **Install into a user library.** The apt-installed R puts `site-library` under `/usr/local/lib/R` (root-owned) and `install.packages` dies with `'lib = "…"' is not writable`. R automatically adds `~/R/x86_64-pc-linux-gnu-library/<maj.min>` to `.libPaths()` **only if that directory already exists** — so `dir.create()` it and no sudo is needed at all.
- **Set `options(timeout = 900)` before any install.** The default 60s cut off downloads on a slow link; the failure surfaces as `dependency 'stringi' is not available`, not as a download error, so one timeout became 40+ apparent package failures.
- **Use precompiled Linux binaries** from Posit Package Manager instead of compiling terra/sf/igraph from source:
  `options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/noble/latest"))`
  pak reports these as `x86_64-pc-linux-gnu-ubuntu-<ver>`; a plain source-CRAN run compiles for tens of minutes on a throttling laptop.
- **Let a resolver order the graph.** A hand-written list fails whenever a dependent precedes its dependency (`gt` before `htmlwidgets`/`sass`/`reactable`, `shiny` before `bslib`) and every retry re-fails identically. Use `pak::pkg_install(pkgs, ask = FALSE, upgrade = FALSE, dependencies = TRUE)` — see `scripts/install-r-packages.R`.
- **Strip two classes of package from the list** or the solver refuses outright: base/recommended packages that ship with R (`foreign`, `lattice`) — asked to install them into another lib it reports a bare `dependency conflict`; and archived CRAN packages — `Can't find package called <x>`. Install an archived one from its CRAN Archive URL only if something actually loads it.
- **A stalled download idles the resolver at ~0% CPU with the log frozen.** Wrap it in a retry loop with a per-pass `timeout`; pak caches to `~/.cache/R/pkgcache`, so each pass resumes cheaply instead of restarting.
- **Install the target with a fallback**: `remotes::install_github("<org>/<repo>")` inside `try()`, then `remotes::install_local("<clone dir>")` if it fails — a dropped large tarball must not block verification.

## 4. Functional verification

Run the project's own tests AND its README example from a script whose output you save, then read the raw log.

- **Count `error`, not just `failed`.** `as.data.frame(testthat::test_dir(...))` puts failures in `failed` and thrown errors in `error`; `stopifnot(sum(df$failed) == 0)` prints PASS over a suite containing real errors. Sum both, and print the per-file table.
- **Never report from your own harness's PASS summary alone** — grep the saved log for `ERROR`, `non-zero exit status`, `could not find function`, `installation of package .* failed`. A `tryCatch`-based harness records an inner step as PASS while the real error sits in the log.
- **Run the project's test suite early, even before your own harness works.** Upstream failures explain your harness failures and are themselves the finding — report them as reproduced bugs with the literal error text, not as your setup being wrong.

### Quarto documents ship non-self-contained

- Read the template's YAML first: a `params:` block must be supplied through `execute_params`, or the render dies with an opaque `Error returned by quarto CLI`.
- A template that calls `knitr::knit_child("child.qmd")` needs **that child file staged in the same working directory** — copying only the parent produces a failing render.
- Rasters the document re-reads with `rast(path)` must first be written from the annotated in-memory objects (`writeRaster`) so category legends and time attributes survive the round trip.
- Render with `quiet = FALSE` when it fails; the default swallows the real cause.

### Use the project's own inputs

When a function accepts both a shipped reference raster and one you built from a polygon or vector, prefer the shipped one — it is the path the project's examples and tests exercise. Details of the resolution mismatch this caused: `references/terra-verification-notes.md`.

## 5. Shiny / interactive verification

- **Confirm the harness can open a listening socket before building an HTTP smoke test.** If the process logs `createTcpServer: permission denied` / `Failed to create server`, that is the environment refusing to bind, NOT an app fault — the app may have loaded every package and initialized cleanly right up to the listen call. Read the log, then say the constraint out loud instead of reporting N module failures. A loop that waits on a port that can never open will happily emit a wall of false failures.
- **Test apps sequentially when they share a fixed port.** Launcher templates routinely pin one port for every module, so two modules can never run at once.
- Porting checklist (Windows-only launcher, `shell.exec`, hardcoded ports): `references/shiny-app-linux-port.md`.

## Reporting this class of work

- Lead with what was verified by real execution, with the actual output (row counts, area figures, HTTP status, rendered file paths) — not a description of the attempt.
- List reproduced upstream bugs separately from your own harness bugs, each with the literal error text.
- State explicitly what is blocked and why, and what was not attempted. Never let an unverified surface appear verified.
- Save the deliverable report to disk; push a summary plus the artifacts to Telegram (body = summary, full docs as attachments).

## Pitfalls

- **`options(timeout)` default 60s silently drops downloads** — the resulting `dependency X is not available` message points at the wrong package entirely.
- **Hand-ordered dependency lists cannot work** — a dependent installed before its dependency fails identically on every retry, which looks like a broken package instead of a broken order.
- **Base and archived packages break graph solvers** — see §3 for the two error strings that identify each.
- **`sum(df$failed) == 0` is not a passing suite** — errors live in a different column.
- **`install.packages` into a root-owned `site-library` fails even with working sudo** — use the user library instead of escalating.
- **GDAL/GEOS/PROJ headers must be present before the R packages that link them**, or terra/sf fail at configure with a message about a missing `gdal-config`.
- **A Shiny launcher that shells out to a hardcoded Windows binary path cannot start a single module on Linux** even though the app file itself parses fine.
