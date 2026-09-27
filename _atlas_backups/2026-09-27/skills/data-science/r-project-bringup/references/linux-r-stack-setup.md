# Linux R stack bring-up: concrete recipe and failure modes

## Why ordering matters

Everything below is cheap except diagnosing a cascade. The rule that prevents the
most wasted time: **set `options(timeout)` and use a real dependency resolver
before the first install attempt.** Without them a slow link produces dozens of
misleading "package not available" errors.

## Layer 1: system (apt) — see `scripts/install-r-system-deps.sh`

| Because of | apt package |
|---|---|
| terra, sf | `libgdal-dev gdal-bin libgeos-dev libproj-dev proj-data proj-bin` |
| magick | `libmagick++-dev imagemagick` |
| fs (libuv bundled but not always used) | `libuv1-dev` |
| units / time handling in terra+sf | `libudunits2-dev` |
| systemfonts, ragg, textshaping (plot text) | `libfontconfig1-dev libfreetype6-dev libharfbuzz-dev libfribidi-dev` |
| rmarkdown / Quarto books | `pandoc` + the Quarto CLI |
| git-backed installs (`remotes`, `pak`) | `libgit2-dev libcurl4-openssl-dev libssl-dev` |

- `pandoc-citeproc` is gone on noble-class releases (merged into pandoc ≥ 2.19).
  A single unavailable name aborts the whole `apt-get install -y` invocation.
- `fs` without `libuv1-dev` fails at configure with `uv.h: No such file or
  directory`; textshaping, sass, bslib, shiny, rmarkdown and pkgload all fall
  over behind it. Easy to mistake for a package-repo problem.
- The Quarto CLI ships as a `.deb` from the quarto-cli GitHub releases; the R
  `quarto` package does not bundle it, and render fails with `System command
  'quarto' failed` when the binary is absent.

## Layer 2: R packages — see `scripts/install-r-packages.R`

### User library instead of sudo

The apt R installs `site-library` under `/usr/local/lib/R` (root-owned), so
`install.packages()` stops with `'lib = "/usr/local/lib/R/site-library"' is not
writable`. R's built-in default `R_LIBS_USER` is
`~/R/x86_64-pc-linux-gnu-library/<major.minor>`, and it is added to `.libPaths()`
**only when the directory already exists**. Creating it needs no privileges and no
`.Renviron` edit, and every later R process (including Shiny sub-apps) installs
into the same place.

### The 60-second download timeout

R's `download.file` default is `options(timeout = 60)`. On a slow link the
download is cut off, and the message that reaches the user is not about the
download at all:

```
Warning: download of package 'stringi' failed
Warning: download of package 'data.table' failed
Warning: download of package 'igraph' failed
ERROR: dependency 'stringi' is not available for package 'stringr'
ERROR: dependencies 'data.table', 'stringi' are not available for package 'textshape'
ERROR: dependency 'data.table' is not available for package 'ModelMetrics'
... installation of N packages failed
```

Three failed downloads produced 40+ reported failures. Raise the timeout first
(`options(timeout = 900)`), then retry.

### Binary vs source packages

Requests to Posit Package Manager's explicitly-distro'd path are served as
precompiled binaries — pak labels them `x86_64-pc-linux-gnu-ubuntu-24.04`:

```r
options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/noble/latest"))
```

Without it, packages like igraph and sf compile from source: hundreds of C/C++
translation units, tens of minutes on a throttling laptop, and more chances for a
configure failure. Keep `Ncpus` at 2-4 on a small box — download parallelism on a
thin link is itself a cause of timeouts.

### Ordering: a resolver, not a hand-written list

A hand-ordered `install.packages()` list fails whenever a dependent precedes its
dependency, and each retry re-fails identically:

```
ERROR: dependencies 'htmlwidgets', 'reactable', 'sass' are not available for package 'gt'
ERROR: dependencies 'data.tree', 'htmlwidgets' are not available for package 'networkD3'
ERROR: dependency 'bslib' is not available for package 'shiny'
```

`pak::pkg_install(pkgs, ask = FALSE, upgrade = FALSE, dependencies = TRUE)` walks
the graph, installs in topological order and retries. Two inputs it cannot solve:
base/recommended packages (`foreign` → bare `dependency conflict`) and archived
CRAN packages (`spatial.tools` → `Can't find package called spatial.tools`).
Remove both from the list; fetch an archived package from
`https://cran.r-project.org/src/contrib/Archive/<pkg>/` only if something loads it.

### Stalled downloads

A resolver can sit at ~0% CPU with the log unchanged for 10+ minutes on a hung
socket. Wrap it in a loop with a per-pass `timeout`, since pak caches to
`~/.cache/R/pkgcache` and a re-run resumes rather than restarting:

```bash
for i in $(seq 1 8); do
  timeout 1500 Rscript install-r-packages.R "${pkgs[@]}" >> /tmp/pak.log 2>&1
  grep -q ALL_DEPS_OK /tmp/pak.log && break
done
```

## Layer 3: verify (never trust the install log)

```r
res  <- testthat::test_dir("tests/testthat", reporter = "summary", stop_on_failure = FALSE)
df   <- as.data.frame(res)
n_fail <- sum(df$failed); n_err <- sum(df$error)   # BOTH columns matter
print(df[, c("file", "test", "nb", "failed", "error")])
```

`stop_on_failure = FALSE` plus summing only `failed` yields a PASS over a suite
that threw errors — the per-file table is the only honest summary. Then grep the
saved log for `ERROR`, `non-zero exit status`, and `installation of package .*
failed`.
