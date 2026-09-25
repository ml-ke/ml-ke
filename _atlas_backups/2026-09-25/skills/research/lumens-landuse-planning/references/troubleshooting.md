# Environment and dependency pitfalls (Ubuntu, R 4.6.1)

Consolidated from a full install/run/test pass. Each item cost real debugging time;
none of them are LUMENSR bugs.

## Disk and download

- R's default `download.file` timeout is **60 s**. On a slow uplink, large tarballs
  (data.table, stringi, igraph) fail silently; the failure then surfaces as dozens of
  "dependency X is not available" errors for unrelated packages.
  Fix: `options(timeout = 900)`.
- Prefer Posit Public Package Manager binaries for the exact distro:
  `repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/noble/latest")`.
  Packages fetched as `x86_64-pc-linux-gnu-ubuntu-24.04` install without compiling;
  anything that starts running `gcc`/`make` will cost tens of minutes on a throttled
  mobile-class CPU (igraph alone compiles hundreds of C files).
- `utils::install.packages()` with a hand-ordered vector installs in that order. If a
  dependent precedes its dependency, the dependent fails and is never retried.
  `pak::pkg_install()` solves order, retries and parallelism.
- `pak` may hang indefinitely on a stalled socket with no log progress. Wrap it in a
  loop with `timeout`, because the package cache makes each pass resume cheaply.
- `pak::pkg_install()` errors out on the whole request if a vector contains an
  archived package (`spatial.tools`) or a base/recommended package (`foreign`).
  Also avoid pinning packages whose conflicts it cannot reconcile.

## System libraries

| Package | Needs |
|---|---|
| terra, sf, rasterVis, lwgeom | libgdal-dev, libgeos-dev, libproj-dev, libudunits2-dev, libsqlite3-dev |
| magick | libmagick++-dev, imagemagick |
| fs | **libuv1-dev** (or `USE_BUNDLED_LIBUV=1`) |
| textshaping, ragg, systemfonts | libfontconfig1-dev, libfreetype6-dev, libharfbuzz-dev, libfribidi-dev |
| git2r / usethis | libgit2-dev |
| tidyterra, terra + PNG/TIFF | libpng-dev, libtiff5-dev, libjpeg-dev, libwebp-dev |

## Library location

R adds `~/R/x86_64-pc-linux-gnu-library/<major.minor>` to `.libPaths()` **only if the
path already exists**. When R comes from apt, the site library is
`/usr/local/lib/R/site-library` and is root-owned, so `install.packages()` fails with
"'lib' is not writable". Create the user directory and nothing needs sudo:

```r
user_lib <- file.path(Sys.getenv("HOME"), "R", "x86_64-pc-linux-gnu-library", "4.6")
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))
```

## sudo inside an agent session

Use `sudo -A` with `SUDO_ASKPASS` pointing at a helper that prints the stored password,
never `sudo -S` with the password on stdin. `-S` replaces the wrapped command's stdin,
so piping into sudo silently delivers nothing and the command appears to hang:

```bash
printf '%s\n' "$PW" | sudo -S -p '' tee file   # BROKEN: tee receives EOF
curl -s URL | sudo -S -p '' tee file           # BROKEN: curl output never reaches tee
SUDO_ASKPASS=/path/to/askpass sudo -A tee file # correct: stdin preserved
```

Also point the wrapper at the absolute `/usr/bin/sudo`: if the helper is installed on
`PATH` as a `sudo` shim, a bare `sudo` recurses into it forever.

## Verifying the GUI in a sandbox

`createTcpServer: permission denied` / `Error in initialize(...) : Failed to create
server` after a successful `Listening on http://...` means the environment blocks
listening sockets. Confirm by binding a trivial server (e.g. `python3 -m http.server`)
before blaming the application, and never report the app as broken on this basis.
