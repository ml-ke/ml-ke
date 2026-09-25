# Install R packages into a user library using precompiled Posit Package Manager
# binaries, with the full dependency graph resolved by pak.
#
#   Rscript install-r-packages.R pkg1 pkg2 pkg3 ...
#
# Or build the list from a project's real imports:
#   grep -rhoE "^\s*(library|require)\(['\"]?[A-Za-z][A-Za-z0-9._]*" --include='*.R' . \
#     | sed "s/.*(//;s/['\"]//" | sort -u > pkgs.txt
#   Rscript install-r-packages.R $(cat pkgs.txt | tr '\n' ' ')
#
# Run it under a timeout+retry loop when the link is slow; pak caches downloads in
# ~/.cache/R/pkgcache so a re-run resumes cheaply instead of starting over.

args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) {
  stop("usage: Rscript install-r-packages.R <pkg> [pkg ...]", call. = FALSE)
}
pkgs <- unique(args)

# --- distro path for the binary repo; adjust for your release ---
DISTRO <- Sys.getenv("R_P3M_DISTRO", "noble")   # jammy | noble | bookworm | ...
options(
  repos = c(CRAN = sprintf("https://packagemanager.posit.co/cran/__linux__/%s/latest", DISTRO)),
  # Default is 60s, which SILENTLY truncates downloads on a slow link and then
  # reports "dependency X is not available" for every dependent package.
  timeout = 900L,
  Ncpus = max(1L, parallel::detectCores() - 1L),
  warn = 1
)

# --- user library: R only picks this up if the directory already exists, and
# using it sidesteps the root-owned /usr/local/lib/R/site-library permission wall ---
user_lib <- file.path(
  Sys.getenv("HOME"), "R", "x86_64-pc-linux-gnu-library",
  paste(R.version$major, strsplit(R.version$minor, ".", fixed = TRUE)[[1]][1], sep = ".")
)
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))

cat("R:", R.version.string, "\nlib:", user_lib, "\nwritable:", file.access(user_lib, 2L) == 0L, "\n")

# --- drop packages that make solvers refuse instead of resolving ---
#   * base/recommended packages ship with R; asked to install them into another
#     lib, the solver reports a bare "dependency conflict"
#   * archived CRAN packages report "Can't find package called <x>"
base_pkgs <- rownames(installed.packages(priority = c("base", "recommended")))
skip <- pkgs %in% base_pkgs
if (any(skip)) {
  cat("skipping base/recommended (already present):", paste(pkgs[skip], collapse = ", "), "\n")
  pkgs <- pkgs[!skip]
}

need <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (!length(need)) { cat("all requested packages already present\nALL_DEPS_OK\n"); quit(status = 0) }

if (!requireNamespace("pak", quietly = TRUE)) install.packages("pak")

# pak resolves install ORDER and retries; a hand-ordered list cannot do this and
# fails identically forever when a dependent precedes its dependency.
ok <- tryCatch({
  pak::pkg_install(need, ask = FALSE, upgrade = FALSE, dependencies = TRUE)
  TRUE
}, error = function(e) { cat("pak error:", conditionMessage(e), "\n"); FALSE })

still <- need[!vapply(need, requireNamespace, logical(1), quietly = TRUE)]
cat("\n=== RESULT ===\nstill missing:",
    if (length(still)) paste(still, collapse = ", ") else "NONE", "\n")
cat("installed packages:", length(list.files(user_lib)), "\n")
cat(if (length(still) == 0) "ALL_DEPS_OK\n" else "DEPS_PARTIAL\n")
quit(status = if (length(still) == 0) 0 else 1)
