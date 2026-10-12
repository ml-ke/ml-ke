# Porting a Windows-authored Shiny app to Linux

Research R projects are frequently developed on Windows and shipped with
Windows-only assumptions baked into the launcher. These are the four that stop
the app dead, in the order they bite.

## 1. The launcher shells out to `rscript.exe`

Symptom: the top-level launcher app starts and renders fine, but pressing any
button does nothing on Linux.

```r
# upstream
command <- paste0("rscript.exe ", r_script)
status  <- system(command)   # rscript.exe does not exist -> silent failure
```

Fix by selecting the binary at runtime rather than hardcoding:

```r
R_BIN <- if (.Platform$OS.type == "windows") "rscript.exe" else "Rscript"
cmd <- sprintf("%s %s > %s 2>&1", R_BIN, shQuote(file.path(APP_HOME, r_script)), shQuote(log_file))
status <- system(cmd, intern = FALSE)
```

**Capture stderr to a per-module log file and surface the path in the UI.** The
upstream pattern throws the output away, so a module that dies on startup looks
identical to one that is still loading — the launcher just appears to hang. With
the log, a failed module reports its own error and its log path.

Note `system()` blocks the launcher process for as long as the module runs; that
is the upstream design ("Running X" notification, then "has closed"). Keep it.

## 2. `shell.exec()` is Windows-only

Typically on "open output folder" and "open report" buttons, across many files:

```bash
grep -rln "shell\.exec" --include='*.R' .
```

It does not exist on Linux, so the button errors when clicked. Guard it, and keep
the useful part (telling the user the path):

```r
if (.Platform$OS.type == "windows") shell.exec(path) else message("Path (open manually): ", path)
```

Back up each file before rewriting it (`cp "$f" "$f.orig"`) so the upstream
original stays diffable.

## 3. Every module pins the same port

`runApp(..., port = 875)` repeated across all the `call*.R` launch scripts means two
modules can never run at once, and the port may already be in use. Make it
overridable and stop forcing a browser:

```r
shiny::runApp(
  '<module>/rscript/',
  launch.browser = tolower(Sys.getenv("LUMENS_OPEN_BROWSER", "false")) %in% c("true","1","yes"),
  port           = as.integer(Sys.getenv("LUMENS_PORT", "875"))
)
```

`launch.browser = TRUE` (the common upstream default) is actively harmful over SSH
or on a headless box. Default it to false and make it opt-in.

## 4. Some modules have no `app.R`

A module may use the `ui.R` + `server.R` (+ optional `global.R`) layout instead.
Shiny auto-detects either, so `runApp('<dir>/')` works unchanged — just do not
grep for `app.R` alone when enumerating modules.

## Other facts worth checking before blaming the port

- Modules commonly `source('../../helper.R')` — relative paths hold only when the
  launcher runs from the repo root, which is how `runApp('<module>/rscript/')` behaves.
- Module start-up blocks often call a `check_and_install_packages()` helper that
  installs missing packages at launch. That is why a module can appear to hang on
  first run: it is installing. Pre-install the full dependency list instead.
- Cross-module numbering can be inconsistent upstream (a `call12.R` may point at a
  differently-numbered module directory). Read each launcher script rather than
  assuming the mapping.
- Verify the rewritten launcher parses before running it:
  `Rscript -e 'invisible(parse("app.R")); cat("parses OK\\n")'`

## Smoke-testing the modules

Boot each module in turn and request the page, but only after confirming the
harness can actually bind (see SKILL.md §5 — a bind-denied environment produces a
wall of false failures). Record one row per module (module dir, HTTP code, log
path) so a partial run still yields evidence, and kill each app before starting
the next since they share a port.
