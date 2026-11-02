#!/usr/bin/env bash
# Render one HTML file to PNG with headless Chrome.
#
#   scripts/html_to_png.sh input.html output.png [width] [height]
#
# Fails loudly when the render produced an empty or placeholder file, which is how a broken
# infographic usually presents itself.

set -euo pipefail

IN="${1:?usage: html_to_png.sh input.html output.png [width] [height]}"
OUT="${2:?missing output path}"
W="${3:-1600}"
H="${4:-900}"

[ -f "$IN" ] || { echo "error: no such input: $IN" >&2; exit 1; }

CHROME="$(command -v google-chrome || command -v chromium || command -v chromium-browser || true)"
[ -n "$CHROME" ] || { echo "error: no headless Chrome/Chromium on PATH" >&2; exit 1; }

# Chrome resolves relative paths against its own cwd: always pass absolute file:// URLs.
ABS_IN="$(cd "$(dirname "$IN")" && pwd)/$(basename "$IN")"
OUT_DIR="$(dirname "$OUT")"
mkdir -p "$OUT_DIR"
ABS_OUT="$(cd "$OUT_DIR" && pwd)/$(basename "$OUT")"

"$CHROME" --headless --disable-gpu --no-sandbox --hide-scrollbars \
  --window-size="$W,$H" --force-device-scale-factor=1 \
  --screenshot="$ABS_OUT" "file://$ABS_IN" >/dev/null 2>&1 || true

SIZE=$(stat -c%s "$ABS_OUT" 2>/dev/null || echo 0)
if [ "$SIZE" -lt 5000 ]; then
  echo "error: render produced ${SIZE} bytes — treat as failure (blank or placeholder image)" >&2
  exit 1
fi

echo "wrote $ABS_OUT ($((SIZE / 1024)) KB, ${W}x${H})"
