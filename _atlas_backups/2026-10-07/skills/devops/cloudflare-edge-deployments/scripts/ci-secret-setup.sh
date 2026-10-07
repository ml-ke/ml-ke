#!/usr/bin/env bash
# Store a CI secret without the empty-value trap, then prove the runner sees it.
#
# WHY THIS EXISTS
#   `gh secret set` stores an empty or whitespace-only value with exit status 0,
#   and bash's `read -s` keeps a trailing CR as data — so a token pasted from a GUI
#   clipboard (CRLF endings) can arrive as "\r": non-empty to the shell, empty in
#   reality. The result is a secret that `gh secret list` shows, that the workflow
#   reads as unset, and that surfaces as a deploy step silently skipped while the
#   job still concludes `success`.
#
# USAGE
#   ci-secret-setup.sh set CLOUDFLARE_API_TOKEN                    # hidden prompt
#   ci-secret-setup.sh set CLOUDFLARE_API_TOKEN=<value> [NAME=<value> ...]
#   ci-secret-setup.sh check docs.yml 'Publish to Cloudflare'
#
# ENV
#   MIN_SECRET_CHARS  plausibility floor for a stored value (default 20)
#   REPO              owner/name; defaults to the repo of the current clone
#   <NAME>            a value exported under the secret's own name is used as-is

set -euo pipefail

MIN_SECRET_CHARS="${MIN_SECRET_CHARS:-20}"
REPO="${REPO:-}"

ok()  { printf '  \033[32mok\033[0m   %s\n' "$*"; }
bad() { printf '  \033[31mFAIL\033[0m %s\n' "$*" >&2; }
say() { printf '\n%s\n' "$*"; }

usage() {
  sed -n '14,22p' "$0" | sed 's/^# \{0,1\}//'
}

# Strip every whitespace character: a CR-only paste normalises to the empty
# string instead of being stored as garbage.
normalise() { printf '%s' "$1" | tr -d '[:space:]'; }

gh_secret_set() {
  if [ -n "$REPO" ]; then gh secret set "$1" --repo "$REPO"; else gh secret set "$1"; fi
}

gh_workflow_run() {
  if [ -n "$REPO" ]; then gh workflow run "$1" --repo "$REPO"; else gh workflow run "$1"; fi
}

# Echo a usable value on stdout, chatter on stderr. Accepts the value from the
# environment under the secret's own name, or reads it hidden with up to three
# attempts, reporting the character count so a bad paste is obvious immediately.
read_value() {
  local name="$1" raw attempt=0 value=""
  value="$(normalise "${!name:-}")"
  [ -n "$value" ] && { printf '%s' "$value"; return 0; }
  if [ ! -t 0 ]; then
    bad "no value for $name and stdin is not a terminal — export $name=... or pass $name=<value>"
    return 1
  fi
  while [ "$attempt" -lt 3 ]; do
    attempt=$((attempt + 1))
    printf '  paste %s (attempt %s/3, hidden): ' "$name" "$attempt" >&2
    read -rs raw
    echo >&2
    value="$(normalise "$raw")"
    unset raw
    if [ -z "$value" ]; then
      bad "nothing usable read — paste the value itself, not a blank line"
    elif [ "${#value}" -lt "$MIN_SECRET_CHARS" ]; then
      bad "only ${#value} characters read; a token is usually longer — copy it again"
      value=""
    else
      ok "read ${#value} characters"
      printf '%s' "$value"
      return 0
    fi
  done
  return 1
}

cmd_set() {
  [ "$#" -ge 1 ] || { bad "usage: $0 set NAME[=VALUE] ..."; exit 2; }
  say "Storing secrets${REPO:+ in $REPO}"
  local pair name value
  for pair in "$@"; do
    name="${pair%%=*}"
    value=""
    if [ "$pair" = "$name" ]; then
      value="$(read_value "$name")" || { bad "no usable value for $name"; exit 1; }
    else
      value="$(normalise "${pair#*=}")"
      [ -n "$value" ] || { bad "$name: empty value"; exit 1; }
      [ "${#value}" -ge "$MIN_SECRET_CHARS" ] || { bad "$name: ${#value} characters is implausibly short"; exit 1; }
    fi
    printf '%s' "$value" | gh_secret_set "$name"
    ok "$name: ${#value} characters stored"
    unset value
  done
  echo "  gh stores whatever it is handed; only a run proves the value is right."
}

cmd_check() {
  local workflow="${1:-}" step_pat="${2:-}" id concl
  [ -n "$workflow" ] || { bad "usage: $0 check WORKFLOW [STEP_PATTERN]"; exit 2; }
  say "Triggering $workflow"
  gh_workflow_run "$workflow"
  sleep 8
  id="$(gh run list --workflow="$workflow" --limit 1 --json databaseId -q '.[0].databaseId')"
  [ -n "$id" ] || { bad "no run found for $workflow"; exit 1; }
  ok "run $id"
  gh run watch "$id" --exit-status >/dev/null 2>&1 || true

  say "Step conclusions (the job's verdict is not evidence)"
  gh run view "$id" --json jobs -q '.jobs[] | .steps[] | "  \(.name) -> \(.conclusion)"'

  say "Credential lengths the runner saw (values are never printed)"
  local lengths
  lengths="$(gh run view "$id" --log 2>/dev/null | grep -o 'secret lengths:.*' | tail -1 || true)"
  if [ -n "$lengths" ]; then echo "  $lengths"; else
    echo "  (no length line — add: echo \"secret lengths: token=\${#TOKEN}\" to the guard step)"
  fi

  [ -n "$step_pat" ] || return 0
  concl="$(gh run view "$id" --json jobs -q ".jobs[].steps[] | select(.name|test(\"$step_pat\")) | .conclusion" | tail -1)"
  case "$concl" in
    success) ok "$step_pat: ran" ;;
    skipped) bad "$step_pat: SKIPPED — the credential it guards is absent or empty"; return 1 ;;
    "")      bad "$step_pat: no step matched that pattern"; return 1 ;;
    *)       bad "$step_pat: $concl"; return 1 ;;
  esac
}

case "${1:-}" in
  set)   shift; cmd_set "$@" ;;
  check) shift; cmd_check "$@" ;;
  *)     usage; exit 2 ;;
esac
