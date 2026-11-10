#!/usr/bin/env bash
# Weekly Antigravity CLI self-update (agy update). Idempotent: prints either
# "already on the latest version" or the update logs. Absolute binary path so
# the cron environment's PATH doesn't matter.
exec /home/pro-g/.local/bin/agy update 2>&1
