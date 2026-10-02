#!/usr/bin/env bash
# Rebuilds site/curves.json (and anything else build_curves_site.py writes
# under site/) from whatever golden-snapshot / CSV-store data is on disk
# right now, and pushes it to origin/main if anything changed — Netlify's
# GitHub integration picks up the push and redeploys automatically.
#
# Invoked via OnSuccess= on quantlib-harness.service — the last of the
# three morning jobs (golden-snapshot 06:00, daily-data 06:20, harness
# 07:00) — so it fires once per morning, after all three have had their
# chance to land new data. The git diff --cached --quiet check below
# still makes it a no-op if nothing actually changed.
set -euo pipefail
cd /home/groku/QuantLibLab

# Pick up anything pushed from elsewhere since our last run, fast-forward
# only. If that's not possible (true divergence), stop here and let the
# OnFailure alert fire rather than force-pushing or rebasing blind.
git fetch origin main
git merge --ff-only origin/main

.venv/bin/python scripts/build_curves_site.py

git add site/
if git diff --cached --quiet; then
    echo "publish_site.sh: no changes under site/ — nothing to commit"
    exit 0
fi

git commit -m "site: refresh curves data ($(date -u +%FT%TZ))"
git push origin main
