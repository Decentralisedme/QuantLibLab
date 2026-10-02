#!/usr/bin/env bash
# Rebuilds site/curves.json (and anything else build_curves_site.py writes
# under site/) from whatever golden-snapshot / CSV-store data is on disk
# right now, and pushes it to origin/main if anything changed — Netlify's
# GitHub integration picks up the push and redeploys automatically.
#
# Invoked via OnSuccess= on quantlib-golden-snapshot.service,
# quantlib-daily-data.service and quantlib-harness.service: any of the
# three can leave new data behind. Re-running this after one that
# didn't is a cheap no-op (the git diff --cached --quiet check below).
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
