#!/usr/bin/env bash
# One-time install of the quantlib-harness systemd service + timer.
# Run with sudo from anywhere: sudo bash deploy/systemd/install.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "run with sudo: sudo bash deploy/systemd/install.sh" >&2
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIT_DIR="$REPO_DIR/deploy/systemd"

UNITS=(
    quantlib-harness.service
    quantlib-harness.timer
    quantlib-harness-alert.service
    quantlib-golden-snapshot.service
    quantlib-golden-snapshot.timer
    quantlib-golden-snapshot-alert.service
    quantlib-daily-data.service
    quantlib-daily-data.timer
    quantlib-daily-data-alert.service
    quantlib-publish-site.service
    quantlib-publish-site-alert.service
)
for unit in "${UNITS[@]}"; do
    install -m 644 "$UNIT_DIR/$unit" "/etc/systemd/system/$unit"
done

mkdir -p "$REPO_DIR/data/logs"
chown groku:groku "$REPO_DIR/data/logs"

mkdir -p /etc/quantlib-harness
if [[ ! -f /etc/quantlib-harness/telegram.env ]]; then
    cat > /etc/quantlib-harness/telegram.env <<'EOF'
# Fill in and save. Required for failure alerts; the timer runs fine without
# this file present, it just won't be able to notify on failure.
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
EOF
    chown groku:groku /etc/quantlib-harness/telegram.env
    chmod 600 /etc/quantlib-harness/telegram.env
    echo "created /etc/quantlib-harness/telegram.env — edit it with your bot token + chat id"
else
    echo "/etc/quantlib-harness/telegram.env already exists, leaving it alone"
fi

if [[ ! -f /etc/quantlib-harness/fred.env ]]; then
    cat > /etc/quantlib-harness/fred.env <<'EOF'
# Fill in and save. Required: quantlib-daily-data and quantlib-golden-snapshot
# load this file and fail to start without it. Free key at
# https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=
EOF
    chown groku:groku /etc/quantlib-harness/fred.env
    chmod 600 /etc/quantlib-harness/fred.env
    echo "created /etc/quantlib-harness/fred.env — edit it with your FRED API key"
else
    echo "/etc/quantlib-harness/fred.env already exists, leaving it alone"
fi

systemctl daemon-reload
systemctl enable --now quantlib-golden-snapshot.timer
systemctl enable --now quantlib-daily-data.timer
systemctl enable --now quantlib-harness.timer
# quantlib-publish-site.service has no timer of its own — it's only ever
# triggered via OnSuccess= from the three services above, so there's
# nothing to enable for it beyond the daemon-reload.

echo
echo "installed. status:"
systemctl list-timers quantlib-golden-snapshot.timer quantlib-daily-data.timer quantlib-harness.timer --no-pager
