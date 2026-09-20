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

install -m 644 "$UNIT_DIR/quantlib-harness.service" /etc/systemd/system/quantlib-harness.service
install -m 644 "$UNIT_DIR/quantlib-harness.timer" /etc/systemd/system/quantlib-harness.timer
install -m 644 "$UNIT_DIR/quantlib-harness-alert.service" /etc/systemd/system/quantlib-harness-alert.service

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

systemctl daemon-reload
systemctl enable --now quantlib-harness.timer

echo
echo "installed. status:"
systemctl status quantlib-harness.timer --no-pager
systemctl list-timers quantlib-harness.timer --no-pager
