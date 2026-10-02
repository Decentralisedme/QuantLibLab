#!/usr/bin/env bash
# Sends a Telegram alert when a quantlib-* systemd service fails.
# Invoked by that service's matching *-alert.service (systemd OnFailure=),
# which sets SERVICE_LABEL / LOG_FILE via Environment= and loads
# TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from /etc/quantlib-harness/telegram.env.
set -euo pipefail

SERVICE_LABEL="${SERVICE_LABEL:-quantlib-harness.service}"
LOG_FILE="${LOG_FILE:-/home/groku/QuantLibLab/data/harness/harness.log}"

if [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]]; then
    echo "telegram_alert.sh: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — skipping alert" >&2
    exit 1
fi

# The failed service redirects its stdout/stderr straight to its own log
# file (StandardOutput=append:...), so nothing lands in the journal for
# that unit — tail the log file itself, not journalctl.
LOGTAIL=$(tail -n 20 "$LOG_FILE" 2>/dev/null || true)
TEXT="⚠️ ${SERVICE_LABEL} FAILED on $(hostname) at $(date -u +%FT%TZ)

${LOGTAIL}"
TEXT="${TEXT:0:3500}"

curl -sS -m 15 -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
    --data-urlencode "text=${TEXT}" >/dev/null
