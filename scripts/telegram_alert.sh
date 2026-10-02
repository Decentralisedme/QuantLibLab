#!/usr/bin/env bash
# Sends a Telegram alert when quantlib-harness.service fails.
# Invoked by quantlib-harness-alert.service (systemd OnFailure=), which
# loads TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from /etc/quantlib-harness/telegram.env.
set -euo pipefail

if [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]]; then
    echo "telegram_alert.sh: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — skipping alert" >&2
    exit 1
fi

# quantlib-harness.service redirects its stdout/stderr straight to
# harness.log (StandardOutput=append:...), so nothing lands in the journal
# for this unit — tail the log file itself, not journalctl.
LOGTAIL=$(tail -n 20 /home/groku/QuantLibLab/data/harness/harness.log 2>/dev/null || true)
TEXT="⚠️ quantlib-harness.service FAILED on $(hostname) at $(date -u +%FT%TZ)

${LOGTAIL}"
TEXT="${TEXT:0:3500}"

curl -sS -m 15 -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
    --data-urlencode "text=${TEXT}" >/dev/null
