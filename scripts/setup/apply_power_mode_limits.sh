#!/usr/bin/env bash
# Apply the clock limits of a Jetson power mode directly, without nvpmodel.
#
#   sudo -v && ./scripts/setup/apply_power_mode_limits.sh          # current mode (nvpmodel -q)
#   sudo -v && POWER_MODE_ID=0 ./scripts/setup/apply_power_mode_limits.sh
#   DRY_RUN=1 ./scripts/setup/apply_power_mode_limits.sh            # only print
#
# Reads the CPU, GPU and memory (EMC) limits of the mode and their control
# files from /etc/nvpmodel.conf, writes them, and checks them. Use it when
# `nvpmodel -q` reports a mode whose limits are not in effect (e.g. the boot
# service fails with "Reboot required for changing to this power mode").
# The limits last until the next reboot.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/prima_common.sh"

echo "[INFO] power mode limits from ${NVPMODEL_CONF:-/etc/nvpmodel.conf}, mode ${POWER_MODE_ID:-$(nvpmodel -q 2>/dev/null | sed -n 2p)}:"
while read -r name kind value path alt; do printf "  %-12s %-9s %-11s -> %s\n" "$name" "$kind" "$value" "$(_prima_limit_path "$path" "$alt" || echo "missing: $path")"; done < <(prima_mode_limits)
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0
sudo -n true 2>/dev/null || { echo "[ERROR] run 'sudo -v' first" >&2; exit 1; }
prima_apply_mode_limits || true
echo "[INFO] check:"
if prima_check_mode_limits | sed 's/^/  /'; then
  echo "[OK] all maximum limits are within the power mode"
else
  echo "[ERROR] some limits are still above the power mode" >&2; exit 1
fi
