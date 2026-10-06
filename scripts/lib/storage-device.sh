# Shared helpers for benchmark scripts that sample block-device counters.
# Source this after setting MODEL. Override FTL_DISKSTATS_DEVICE if the model's
# filesystem is backed by a device-mapper stack and physical-device counters are
# preferred over the mapped device's counters.

storage_device_from_model() {
    local source device
    source=$(findmnt --noheadings --output SOURCE --target "$MODEL" 2>/dev/null | awk 'NR == 1 { print $1 }')
    source=${source%%\[*}
    [[ "$source" == /dev/* ]] || return 1

    device=$(lsblk --noheadings --output PKNAME -- "$source" 2>/dev/null | awk 'NF { print; exit }')
    if [[ -z "$device" ]]; then
        device=$(lsblk --noheadings --nodeps --output KNAME -- "$source" 2>/dev/null | awk 'NF { print; exit }')
    fi
    [[ -n "$device" ]] && printf '%s\n' "${device##*/}"
}

FTL_DISKSTATS_DEVICE="${FTL_DISKSTATS_DEVICE:-$(storage_device_from_model || true)}"
FTL_DISKSTATS_DEVICE="${FTL_DISKSTATS_DEVICE##*/}"

diskstats_sectors_read() {
    [[ -n "$FTL_DISKSTATS_DEVICE" ]] || { printf '0\n'; return; }
    awk -v device="$FTL_DISKSTATS_DEVICE" \
        '$3 == device { print $6; found = 1; exit } END { if (!found) print 0 }' \
        /proc/diskstats
}

storage_read_ahead_kb() {
    [[ -n "$FTL_DISKSTATS_DEVICE" ]] || { printf 'unknown\n'; return; }
    cat "/sys/class/block/$FTL_DISKSTATS_DEVICE/queue/read_ahead_kb" 2>/dev/null || printf 'unknown\n'
}
