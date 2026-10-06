# Shared safety check for benchmark harnesses that launch llama-server.
# Refuse to run beside an existing server rather than terminating it by name.

refuse_if_running_llama_server() {
    if ! command -v pgrep >/dev/null 2>&1; then
        printf '%s\n' 'pgrep is required for the benchmark safety check.' >&2
        return 1
    fi
    if pgrep -x llama-server >/dev/null 2>&1; then
        printf '%s\n' \
            'An existing llama-server process is running. Stop it manually before benchmarking; this script will not terminate it.' \
            >&2
        return 1
    fi
}
