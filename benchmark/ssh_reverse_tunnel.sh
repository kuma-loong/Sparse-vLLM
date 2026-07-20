#!/usr/bin/env bash
set -euo pipefail

SSH_DESTINATION="${SSH_DESTINATION:?SSH_DESTINATION is required}"
SSH_REVERSE_FORWARD="${SSH_REVERSE_FORWARD:?SSH_REVERSE_FORWARD is required}"
SSH_JUMP_HOST="${SSH_JUMP_HOST:-}"
SSH_PORT="${SSH_PORT:-22}"
SSH_BIN="${SSH_BIN:-ssh}"
SSH_LOG="${SSH_LOG:-/dev/null}"
SSH_STATUS_FILE="${SSH_STATUS_FILE:-}"
SSH_MAX_RECONNECTS="${SSH_MAX_RECONNECTS:-100}"
SSH_RECONNECT_DELAY_S="${SSH_RECONNECT_DELAY_S:-5}"
SSH_CONNECT_TIMEOUT_S="${SSH_CONNECT_TIMEOUT_S:-15}"

for value in SSH_PORT SSH_MAX_RECONNECTS SSH_RECONNECT_DELAY_S SSH_CONNECT_TIMEOUT_S; do
  if [[ ! "${!value}" =~ ^[0-9]+$ ]]; then
    printf '%s must be a non-negative integer, got %s\n' "$value" "${!value}" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "$SSH_LOG")"
if [[ -n "$SSH_STATUS_FILE" ]]; then
  mkdir -p "$(dirname "$SSH_STATUS_FILE")"
  if [[ ! -e "$SSH_STATUS_FILE" ]]; then
    printf 'time\tstate\tattempt\texit_code\n' >"$SSH_STATUS_FILE"
  fi
fi

record_status() {
  if [[ -n "$SSH_STATUS_FILE" ]]; then
    printf '%s\t%s\t%s\t%s\n' \
      "$(date --iso-8601=seconds)" "$1" "$2" "${3:-}" >>"$SSH_STATUS_FILE"
  fi
}

stopping=0
child_pid=
stop_tunnel() {
  stopping=1
  if [[ -n "$child_pid" ]]; then
    kill "$child_pid" 2>/dev/null || true
  fi
}
trap stop_tunnel INT TERM
trap 'if [[ -n "$child_pid" ]]; then kill "$child_pid" 2>/dev/null || true; fi' EXIT

ssh_args=(
  -N
  -T
  -p "$SSH_PORT"
  -o ExitOnForwardFailure=yes
  -o "ConnectTimeout=${SSH_CONNECT_TIMEOUT_S}"
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=2
  -o TCPKeepAlive=yes
  -R "$SSH_REVERSE_FORWARD"
)
if [[ -n "$SSH_JUMP_HOST" ]]; then
  ssh_args+=(-J "$SSH_JUMP_HOST")
fi
ssh_args+=("$SSH_DESTINATION")

attempt=0
while (( attempt <= SSH_MAX_RECONNECTS )); do
  attempt=$((attempt + 1))
  record_status connecting "$attempt"
  "$SSH_BIN" "${ssh_args[@]}" >>"$SSH_LOG" 2>&1 &
  child_pid=$!
  set +e
  wait "$child_pid"
  exit_code=$?
  set -e
  child_pid=
  if (( stopping )); then
    record_status stopped "$attempt" "$exit_code"
    exit 0
  fi
  record_status disconnected "$attempt" "$exit_code"
  if (( attempt > SSH_MAX_RECONNECTS )); then
    record_status failed "$attempt" "$exit_code"
    if (( exit_code == 0 )); then
      exit 1
    fi
    exit "$exit_code"
  fi
  sleep "$SSH_RECONNECT_DELAY_S"
done
