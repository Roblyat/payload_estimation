#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/up_stack.sh up [log] [service]
#   ./scripts/up_stack.sh down
#   ./scripts/up_stack.sh restart [log] [service]
#   ./scripts/up_stack.sh status
#   ./scripts/up_stack.sh log [service]
#
# Examples:
#   ./scripts/up_stack.sh up
#   ./scripts/up_stack.sh up log
#   ./scripts/up_stack.sh up log delan
#   ./scripts/up_stack.sh log lstm
#   ./scripts/up_stack.sh log all

cmd="${1:-up}"
opt1="${2:-}"   # "log" or service when cmd=log
opt2="${3:-}"   # service when cmd=up/restart with log

ALL_SERVICES=(preprocess delan_jax lstm evaluation runner_gui)

resolve_services() {
  local sel="${1:-all}"

  case "$sel" in
    ""|all)
      echo "${ALL_SERVICES[@]}"
      ;;
    preprocess|pre)
      echo "preprocess"
      ;;
    dlan_jax|delan_jax)
      echo "delan_jax"
      ;;
    dlan_torch|delan_torch)
      echo "delan_jax"
      ;;
    lstm)
      echo "lstm"
      ;;
    evaluation|eval)
      echo "evaluation"
      ;;
    gui|runner_gui)
      echo "runner_gui"
      ;;
    *)
      echo "Unknown service selector: $sel" >&2
      echo "Use one of: all | preprocess | delan | lstm | evaluation | gui" >&2
      exit 1
      ;;
  esac
}

stream_logs() {
  local sel="${1:-all}"
  local services
  services=($(resolve_services "$sel"))
  echo "Streaming logs for: ${services[*]}"
  echo "(Ctrl+C stops streaming; containers keep running)..."
  docker compose logs -f --tail=200 "${services[@]}"
}

start_stack() {
  docker compose up -d preprocess
  docker compose up -d delan_jax
  docker compose up -d delan_torch
  docker compose up -d lstm
  docker compose up -d evaluation
  docker compose up -d runner_gui
  echo "Stack is up: preprocess -> delan_jax -> delan_torch -> lstm -> evaluation -> runner_gui"
  echo "GUI: http://localhost:8501"
}

case "$cmd" in
  up)
    start_stack
    if [[ "${opt1:-}" == "log" ]]; then
      echo
      stream_logs "${opt2:-all}"
    fi
    ;;
  down)
    docker compose down
    ;;
  restart)
    docker compose down
    start_stack
    if [[ "${opt1:-}" == "log" ]]; then
      echo
      stream_logs "${opt2:-all}"
    fi
    ;;
  status)
    docker compose ps
    ;;
  log)
    # ./scripts/up_stack.sh log [service]
    stream_logs "${opt1:-all}"
    ;;
  *)
    echo "Unknown command: $cmd"
    echo "Use: up [log [service]] | down | restart [log [service]] | status | log [service]"
    exit 1
    ;;
esac