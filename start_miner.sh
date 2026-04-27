#!/bin/bash
set -euo pipefail

MINER_NAME=wolf_miner_5_clique
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINER_ARGS=("$@")

cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/../.venv"

ensure_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
    fi
}

start_miner() {
    ensure_venv
    source "$VENV_DIR/bin/activate"
    # pip install -e .

    if pm2 list | grep -q "$MINER_NAME"; then
        pm2 delete "$MINER_NAME" 2>/dev/null || true
    fi

    pm2 start "$VENV_DIR/bin/python" --name "$MINER_NAME" --interpreter none -- \
        -m CliqueAI.miner \
        "${MINER_ARGS[@]}"
}

usage() {
    cat <<EOF
Usage:
  ./start_miner.sh [miner args...]
EOF
}

case "${1:-start}" in
    -h | --help | help)
        usage
        ;;
    *)
        start_miner
        ;;
esac
