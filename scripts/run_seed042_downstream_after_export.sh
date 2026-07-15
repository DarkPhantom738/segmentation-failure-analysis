#!/usr/bin/env bash
# Back-compat wrapper → scripts/run_seed_downstream.sh 42
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT/scripts/run_seed_downstream.sh" 42 "$@"
