#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_ARGS=()

if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  TORCH_INDEX_ARGS+=(--index-url "${TORCH_INDEX_URL}")
fi

usage() {
  cat <<'USAGE'
Usage:
  ./train.sh setup
  ./train.sh ssl [train_ssl.py args...]
  ./train.sh classification [train_classification.py args...]
  ./train.sh segmentation [train_segmentation.py args...]
  ./train.sh shell

Environment:
  VENV_DIR       Virtualenv path. Default: ./.venv
  PYTHON_BIN     Python executable for creating the venv. Default: python3
  TORCH_INDEX_URL Optional PyTorch index URL, useful for CUDA-specific wheels.
  SKIP_INSTALL   Set to 1 to skip dependency installation checks.

Examples:
  ./train.sh setup
  ./train.sh ssl --amp
  ./train.sh classification --source pathmnist_128 --freeze-encoder
  ./train.sh segmentation --batch-size 4
USAGE
}

create_venv() {
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
}

activate_venv() {
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
}

install_dependencies() {
  if [[ "${SKIP_INSTALL:-0}" == "1" ]]; then
    return
  fi

  python -m pip install --upgrade pip wheel setuptools
  python -m pip install matplotlib numpy pillow pyarrow tensorboard tqdm

  if ! python - <<'PY'
import importlib.util

modules = [
    "torch",
    "torchvision",
]

missing = [name for name in modules if importlib.util.find_spec(name) is None]
raise SystemExit(1 if missing else 0)
PY
  then
    python -m pip install "${TORCH_INDEX_ARGS[@]}" \
      torch \
      torchvision
  fi
}

setup_env() {
  cd "${PROJECT_DIR}"
  create_venv
  activate_venv
  install_dependencies
}

run_python() {
  local script_name="$1"
  shift

  setup_env
  python "${PROJECT_DIR}/${script_name}" "$@"
}

command="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${command}" in
  setup)
    setup_env
    echo "venv ready: ${VENV_DIR}"
    echo "activate with: source ${VENV_DIR}/bin/activate"
    ;;
  ssl|pretrain)
    run_python "train_ssl.py" "$@"
    ;;
  classification|classify|cls)
    run_python "train_classification.py" "$@"
    ;;
  segmentation|segment|seg)
    run_python "train_segmentation.py" "$@"
    ;;
  shell)
    setup_env
    exec "${SHELL:-/bin/bash}"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: ${command}" >&2
    usage >&2
    exit 2
    ;;
esac
