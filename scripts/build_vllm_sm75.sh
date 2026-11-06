#!/usr/bin/env bash
# Reproducible SM75 / CUDA 13.0 build of vLLM 0.25.1 native extensions (no install, no venv).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/work/upstream/vllm-0.25.1"
export PYTHONPATH="$ROOT/work/buildenv/site${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=7.5 CMAKE_CUDA_ARCHITECTURES=75
export VLLM_TARGET_DEVICE=cuda MAX_JOBS=1 NVCC_THREADS=1
export SETUPTOOLS_SCM_PRETEND_VERSION=0.25.1   # source is a git-less tag tarball
export CCACHE_DISABLE=1
cd "$SRC"
{ echo "nvcc: $(nvcc --version | tail -2 | head -1)"; gcc --version | head -1; cmake --version | head -1; ninja --version; } \
  | tee "$ROOT/logs/build-toolchain.txt"
/usr/local/python3.13/bin/python3.13 setup.py bdist_wheel --dist-dir "$ROOT/work/build/vllm-wheel" --py-limited-api=cp38
