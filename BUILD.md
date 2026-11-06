# Build / reproduce the release candidate
Inputs: `work/downloads/vllm-v0.25.1.tar.gz` (sha256 `6e41ae18…e5a0`, commit `752a3a50…`), `stage.zip` (sha256 `7ea5e7a7…4798`), the existing local build `work/build/vllm-wheel/…whl` (NVCC 13.0.48, SM75).
```
python3.13 scripts/audit/audit_stage_zip.py stage.zip work/upstream/vllm-0.25.1 work/rc-stage-extract evidence-rc
python3.13 scripts/release/build_t4_tree.py work/rc-stage-extract/stage/vllm <extracted original wheel> work/rc-tree patches
python3.13 scripts/release/build_t4_wheel.py work/rc-tree work/build/vllm-wheel/<orig>.whl dist-rc/wheels
python3.13 -m build --wheel --no-isolation -o dist-rc/wheels .
python3.13 scripts/release/make_binary_evidence.py … ; python3.13 scripts/release/assemble_rc.py
```
`scripts/build_vllm_sm75.sh` is the original native build (it still compiles FlashAttention targets; a clean no-FA native rebuild needs a `setup.py` patch that has not been written/tested).
Toolchain: CPython 3.13.15, torch 2.11.0+cu130, NVCC 13.0.48 (Kaggle: 13.0.88 — a difference), GCC 13.3.0, CMake 4.4.4, Ninja 1.13.2, `TORCH_CUDA_ARCH_LIST=7.5`.
