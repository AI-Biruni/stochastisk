# Build (reproducibility)
Source: vLLM `v0.25.1`, commit `752a3a504485790a2e8491cacbb35c137339ad34` (verified: tarball pax header and GitHub tag API agree), tarball SHA256 `6e41ae18…e5a0`.
Toolchain (local): CPython 3.13.15, torch 2.11.0+cu130, NVCC 13.0.48, GCC 13.3.0, CMake 4.4.4, Ninja 1.13.2, rustc 1.95; `TORCH_CUDA_ARCH_LIST=7.5`, `CMAKE_CUDA_ARCHITECTURES=75`, `MAX_JOBS=1`.
Command: `scripts/build_vllm_sm75.sh` (`setup.py bdist_wheel --py-limited-api=cp38`). Then `scripts/release/repack_vllm_wheel.py` (removes FlashAttention binaries). SDK: `python3.13 -m build --wheel --no-isolation`. Bundle: `scripts/release/assemble_bundle.py`.
To avoid compiling FlashAttention at all in a future rebuild see `patches/README.md`. A Rust-frontend rebuild needs `protoc`.
