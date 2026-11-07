# Build / reproduce
Toolchain (verified before any build): CPython 3.13, PyTorch 2.11.0+cu130, **CUDA Toolkit 13.0 only** (local 13.0.48; Kaggle runs 13.0.88), GCC 13.3, `TORCH_CUDA_ARCH_LIST=7.5`. CUDA 12.x toolkits must not be on `PATH`/`CUDA_HOME`.
* `scripts/build_vllm_sm75.sh` — the original native vLLM build (SM75). `scripts/release/build_t4_tree.py`, `build_t4_wheel.py`, `repack_vllm_wheel.py`, `patch_vllm_wheel_t4.py` — derive the T4-only Python tree/wheel from that build and `patches/0001–0003`.
* `scripts/make_notebooks.py` — generates the six notebooks (never hand-edit them). `scripts/fetch_release.py` is embedded in them verbatim.
* `scripts/release/assemble_release.py NEW BASE` — assembles `dist-rc/NEW` from `dist-rc/BASE` (vLLM wheel carried over byte-identical, SDK wheel rebuilt and checked against `sdk/`), runs the CPU tests, writes manifests/SHA256SUMS, builds and re-verifies the ZIP.
* `scripts/release/make_kaggle_input_zip.py NAME` — reproducible, idempotent Kaggle/Hub ZIP + `.sha256`.
* `scripts/audit/native_audit.py` — ELF/cubin/GLIBC/CUDA-12-marker audit of every native object. `scripts/audit/forensic_notebooks.py` — executed-notebook forensics.
Tests: `python3.13 -m pytest tests` (CPU only; they do not prove GPU behaviour).
