# Stokastisk 0.3.0 — T4/Triton-only architecture

**Status: rc6 passed notebooks 01–06 on Kaggle (2 × T4) — see `RELEASE.md`.**

## Target (single supported configuration)
Kaggle, Python 3.13, PyTorch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0, glibc 2.39, driver 580.x, **4 CPU cores, ~30 GiB RAM, 20 GiB `/kaggle/working`, 2 × Tesla T4 (SM75, ~15 GiB each), no peer access, PHB topology** (`stokastisk.LIMITS`). NCCL: torch is compiled against 2.28.9; the runtime library on Kaggle is whatever `nvidia/nccl/lib/libnccl.so.2` is (observed 2.32.3, see `RELEASE.md`).
Other GPUs are unsupported on purpose: the native extensions contain `sm_75` cubins only (no PTX).

## Components
| Piece | What it is |
|---|---|
| `stokastisk-0.3.0-py3-none-any.whl` | pure-Python SDK (`import stokastisk`): config validation, T4 checks, NCCL helpers, prohibited-package diagnostics, thin wrapper over `vllm.LLM` |
| `vllm-0.25.1-cp38-abi3-linux_x86_64.whl` | vLLM 0.25.1 **derived** T4-only distribution (see below) |
| `requirements/` | exact `--no-deps` pins + constraints + prohibited list |
| `scripts/kaggle` | installer and the six acceptance stages (run by the project owner on Kaggle) |

### How the vLLM wheel was made (honest provenance)
1. Source: vLLM `v0.25.1`, commit `752a3a504485790a2e8491cacbb35c137339ad34` (verified against the tarball header and the GitHub tag API).
2. Native extensions (`_C_stable_libtorch`, `_moe_C_stable_libtorch`, `cumem_allocator`, `spinloop`, `fs_io_C`, `_rust_tool_parser`) come from the **existing 5.5 h local build** (NVCC 13.0.48, `TORCH_CUDA_ARCH_LIST=7.5`). They were *not* rebuilt: the previous build was audited and found CUDA-13/sm_75-only. That build also compiled FlashAttention 2/3 binaries, which are **not included**. A from-scratch rebuild that never compiles them was not performed (≈5.5 h) and remains an open item.
3. Python tree: `stage.zip` (stripped vLLM Python tree, 2,560 files) — but `stage.zip`'s own `.so` files link **`libcudart.so.12`** and were therefore discarded and replaced by the CUDA-13 build outputs above.
4. FlashAttention/FlashInfer/FlashMLA removal (`scripts/release/build_t4_tree.py`): `vllm_flash_attn/`, `third_party/flashmla/` removed wholesale; 15 further modules removed because they import FlashInfer/FlashAttention/FlashMLA at module level (AST-detected, listed in the release `evidence/`).
5. Patches (`patches/`): 5 files changed so that the core still imports — `fa_utils.py` (names stubbed), `rotary_embedding/common.py` (`forward_cuda` → native torch implementation instead of the Triton rotary kernel that lived inside `vllm_flash_attn`), `fused_moe/all2all_utils.py`, `compilation/passes/pass_manager.py`, `config/compilation.py` (FlashInfer allreduce fusion absent).
6. Metadata rewritten: `Requires-Dist` is only the import-proven core; the upstream `cuda.txt` dependencies (flashinfer, cutlass-dsl, humming-kernels, …) are gone. One deliberate deviation: `fastapi>=0.133.0` instead of `<0.137` (Kaggle ships 0.141.1; the import sweep passes with it).

`vllm/utils/flashinfer.py` and about 30 other `*flash*`-named upstream modules are **retained** because core modules import them; they only probe with `importlib.util.find_spec` or contain lazy imports and never import a FlashInfer/FlashAttention package at module level (AST-verified). They are inert unless the FlashInfer package is installed — which the installer, the SDK and the audits all forbid.

## Attention
`StokastiskConfig(attention_backend="TRITON_ATTN")` is the only accepted value. It is passed as `attention_config={"backend": "TRITON_ATTN"}` (verified against `EngineArgs`/`AttentionConfig` in 0.25.1: dict → `AttentionConfig`, string → `AttentionBackendEnum.TRITON_ATTN`; verified on CPU that `EngineArgs(...).attention_config.backend` is `TRITON_ATTN`). vLLM 0.25.1 raises (`Selected backend … is not valid`) rather than falling back when an explicitly chosen backend is invalid, and logs `Using AttentionBackendEnum.TRITON_ATTN backend.` on success. Acceptance evidence (notebooks 05/06) is three-fold: engine config, that log line per worker, and compiled `kernel_unified_attention` artifacts in a fresh `TRITON_CACHE_DIR`. MLA / hybrid / sparse-attention model families select other backends and are unsupported here.

## CUDA components
| Component | Where it comes from |
|---|---|
| `libcudart.so.13` | dynamic; Kaggle's CUDA 13.0 runtime (NEEDED by the two big extensions) |
| `libcuda.so.1` | dynamic; NVIDIA driver |
| `libtorch*`, `libc10*` | dynamic; Kaggle's PyTorch |
| Triton attention kernels | JIT-compiled by Triton 3.6.0 at run time (no cubin shipped) |
| device code in `.so` | `sm_75` cubins only (`manifests/CUDA-ARCH-MANIFEST.json`) |
| NCCL | the `libnccl.so.2` found by torch's RUNPATH in Kaggle's image (observed 2.32.3+cuda12.9) |
No CUDA library is bundled. Static-vs-dynamic of individual CUDA symbols was not analysed beyond `DT_NEEDED`. glibc requirement of the native files: max GLIBC 2.38 (Kaggle has 2.39).

## NCCL / tensor parallel
No P2P, no NVLink. Defaults set before engine start (never overriding user values): `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, `NCCL_CUMEM_ENABLE=0`; vLLM `disable_custom_all_reduce=True`; `enforce_eager=True`. Notebook 04 runs real all-reduce/all-gather/reduce-scatter and saves `NCCL_DEBUG` logs; TP=2 speed is measured, not assumed.
