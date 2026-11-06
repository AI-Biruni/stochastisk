# Architecture (Stokastisk 0.3.0)

**Status: EXPERIMENTAL — KAGGLE DUAL-T4 VALIDATION PENDING.**

## Packaging (decision: two wheels in one offline bundle)
| Wheel | Tag | Contents |
|---|---|---|
| `stokastisk-0.3.0-py3-none-any.whl` | pure Python | `import stokastisk` SDK; declares `Requires-Dist: vllm==0.25.1`, `Requires-Python >=3.13,<3.14` |
| `vllm-0.25.1-cp38-abi3-linux_x86_64.whl` | stable-ABI, linux_x86_64 (NOT manylinux) | real vLLM 0.25.1 + native extensions (`_C_stable_libtorch`, `_moe_C_stable_libtorch`, `cumem_allocator`, `spinloop`, `fs_io_C`, `_rust_tool_parser`) built for sm_75 against CUDA 13.0 |

A single merged wheel was rejected: it would collide with vLLM's own `.dist-info`/metadata and make `pip` treat Stokastisk as vLLM.
The SDK wheel is pure Python, so the honest tag is `py3-none-any`; the vLLM wheel keeps the tag its build produced.
The vLLM wheel is *derived*: the original 285 MB build output contained FlashAttention binaries (`_vllm_fa2_C`, `_vllm_fa3_C`),
which this release forbids; they were removed (see `patches/README.md`). Python sources of `vllm/vllm_flash_attn` remain (they are only imported by the FlashAttention backend).
Wheel-level `Requires-Dist` is pinned `vllm==0.25.1`; install both with `--no-deps` (`02_install.py` does).

## Attention
`StokastiskConfig.attention_backend` is restricted to `TRITON_ATTN`; any FlashAttention name raises `ConfigurationError`.
It is passed as `attention_config={"backend": "TRITON_ATTN"}`. Verified in vLLM 0.25.1 source: `EngineArgs.__post_init__` converts the dict to `AttentionConfig`,
whose `validate_backend_before` maps the string to `AttentionBackendEnum.TRITON_ATTN`. On CPU I verified `EngineArgs(...).attention_config.backend == AttentionBackendEnum.TRITON_ATTN`.
In `vllm/platforms/cuda.py` an explicitly selected backend that is invalid for the platform raises; there is no silent fallback. `TritonAttentionBackend.supports_compute_capability` returns True.
`LLM.verify_attention_backend()` checks the live engine's `VllmConfig`; this is config-level evidence. Kernel-level evidence is vLLM's own log line `Using AttentionBackendEnum.TRITON_ATTN backend.`, which notebook 05 greps for.
Model-specific limits: MLA / hybrid-Mamba / sparse-attention models choose their own backends in vLLM and may not be usable with `TRITON_ATTN`; use dense decoder-only models (e.g. Qwen2.5, Llama) on T4. T4 has no bf16: use `half`.

## NCCL / tensor parallel
Kaggle audit: no CUDA peer access, PHB topology. Defaults: `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, `NCCL_CUMEM_ENABLE=0` (set before engine start, never overriding user values), vLLM `disable_custom_all_reduce=True`, `enforce_eager=True`.
These are settings, not proof: notebook 04 runs real collectives and saves NCCL_DEBUG logs so the transport (SHM vs socket) can be read. Default TP=1; TP=2 is selectable (`tensor_parallel_size`), speed-up is not assumed (notebook 06 measures).

## Native binary facts (static, from the local build; NVCC 13.0.48; Kaggle has 13.0.88)
See `manifests/ELF-DEPENDENCY-MANIFEST.json` and `CUDA-ARCH-MANIFEST.json`. All CUDA-using extensions NEED `libcudart.so.13` (none CUDA 12) and contain only `sm_75` cubins (no PTX, so no forward compatibility — T4 only).
`_C_stable_libtorch.abi3.so` requires GLIBC 2.38 (Ubuntu 24.04 or newer); notebook 01 checks this. Libraries not bundled and supplied by the target: libtorch*, libc10*, libcudart.so.13, libcuda.so.1 (driver), libstdc++, libgcc_s, libc.
ABI note: files are named `*.abi3.so` and the wheel is tagged `cp38-abi3`; this was produced by the upstream `--py-limited-api=cp38` build. Import under CPython 3.13 was verified locally (CPU); full stable-ABI conformance of every symbol was not independently proven.

## Known build-log findings
The log shows `vllm-server` (Rust frontend) failed: `Could not find protoc`. The wheel still assembled; the Python engine is unaffected, but the optional Rust HTTP frontend binary is absent. `_rust_tool_parser` did build. Log has no `Traceback`/`FAILED:`/`ninja: build stopped`; the final lines show wheel assembly (`adding ...dist-info/RECORD`, `removing build/bdist...`). The process exit code itself was not recorded in the log.
