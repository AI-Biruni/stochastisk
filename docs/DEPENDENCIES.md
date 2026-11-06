# Dependencies and the failed first installer

## What went wrong in the first release (traced from the executed Notebook 02 log)
1. **Transformers 5.20.2 → 4.57.6.** The installer passed ~40 requirements to one `pip install`. pip's resolver backtracked at `xgrammar` (log line “looking at multiple versions of xgrammar”) and then re-collected `transformers>=4.45.0` (from `compressed-tensors==0.17.0`, which has no upper bound) choosing 4.57.6, dragging `huggingface-hub<1.0` and `tokenizers<=0.23.0` with it. `transformers` was not protected by any constraint (only torch/triton/numpy were). vLLM 0.25.1 then refused to import.
2. **CUDA 13.0 → 13.2 compiler wheels.** `flashinfer-python → cuda-tile[tileiras] → nvidia-nvvm>=13.2` and `humming-kernels[cu13] → nvidia-cuda-nvcc` replaced `nvidia-nvvm`/`nvidia-cuda-nvcc` 13.0.88 with 13.2.86.
3. **FlashInfer installed.** The vLLM wheel's `Requires-Dist` carried upstream `cuda.txt` (flashinfer-python/cubin 0.6.13, cutlass-dsl, quack, …); my installer only excluded the name `flashinfer`, not `flashinfer-python`/`-cubin`, and only checked FlashAttention binaries.
4. 15 preinstalled packages were replaced (fastapi 0.141.1→0.136.3, opentelemetry 1.42.1→1.45.2 …), creating new conflicts for gradio, diffusers, google-adk, cuml/cudf, ydata-profiling.

## The new design
* The wheel's metadata no longer contains those packages. `Requires-Dist` = packages the core import path was *proven* (import sweep, `scripts/audit/import_sweep.py`) to need.
* The installer never runs a resolver on Kaggle. It plans from a before-snapshot, fails untouched on any unsatisfied requirement, installs both wheels `--no-deps`, then only the exact pins in `requirements/kaggle-t4-core.txt` with `--no-deps`, then requires the after-snapshot to equal the before-snapshot plus exactly those packages.

## Core pins (absent on Kaggle; version pip itself chose in the executed run; 57.3 MB total download)
apache-tvm-ffi 0.1.9, xgrammar 0.2.5.post1 (48 MB), llguidance 1.7.6, cbor2 6.2.0.post2, msgspec 0.22.0, openai-harmony 0.0.8, partial-json-parser 0.2.1.1.post7, pybase64 1.5.1.
Everything else the core imports (numpy, transformers ≥5.5.3, tokenizers, fastapi, pydantic, aiohttp, openai, …) is already on Kaggle and untouched. Verified: the import sweep passes against Kaggle's *exact* preinstalled versions (transformers 5.20.2, tokenizers 0.23.3, huggingface-hub 1.33.0, fastapi 0.141.1).

## Not installed by the core profile (and why)
| Package(s) | Status |
|---|---|
| flashinfer-python/-cubin, flash-attn, vllm-flash-attn | PROHIBITED |
| cuda-tile, nvidia-cuda-tileiras, nvidia-cutlass-dsl*, humming-kernels, quack-kernels, tilelang, tokenspeed-* | PROHIBITED (CUDA-13.2 / FlashInfer chain; not used by Triton attention) |
| PyNvVideoCodec, torchcodec, nvtx, fastsafetensors, nvidia-cudnn-frontend, numba, z3, … | not imported by the core path; not installed |
| fastapi-cli, sse-starlette, pydantic-settings, model-hosting-container-standards, prometheus-fastapi-instrumentator | optional-server |
| opentelemetry-exporter-otlp, -semantic-conventions-ai | optional-observability (would upgrade opentelemetry-api/sdk) |
| compressed-tensors, mistral_common, lm-format-enforcer, outlines_core, diskcache, interegular, depyf, ijson, setproctitle | optional-models |
Caveat: lazy imports on code paths not reached by the import sweep (e.g. a specific quantization or tokenizer) can still need an optional package; the acceptance notebooks will reveal this for Qwen2.5.
