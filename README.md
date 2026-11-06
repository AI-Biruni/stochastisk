# Stokastisk 0.3.0 — EXPERIMENTAL, KAGGLE DUAL-T4 VALIDATION PENDING

Python SDK (`import stokastisk`) around the real vLLM 0.25.1 engine for 2 × Tesla T4 (SM75), CPython 3.13, CUDA 13.0.
Attention is explicitly `TRITON_ATTN` (vLLM's built-in Triton backend). FlashAttention is not shipped or selectable.
See `docs/` and `INSTALL.md`.
