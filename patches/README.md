# Patches
No source patches are applied to `work/upstream/vllm-0.25.1` (kept pristine).

Upstream `setup.py` (0.25.1, ~line 1111, `if _is_cuda():`) unconditionally registers the FlashAttention extensions
(`_vllm_fa2_C`, `_vllm_fa3_C`, `_vllm_fa4_cutedsl_C`, FlashMLA). A from-source rebuild that never compiles them would need that block removed;
no such patch has been written or tested.

The released vLLM wheel is instead *derived* from the existing build by `scripts/release/repack_vllm_wheel.py`, which deletes only
`vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so` and `_vllm_fa3_C.abi3.so` and regenerates RECORD. Every other file is byte-identical.
