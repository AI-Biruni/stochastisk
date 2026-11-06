"""CPU import sweep of the core inference path; prints JSON {ok, errors, third_party_dists_imported}."""
import importlib, importlib.metadata as md, json, sys, traceback
MODS = ["vllm", "vllm.LLM" if False else "vllm.entrypoints.llm", "vllm.sampling_params", "vllm.engine.arg_utils",
 "vllm.config", "vllm.v1.attention.backends.registry", "vllm.v1.attention.selector", "vllm.v1.attention.backends.triton_attn",
 "vllm.v1.attention.ops.triton_unified_attention", "vllm.v1.engine.core", "vllm.v1.engine.llm_engine", "vllm.v1.engine.core_client",
 "vllm.v1.executor.multiproc_executor", "vllm.v1.executor.uniproc_executor", "vllm.v1.worker.gpu_worker", "vllm.v1.worker.gpu_model_runner",
 "vllm.model_executor.models.qwen2", "vllm.model_executor.model_loader", "vllm.distributed.parallel_state",
 "vllm.distributed.device_communicators.custom_all_reduce", "vllm.distributed.device_communicators.pynccl", "vllm.tokenizers",
 "vllm.platforms", "vllm.transformers_utils.config", "vllm._custom_ops", "vllm.model_executor.layers.attention.attention",
 "vllm._C_stable_libtorch", "vllm._moe_C_stable_libtorch", "vllm.cumem_allocator", "vllm.spinloop", "vllm.fs_io_C"] + sys.argv[1:]
errs = {}
for m in MODS:
    try: importlib.import_module(m)
    except BaseException as e: errs[m] = f"{type(e).__name__}: {e}"[:300]
pd = md.packages_distributions(); used = set()
for name in list(sys.modules):
    top = name.split(".")[0]
    if top in pd and top not in ("vllm", "stokastisk"):
        used.update(pd[top])
print(json.dumps({"ok": not errs, "errors": errs, "third_party_dists_imported": sorted(used)}))
