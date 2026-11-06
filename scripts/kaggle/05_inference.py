"""Stage 05: real vLLM generation with TRITON_ATTN. usage: 05_inference.py TP MODEL
Run as a subprocess with stdout+stderr captured to a log (the notebook does) so vLLM's own backend log lines
can be inspected. Evidence of the attention backend (all required):
  1. engine config recorded TRITON_ATTN (config-level)
  2. a FRESH TRITON_CACHE_DIR contains compiled `kernel_unified_attention` artifacts after generation (execution-level)
  3. the notebook greps the log for `Using AttentionBackendEnum.TRITON_ATTN backend.` (once per worker) and for absence of flash backends"""
import json, os, pathlib, sys, time
from _common import OUT
tp = int(sys.argv[1]) if len(sys.argv) > 1 else 1
model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-0.5B-Instruct"
cache = OUT / f"triton-cache-tp{tp}"
import shutil; shutil.rmtree(cache, ignore_errors=True); cache.mkdir(parents=True)
os.environ["TRITON_CACHE_DIR"] = str(cache)          # fresh: forces JIT, so artifacts prove kernels really ran
import torch, stokastisk
from _common import check, finish
R = []
stokastisk.validate_environment(require_gpus=max(tp, 1))
cfg = stokastisk.StokastiskConfig(tensor_parallel_size=tp, max_model_len=4096)
t0 = time.time(); llm = stokastisk.LLM(model, config=cfg); load_s = time.time() - t0
check(R, "engine config backend == TRITON_ATTN", llm.selected_attention_backend() == "TRITON_ATTN", llm.selected_attention_backend())
check(R, "no prohibited packages/modules/binaries", stokastisk.prohibited_report()["clean"])
from vllm import SamplingParams
sp = SamplingParams(max_tokens=64, temperature=0.0)
short = "The capital of France is"
long_p = "Summarize the following text in one sentence.\n" + ("The quick brown fox jumps over the lazy dog. " * 120)   # ~1.2k tokens -> chunked prefill path
res = {}
for name, p in (("decode_short", short), ("prefill_long", long_p)):
    t = time.time(); a = llm.generate([p], sp)[0]; dt = time.time() - t
    b = llm.generate([p], sp)[0]; txt = a.outputs[0].text
    res[name] = {"prompt_tokens": len(a.prompt_token_ids), "new_tokens": len(a.outputs[0].token_ids), "seconds": dt, "text": txt}
    check(R, f"{name}: non-empty output", len(txt.strip()) > 0, txt[:80])
    check(R, f"{name}: deterministic at temperature 0", txt == b.outputs[0].text)
check(R, "prefill prompt >= 1000 tokens", res["prefill_long"]["prompt_tokens"] >= 1000, res["prefill_long"]["prompt_tokens"])
arts = sorted({p.name for p in cache.rglob("*") if p.is_file() and "unified_attention" in p.name})
check(R, "Triton cache holds compiled kernel_unified_attention (kernel actually ran)", bool(arts), arts[:4] or "no artifacts")
print("content check (informational):", "Paris" in res["decode_short"]["text"])
mem = [(torch.cuda.mem_get_info(i)[1] - torch.cuda.mem_get_info(i)[0]) >> 20 for i in range(torch.cuda.device_count())]
llm.close()
finish(f"05_inference_tp{tp}", R, {"model": model, "tp": tp, "load_seconds": load_s, "generation": res, "triton_attention_artifacts": arts[:10],
       "paris_in_output": "Paris" in res["decode_short"]["text"], "gpu_used_MiB_before_close": mem})
