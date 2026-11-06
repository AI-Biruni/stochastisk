"""Stage 05: real vLLM generation. usage: 05_inference.py TP MODEL
Run as a subprocess with stdout+stderr captured to a log (the notebook does this) so vLLM's own backend
log lines can be inspected. Covers prefill (long prompt) and decode (64 tokens), determinism, memory, latency."""
import json, sys, time
import torch, stokastisk
from _common import check, finish, save

tp = int(sys.argv[1]) if len(sys.argv) > 1 else 1
model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-0.5B-Instruct"
R = []
stokastisk.validate_environment(require_gpus=max(tp, 1))
cfg = stokastisk.StokastiskConfig(tensor_parallel_size=tp, max_model_len=4096)
t0 = time.time(); llm = stokastisk.LLM(model, config=cfg); load_s = time.time() - t0
check(R, "engine config backend == TRITON_ATTN", llm.selected_attention_backend() == "TRITON_ATTN", llm.selected_attention_backend())
check(R, "no FlashAttention native module in this process", stokastisk.report()["flash_attn_native_loaded"] == [])
from vllm import SamplingParams
sp = SamplingParams(max_tokens=64, temperature=0.0)
short = "The capital of France is"
long_p = "Summarize the following text in one sentence.\n" + ("The quick brown fox jumps over the lazy dog. " * 120)   # ~1.2k tokens -> prefill path
res = {}
for name, p in (("decode_short", short), ("prefill_long", long_p)):
    t = time.time(); a = llm.generate([p], sp)[0]; dt = time.time() - t
    b = llm.generate([p], sp)[0]
    txt = a.outputs[0].text
    res[name] = {"prompt_tokens": len(a.prompt_token_ids), "new_tokens": len(a.outputs[0].token_ids), "seconds": dt, "text": txt}
    check(R, f"{name}: non-empty output", len(txt.strip()) > 0, txt[:80])
    check(R, f"{name}: deterministic at temperature 0", txt == b.outputs[0].text)
check(R, "prefill prompt >= 1000 tokens", res["prefill_long"]["prompt_tokens"] >= 1000, res["prefill_long"]["prompt_tokens"])
print("content check (informational):", "Paris" in res["decode_short"]["text"])
mem = [(torch.cuda.mem_get_info(i)[1] - torch.cuda.mem_get_info(i)[0]) >> 20 for i in range(torch.cuda.device_count())]
llm.close()
finish(f"05_inference_tp{tp}", R, {"model": model, "tp": tp, "load_seconds": load_s, "generation": res,
       "paris_in_output": "Paris" in res["decode_short"]["text"], "gpu_used_MiB_before_close": mem})
