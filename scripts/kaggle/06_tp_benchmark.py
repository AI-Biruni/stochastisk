"""Stage 06: one TP configuration per process. usage: 06_tp_benchmark.py TP MODEL"""
import sys, time, torch, stokastisk
from vllm import SamplingParams
from _common import check, finish

tp = int(sys.argv[1]); model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-1.5B-Instruct"
R = []
t0 = time.time(); llm = stokastisk.LLM(model, config=stokastisk.StokastiskConfig(tensor_parallel_size=tp)); load = time.time() - t0
prompts = [f"Explain topic number {i} in tensor parallelism." for i in range(32)]
sp = SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True)
llm.generate(prompts[:2], sp)                      # warm-up
lat = []
for _ in range(3):
    t = time.time(); out = llm.generate(prompts, sp); lat.append(time.time() - t)
toks = sum(len(o.outputs[0].token_ids) for o in out)
mem = [(torch.cuda.mem_get_info(i)[1] - torch.cuda.mem_get_info(i)[0]) >> 20 for i in range(torch.cuda.device_count())]
check(R, "generated expected token count", toks == 32 * 128, toks)
best = min(lat)
finish(f"06_bench_tp{tp}", R, {"tp": tp, "model": model, "load_seconds": load, "runs_seconds": lat, "tokens_per_run": toks,
       "tok_per_s_best": toks / best, "gpu_used_MiB": mem})
