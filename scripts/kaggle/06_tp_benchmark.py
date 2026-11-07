"""Stage 06: one TP configuration per process. usage: 06_tp_benchmark.py TP MODEL
Same spawn rule as stage 05: all vLLM work is under `if __name__ == "__main__":` because vLLM's spawned workers re-import this file."""
import os, sys, time
from _common import OUT, check, finish, gpu_used_mib, model_revision, process_tree, run_guarded, tree_loaded_libs, versions


def main() -> int:
    tp = int(sys.argv[1]); model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-1.5B-Instruct"
    os.environ["TRITON_CACHE_DIR"] = str(OUT / f"triton-cache-bench-tp{tp}")
    import stokastisk
    from vllm import SamplingParams
    R = []
    mem_before = gpu_used_mib()
    t0 = time.time(); llm = stokastisk.LLM(model, config=stokastisk.StokastiskConfig(tensor_parallel_size=tp)); load = time.time() - t0
    prompts = [f"Explain topic number {i} in tensor parallelism." for i in range(32)]
    sp = SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True)
    llm.generate(prompts[:2], sp)                      # warm-up
    lat = []
    for _ in range(3):
        t = time.time(); out = llm.generate(prompts, sp); lat.append(time.time() - t)
    toks = sum(len(o.outputs[0].token_ids) for o in out)
    peak = gpu_used_mib()
    check(R, "generated expected token count", toks == 32 * 128, toks)
    llm_backend = llm.selected_attention_backend()
    check(R, "engine config backend == TRITON_ATTN", llm_backend == "TRITON_ATTN", llm_backend)
    # prefill-heavy phase: 8 prompts of ~1.2k tokens, 1 new token each -> prefill throughput
    lp = ["Summarize: " + ("alpha beta gamma delta " * 300) + str(i) for i in range(8)]
    sp1 = SamplingParams(max_tokens=1, temperature=0.0)
    llm.generate(lp[:1], sp1); tp0 = time.time(); po = llm.generate(lp, sp1); pdt = time.time() - tp0
    ptoks = sum(len(o.prompt_token_ids) for o in po)
    best = min(lat)
    peak = [max(a, b) for a, b in zip(peak or [], gpu_used_mib() or [])] or peak
    libs = tree_loaded_libs(); children = process_tree()
    llm.close(); time.sleep(3); leftover = process_tree()
    check(R, "engine shutdown left no child processes", not leftover, leftover)
    finish(f"06_bench_tp{tp}", R, {"tp": tp, "model": model, "model_revision": model_revision(model), "attention_backend_config": llm_backend, "loaded_libraries_by_pid": libs, "engine_child_pids": children, "versions": versions(), "load_seconds": load, "runs_seconds": lat, "tokens_per_run": toks,
           "tok_per_s_best": toks / best, "decode_latency_s_best": best, "prefill_tokens": ptoks, "prefill_seconds": pdt, "prefill_tok_per_s": ptoks / pdt,
           "gpu_used_MiB": {"before_load": mem_before, "peak_sampled": peak, "after_close": gpu_used_mib()}})


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run_guarded(f"06_bench_tp{sys.argv[1] if len(sys.argv) > 1 else 1}", main)
