"""Stage 05: real vLLM generation with TRITON_ATTN. usage: 05_inference.py TP MODEL
Run as a subprocess with stdout+stderr captured to a log (the notebook does) so vLLM's own backend log lines can be
inspected by 05_backend_evidence.py.

IMPORTANT: everything that touches vLLM lives under `if __name__ == "__main__":`. vLLM forces the `spawn` start method
(CUDA is already initialised in this process), and spawn RE-IMPORTS this file in every engine/worker child. Module-level
engine code therefore crashed Kaggle notebook 05 with "An attempt has been made to start a new process before the current
process has finished its bootstrapping phase" (and would also have wiped the Triton cache from inside the children).

Evidence of the attention backend (all required, none sufficient alone):
  1. engine config recorded TRITON_ATTN (config-level)
  2. a FRESH TRITON_CACHE_DIR holds compiled `kernel_unified_attention` artifacts after generation (execution-level)
  3. 05_backend_evidence.py finds vLLM's own `Using AttentionBackendEnum.TRITON_ATTN backend.` log line and no other backend."""
import json, os, pathlib, shutil, sys, time
from _common import OUT, check, finish, gpu_used_mib, run_guarded, versions


def main():
    tp = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-0.5B-Instruct"
    stage = f"05_inference_tp{tp}"
    cache = OUT / f"triton-cache-tp{tp}"
    shutil.rmtree(cache, ignore_errors=True); cache.mkdir(parents=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache)      # fresh + inherited by spawned workers: artifacts prove kernels really ran
    import torch, stokastisk
    from vllm import SamplingParams
    R = []
    stokastisk.validate_environment(require_gpus=max(tp, 1))
    mem_before = gpu_used_mib()
    cfg = stokastisk.StokastiskConfig(tensor_parallel_size=tp, max_model_len=4096)
    t0 = time.time(); llm = stokastisk.LLM(model, config=cfg); load_s = time.time() - t0
    check(R, "engine config backend == TRITON_ATTN", llm.selected_attention_backend() == "TRITON_ATTN", llm.selected_attention_backend())
    check(R, "no prohibited packages/modules/binaries", stokastisk.prohibited_report()["clean"])
    short = "The capital of France is"
    long_p = "Summarize the following text in one sentence.\n" + ("The quick brown fox jumps over the lazy dog. " * 120)   # ~1.2k tokens
    sp = SamplingParams(max_tokens=64, temperature=0.0)
    sp1 = SamplingParams(max_tokens=1, temperature=0.0)
    res, peak = {}, list(gpu_used_mib() or [])
    for name, p in (("decode_short", short), ("prefill_long", long_p)):
        t = time.time(); pre = llm.generate([p], sp1)[0]; pre_s = time.time() - t          # prefill + 1 token
        t = time.time(); a = llm.generate([p], sp)[0]; total_s = time.time() - t           # prefill + full decode
        b = llm.generate([p], sp)[0]; txt = a.outputs[0].text
        n_new = len(a.outputs[0].token_ids); dec_s = max(total_s - pre_s, 1e-9)
        res[name] = {"prompt_tokens": len(a.prompt_token_ids), "new_tokens": n_new, "prefill_seconds_incl_1_token": pre_s,
                     "end_to_end_seconds": total_s, "decode_seconds_est": dec_s,
                     "decode_tok_per_s_est": (n_new - 1) / dec_s if n_new > 1 else None,
                     "end_to_end_tok_per_s": n_new / total_s, "text": txt}
        check(R, f"{name}: non-empty output", len(txt.strip()) > 0, txt[:80])
        check(R, f"{name}: deterministic at temperature 0", txt == b.outputs[0].text)
        cur = gpu_used_mib() or []
        peak = [max(x, y) for x, y in zip(peak, cur)] if peak else cur
    check(R, "prefill prompt >= 1000 tokens", res["prefill_long"]["prompt_tokens"] >= 1000, res["prefill_long"]["prompt_tokens"])
    arts = sorted({p.name for p in cache.rglob("*") if p.is_file() and "kernel_unified_attention" in p.name})
    check(R, "Triton cache holds compiled kernel_unified_attention (kernel actually ran)", bool(arts), arts[:4] or "no artifacts")
    check(R, "GPU memory measured", bool(peak), peak)
    backend = llm.selected_attention_backend()
    llm.close()
    time.sleep(2)
    mem_after = gpu_used_mib()
    finish(stage, R, {"model": model, "tp": tp, "attention_backend_config": backend, "versions": versions(), "load_seconds": load_s,
                      "generation": res, "triton_attention_artifacts": arts[:10],
                      "paris_in_output (informational)": "Paris" in res["decode_short"]["text"],
                      "gpu_used_MiB": {"before_load": mem_before, "peak_sampled_after_generation": peak, "after_close": mem_after},
                      "note": "gpu_used_MiB is device-wide nvidia-smi; 'peak' is the max of samples taken between generations, not a continuous peak."})


if __name__ == "__main__":
    run_guarded(f"05_inference_tp{sys.argv[1] if len(sys.argv) > 1 else 1}", main)
