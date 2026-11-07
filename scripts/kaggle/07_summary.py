"""Stage 07: verdict for THIS notebook (06) from its own stage JSONs. Kaggle gives every notebook a fresh session, so the JSONs of
notebooks 01-05 are normally absent here: they are reported as NOT_RUN (informational) and never counted as a pass.
The notebook verdict depends only on the TP=1 and TP=2 benchmark results written in this session. Exit code 1 unless both PASS."""
import json, sys
from _common import OUT

REQUIRED = ["06_bench_tp1", "06_bench_tp2"]
INFO = ["01_env_audit", "02_install", "03_single_gpu", "04_nccl_dual_gpu", "05_inference_tp1", "05_inference_tp2"]
rows = {}
for n in REQUIRED + INFO:
    try: rows[n] = json.load(open(OUT / f"{n}.json"))["status"]
    except FileNotFoundError: rows[n] = "NOT_RUN"
    except Exception as e: rows[n] = f"ERROR({type(e).__name__})"
for k, v in rows.items(): print(f"{v:9s} {k}" + ("" if k in REQUIRED else "  (other notebook; informational)"))
b = {}
for tp in (1, 2):
    try: b[tp] = json.load(open(OUT / f"06_bench_tp{tp}.json"))["tok_per_s_best"]
    except Exception: pass
measured = None
if len(b) == 2:
    measured = {"tp1_tok_per_s": b[1], "tp2_tok_per_s": b[2], "ratio_tp2_over_tp1": b[2] / b[1]}
    print(f"measured (not assumed): TP1 {b[1]:.1f} tok/s, TP2 {b[2]:.1f} tok/s, ratio {b[2]/b[1]:.2f}x")
ok = all(rows[n] == "PASS" for n in REQUIRED)
verdict = "NOTEBOOK 06 TP BENCHMARK: " + ("PASS" if ok else "FAIL")
(OUT / "ACCEPTANCE_06.json").write_text(json.dumps({"verdict": verdict, "required": {n: rows[n] for n in REQUIRED}, "other_notebooks": {n: rows[n] for n in INFO}, "measured": measured}, indent=1))
print(verdict)
sys.exit(0 if ok else 1)
