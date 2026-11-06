"""Stage 07: overall acceptance verdict from all saved stage JSONs."""
import glob, json
from _common import OUT
need = ["01_env_audit", "02_install", "03_single_gpu", "04_nccl_dual_gpu", "05_inference_tp1", "05_inference_tp2", "06_bench_tp1", "06_bench_tp2"]
rows = {}
for n in need:
    try: rows[n] = json.load(open(OUT / f"{n}.json"))["status"]
    except FileNotFoundError: rows[n] = "NOT RUN"
for k, v in rows.items(): print(f"{v:8s} {k}")
b = {}
for tp in (1, 2):
    try: b[tp] = json.load(open(OUT / f"06_bench_tp{tp}.json"))["tok_per_s_best"]
    except Exception: pass
if len(b) == 2: print(f"measured: TP1 {b[1]:.1f} tok/s, TP2 {b[2]:.1f} tok/s, ratio {b[2]/b[1]:.2f}x")
ok = all(v == "PASS" for v in rows.values())
verdict = "KAGGLE DUAL-T4 RUNTIME ACCEPTANCE: " + ("PASS" if ok else "FAIL / INCOMPLETE")
(OUT / "ACCEPTANCE.txt").write_text(verdict + "\n" + json.dumps(rows, indent=1)); print(verdict)
