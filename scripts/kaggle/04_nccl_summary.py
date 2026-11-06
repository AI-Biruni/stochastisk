"""Aggregate both ranks, report NCCL transport (SHM vs socket) from NCCL_DEBUG logs."""
import glob, re
from _common import OUT, check, finish
import json
R = []; ranks = [json.load(open(p)) for p in sorted(glob.glob(str(OUT / "04_nccl_rank*.json")))]
check(R, "two ranks reported", sorted(r["rank"] for r in ranks) == [0, 1], [r["rank"] for r in ranks])
for r in ranks:
    check(R, f"rank{r['rank']} all-reduce/all-gather/reduce-scatter", r["ok"], r)
    check(R, f"rank{r['rank']} cleanup", r.get("cleanup") == "process group destroyed")
logs = "".join(open(p, errors="replace").read() for p in glob.glob(str(OUT / "nccl*.log")))
transport = sorted(set(re.findall(r"via (\S+)", logs)))
print("NCCL transports observed:", transport or "none captured (set NCCL_DEBUG=INFO and NCCL_DEBUG_FILE)")
finish("04_nccl_dual_gpu", R, {"nccl_transports": transport})
