# run: torchrun --nproc-per-node=2 04_nccl_dual_gpu.py   (NCCL_DEBUG=INFO NCCL_DEBUG_FILE=<out>/nccl.%h.%p.log)
# Each rank writes its own JSON; 04_nccl_summary.py (or the notebook) aggregates and inspects the transport.
import json, os
os.environ.setdefault("NCCL_P2P_DISABLE", "1"); os.environ.setdefault("NCCL_IB_DISABLE", "1")
from stokastisk.nccl import run_collective_selftest
from _common import save
r = run_collective_selftest()
r["ok"] = bool(r["all_reduce_ok"] and r["all_gather_ok"] and r["reduce_scatter_ok"])
save(f"04_nccl_rank{r['rank']}.json", r); print(json.dumps(r))
raise SystemExit(0 if r["ok"] else 1)
