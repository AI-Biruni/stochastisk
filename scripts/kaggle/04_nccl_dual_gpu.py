"""Stage 04 WORKER (one process per GPU). Launched by 04_nccl_launch.py via torch.distributed.run; do not run directly.

Calls the SDK self-test, which does NOT touch NCCL's transport selection (no NCCL_P2P_DISABLE/SHM/IB overrides): the
primary acceptance run reports NCCL's own default choice. Each rank writes its own JSON, even when it fails."""
import json, os, sys
from _common import save
from stokastisk.nccl import run_collective_selftest

r = run_collective_selftest(expected_world=2)
save(f"04_nccl_rank{r['rank']}.json", r)
print(json.dumps({k: r.get(k) for k in ("rank", "local_rank", "world", "backend", "status", "ok", "error")}), flush=True)
sys.exit(0 if r["ok"] else 1)
