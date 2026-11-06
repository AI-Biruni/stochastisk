"""NCCL validation helpers (launch with ``python -m torch.distributed.run --standalone --nproc_per_node=2``).

Design rules (see reports/rc2-notebook-04-forensic-diagnosis.md):
  * one process per GPU; ``torch.cuda.set_device(LOCAL_RANK)`` happens BEFORE any CUDA tensor is created;
  * NCCL's default transport selection is NOT altered here (no NCCL_P2P_DISABLE / NCCL_SHM_DISABLE / NCCL_IB_DISABLE);
  * a finite, configurable process-group timeout (``STOKASTISK_NCCL_TIMEOUT_S``, default 120 s);
  * every collective gets input description, expected value, actual-vs-expected comparison, duration and its own status;
  * the function never raises: failures become ERROR records with full tracebacks, and the process group is always destroyed;
  * ``STOKASTISK_NCCL_CONTROL=gloo`` runs the same logic on CPU/Gloo as a *control experiment* for local testing only;
    such a run is flagged ``acceptance_valid: false`` and can never count as NCCL acceptance.
"""
from __future__ import annotations

import datetime
import os
import time
import traceback
from typing import Any

SCHEMA_VERSION = 2
PASS, FAIL, ERROR, SKIPPED, NOT_RUN = "PASS", "FAIL", "ERROR", "SKIPPED", "NOT_RUN"


def _rec(name: str, status: str, **kw: Any) -> dict[str, Any]:
    return {"name": name, "status": status, **kw}


def run_collective_selftest(expected_world: int = 2) -> dict[str, Any]:
    """Run all-reduce / all-gather / reduce-scatter / broadcast / barrier over the torchrun world. Never raises."""
    import torch
    import torch.distributed as dist

    control = os.environ.get("STOKASTISK_NCCL_CONTROL", "").lower() == "gloo"
    backend = "gloo" if control else "nccl"
    timeout_s = float(os.environ.get("STOKASTISK_NCCL_TIMEOUT_S", "120"))
    res: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "backend": backend, "acceptance_valid": not control,
        "rank": int(os.environ.get("RANK", -1)), "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
        "world": int(os.environ.get("WORLD_SIZE", -1)), "timeout_s": timeout_s,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "python_pid": os.getpid(),
        "env": {k: os.environ.get(k) for k in (
            "CUDA_VISIBLE_DEVICES", "MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
            "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_DEBUG_FILE", "NCCL_SOCKET_IFNAME", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE",
            "NCCL_IB_DISABLE", "TORCH_NCCL_ASYNC_ERROR_HANDLING", "TORCH_DISTRIBUTED_DEBUG", "OMP_NUM_THREADS")},
        "collectives": [], "tracebacks": {}, "status": ERROR,
    }
    ops = res["collectives"]
    initialized = False
    try:
        rank, world, local = res["rank"], res["world"], res["local_rank"]
        if rank < 0 or world < 0 or local < 0:
            raise RuntimeError("RANK/LOCAL_RANK/WORLD_SIZE not set: launch with torch.distributed.run")
        if world != expected_world:
            raise RuntimeError(f"world size {world} != required {expected_world}")
        if control:
            dev = torch.device("cpu")
        else:
            if not torch.cuda.is_available() or local >= torch.cuda.device_count():
                raise RuntimeError(f"LOCAL_RANK {local} has no CUDA device (device_count={torch.cuda.device_count()})")
            torch.cuda.set_device(local)                       # BEFORE any CUDA tensor
            dev = torch.device("cuda", local)
            p = torch.cuda.get_device_properties(dev)
            res["device"] = {"index": local, "name": p.name, "compute_capability": [p.major, p.minor], "total_MiB": p.total_memory >> 20,
                             "uuid": str(getattr(p, "uuid", "")), "pci_bus_id": f"{getattr(p, 'pci_domain_id', 0):04x}:{getattr(p, 'pci_bus_id', 0):02x}:{getattr(p, 'pci_device_id', 0):02x}.0"}
            res["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
            if rank == 0:
                res["peer_access"] = {f"{a}->{b}": bool(torch.cuda.can_device_access_peer(a, b))
                                      for a in range(torch.cuda.device_count()) for b in range(torch.cuda.device_count()) if a != b}
            torch.zeros(1, device=dev); torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            res["gpu_mem_used_MiB_baseline"] = (total - free) >> 20   # CUDA context only, before the process group exists
        t0 = time.time()
        dist.init_process_group(backend, timeout=datetime.timedelta(seconds=timeout_s))
        initialized = True
        res["init_duration_s"] = round(time.time() - t0, 4)

        def sync():
            if not control:
                torch.cuda.synchronize()

        def run(name: str, fn):
            """Run one collective; ERROR (with traceback) stops later collectives (they could hang) -> NOT_RUN."""
            if any(o["status"] in (ERROR, FAIL) and o.get("fatal") for o in ops):
                ops.append(_rec(name, NOT_RUN, reason="an earlier collective errored")); return
            t = time.time()
            try:
                r = fn(); sync()
                r["duration_s"] = round(time.time() - t, 6); ops.append(_rec(name, PASS if r.pop("ok") else FAIL, **r))
            except Exception as e:  # noqa: BLE001
                res["tracebacks"][name] = traceback.format_exc()
                ops.append(_rec(name, ERROR, fatal=True, error=f"{type(e).__name__}: {e}", duration_s=round(time.time() - t, 6)))

        # distinct devices (NCCL only): every rank must be on a different physical GPU
        if not control:
            def distinct():
                objs = [None] * world; dist.all_gather_object(objs, res["device"]["uuid"] or f"idx{local}")
                return {"ok": len(set(objs)) == world, "input": "GPU uuid per rank", "expected": f"{world} distinct", "actual": objs}
            run("distinct_devices", distinct)

        def all_reduce():
            x = torch.full((1 << 20,), float(rank + 1), device=dev, dtype=torch.float32); dist.all_reduce(x)
            exp = float(sum(range(1, world + 1))); err = float((x - exp).abs().max())
            return {"ok": err == 0.0, "input": f"float32[1<<20] filled with rank+1", "expected": exp, "actual_min_max": [float(x.min()), float(x.max())], "max_abs_err": err}

        def all_gather():
            g = [torch.empty(1024, device=dev, dtype=torch.float16) for _ in range(world)]
            dist.all_gather(g, torch.full((1024,), float(rank), device=dev, dtype=torch.float16))
            errs = [float((t.float() - i).abs().max()) for i, t in enumerate(g)]
            return {"ok": all(e == 0.0 for e in errs), "input": "float16[1024] filled with rank", "expected": "chunk i == i", "actual": [float(t[0]) for t in g], "max_abs_err": max(errs)}

        def reduce_scatter():
            if control:                                        # Gloo has no reduce_scatter
                return {"ok": True, "skipped": True}
            ins = [torch.full((1024,), float((rank + 1) * (j + 1)), device=dev, dtype=torch.float32) for j in range(world)]
            out = torch.empty(1024, device=dev, dtype=torch.float32); dist.reduce_scatter(out, ins)
            exp = float(sum((src + 1) * (rank + 1) for src in range(world)))   # sum over source ranks of chunk[rank]
            err = float((out - exp).abs().max())
            return {"ok": err == 0.0, "input": "chunk j on rank r = (r+1)*(j+1)", "expected": exp, "actual_min_max": [float(out.min()), float(out.max())], "max_abs_err": err}

        def broadcast():
            x = torch.arange(1024, device=dev, dtype=torch.float32) if rank == 0 else torch.zeros(1024, device=dev, dtype=torch.float32)
            dist.broadcast(x, src=0); err = float((x - torch.arange(1024, device=dev, dtype=torch.float32)).abs().max())
            return {"ok": err == 0.0, "input": "rank0: arange(1024); others: zeros", "expected": "arange(1024) on every rank", "max_abs_err": err}

        def barrier():
            dist.barrier(); return {"ok": True, "input": "barrier", "expected": "all ranks continue"}

        run("all_reduce_sum", all_reduce); run("all_gather", all_gather)
        if control:
            ops.append(_rec("reduce_scatter", SKIPPED, reason="Gloo has no reduce_scatter (control run)"))
        else:
            run("reduce_scatter", reduce_scatter)
        run("broadcast", broadcast); run("barrier", barrier)
        res["status"] = PASS if ops and all(o["status"] in (PASS, SKIPPED) for o in ops) else (ERROR if any(o["status"] == ERROR for o in ops) else FAIL)
    except Exception as e:  # noqa: BLE001
        res["tracebacks"]["setup"] = traceback.format_exc()
        res["error"] = f"{type(e).__name__}: {e}"; res["status"] = ERROR
    finally:
        try:
            if initialized and dist.is_initialized():
                dist.destroy_process_group()
            res["cleanup"] = "process group destroyed" if initialized else "process group was never initialized"
        except Exception as e:  # noqa: BLE001
            res["cleanup"] = f"destroy_process_group failed: {type(e).__name__}: {e}"; res["tracebacks"]["cleanup"] = traceback.format_exc()
        res["dist_initialized_after_teardown"] = dist.is_initialized()
        try:
            if not control and torch.cuda.is_available():
                torch.cuda.synchronize(); torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info()
                res["gpu_mem_used_MiB_after_teardown"] = (total - free) >> 20
        except Exception as e:  # noqa: BLE001
            res["tracebacks"]["post_teardown_memory"] = traceback.format_exc()
    res["ok"] = res["status"] == PASS and res.get("dist_initialized_after_teardown") is False
    return res
