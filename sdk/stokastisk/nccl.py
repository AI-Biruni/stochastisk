"""NCCL validation helpers (launch with torchrun --nproc-per-node=2)."""
from __future__ import annotations

import os
from typing import Any


def run_collective_selftest() -> dict[str, Any]:
    """Run all-reduce / all-gather / reduce-scatter across the torchrun world. Requires GPUs."""
    import torch
    import torch.distributed as dist
    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group("nccl")
    dev = torch.device("cuda", torch.cuda.current_device())
    res: dict[str, Any] = {"rank": rank, "local_rank": torch.cuda.current_device(), "world": world, "device": torch.cuda.get_device_name(dev), "nccl": ".".join(map(str, torch.cuda.nccl.version()))}
    try:
        x = torch.full((1 << 20,), float(rank + 1), device=dev, dtype=torch.float16)
        dist.all_reduce(x); torch.cuda.synchronize()
        expect = sum(range(1, world + 1))
        res["all_reduce_ok"] = bool((x == expect).all())
        g = [torch.empty(1024, device=dev, dtype=torch.float16) for _ in range(world)]
        dist.all_gather(g, torch.full((1024,), float(rank), device=dev, dtype=torch.float16))
        res["all_gather_ok"] = all(bool((t == i).all()) for i, t in enumerate(g))
        out = torch.empty(1024, device=dev, dtype=torch.float16)
        dist.reduce_scatter(out, [torch.ones(1024, device=dev, dtype=torch.float16) for _ in range(world)])
        torch.cuda.synchronize()
        res["reduce_scatter_ok"] = bool((out == world).all())
    finally:
        dist.destroy_process_group()
        res["cleanup"] = "process group destroyed"
    return res
