"""Shared helpers for the Stokastisk Kaggle acceptance scripts."""
import hashlib, importlib, json, os, pathlib, subprocess, sys, time, traceback

# Statuses: PASS FAIL ERROR SKIPPED WARNING NOT_RUN. Only these never fail a stage; FAIL/ERROR/NOT_RUN do.
OK_STATUSES = ("PASS", "WARNING", "SKIPPED")   # INSUFFICIENT_EVIDENCE is deliberately NOT here
OUT = pathlib.Path(os.environ.get("STOKASTISK_OUT", "/kaggle/working/stokastisk-acceptance"))
OUT.mkdir(parents=True, exist_ok=True)


def save(name, obj):
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, default=str))
    return p


def sh(cmd, **kw):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, **kw)
    return (r.stdout + r.stderr).strip()


def check(results, name, ok, detail=""):
    results.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)})
    print(("PASS " if ok else "FAIL ") + name + (f" -- {detail}" if detail else ""))
    return ok


def finish(stage, results, extra=None):
    # WARNING/SKIPPED = a documented limitation (e.g. cuobjdump absent); it never hides a FAIL and never counts as a verified PASS.
    status = "PASS" if results and all(r["status"] in OK_STATUSES for r in results) else "FAIL"
    try:   # Kaggle's /kaggle/working is limited to 20 GiB: the evidence directory itself must fit
        from stokastisk.limits import dir_size_bytes, LIMITS
        used = dir_size_bytes(OUT) / 1024 ** 3
        (extra := dict(extra or {}))["output_dir_gib"] = {"used": round(used, 3), "quota": LIMITS.working_dir_gib}
        if used > LIMITS.working_dir_gib:
            results.append({"check": "evidence directory within the 20 GiB Kaggle quota", "status": "FAIL", "detail": f"{used:.2f} GiB"}); status = "FAIL"
    except Exception:  # noqa: BLE001
        pass
    save(f"{stage}.json", {"stage": stage, "status": status, "checks": results, **(extra or {})})
    print(f"\n[{stage}] OVERALL: {status}")
    if status != "PASS":
        sys.exit(1)


def warn(results, name, detail=""):
    results.append({"check": name, "status": "WARNING", "detail": str(detail)})
    print(f"WARNING {name} -- {detail}")


def skipped(results, name, detail=""):
    results.append({"check": name, "status": "SKIPPED", "detail": str(detail)})
    print(f"SKIPPED {name} -- {detail}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_installed(stage, modules=("stokastisk", "vllm")):
    """Kaggle runs every notebook in a FRESH session: packages installed by notebook 02 do not exist in notebooks 03-06.
    Fail early, with machine-readable evidence and the fix, instead of an anonymous ModuleNotFoundError."""
    R, missing = [], {}
    for m in modules:
        try:
            importlib.import_module(m)
            check(R, f"import {m}", True)
        except Exception as e:  # noqa: BLE001
            missing[m] = traceback.format_exc()
            check(R, f"import {m}", False, f"{type(e).__name__}: {e}")
    if missing:
        hint = ("Kaggle starts each notebook in a fresh session; install first with "
                "`python scripts/kaggle/02_install.py <bundle>` (idempotent). The generated notebook 03-06 install cell does this.")
        print("\n" + hint)
        finish(stage, R, {"missing_modules": missing, "hint": hint, "script": os.path.abspath(sys.argv[0]),
                          "script_sha256": sha256(sys.argv[0])})


def gpu_used_mib():
    """Per-GPU used memory (MiB) from nvidia-smi -- device-wide, creates no CUDA context in the caller. None if unavailable."""
    out = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
    try:
        return [int(x) for x in out.split()]
    except ValueError:
        return None


def versions():
    """Observed (never assumed) runtime versions. `nccl_compiled` is what torch was built against; the NCCL actually loaded
    at runtime is recorded from NCCL's own log by the NCCL stage."""
    v = {"python": sys.version.split()[0]}
    for m in ("torch", "triton", "vllm", "transformers", "stokastisk"):
        try:
            v[m] = importlib.import_module(m).__version__
        except Exception as e:  # noqa: BLE001
            v[m] = f"unavailable: {type(e).__name__}"
    try:
        import torch
        v["torch_cuda"] = torch.version.cuda
        v["nccl_compiled"] = ".".join(map(str, torch.cuda.nccl.version()))
        v["gpus"] = [{"name": torch.cuda.get_device_name(i), "capability": list(torch.cuda.get_device_capability(i))}
                     for i in range(torch.cuda.device_count())]
    except Exception as e:  # noqa: BLE001
        v["torch_probe_error"] = f"{type(e).__name__}: {e}"
    try:
        v["cuda_toolkit"] = toolkit_versions()
    except Exception as e:  # noqa: BLE001
        v["cuda_toolkit_error"] = f"{type(e).__name__}: {e}"
    return v


def run_guarded(stage, fn):
    """Run `fn()` so that an exception ALWAYS leaves <stage>.json with status ERROR and the complete traceback
    (previously a crash left no result file at all), then exits non-zero."""
    try:
        rc = fn()
        if isinstance(rc, int) and rc != 0:
            sys.exit(rc)
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        save(f"{stage}.json", {"stage": stage, "status": "ERROR", "checks": [], "traceback": tb})
        print(f"\n[{stage}] OVERALL: ERROR")
        sys.exit(1)


LIB_PAT = ("libnccl", "libcudart", "libcublas", "libcudnn", "libnvrtc", "libflash", "flashinfer", "vllm_flash_attn", "_vllm_fa", "flashmla")


def loaded_libs(pid):
    """Shared objects of interest actually mapped into process `pid` (from /proc/<pid>/maps) -> sorted unique paths. Never raises."""
    out = set()
    try:
        for line in open(f"/proc/{pid}/maps", errors="replace"):
            path = line.split(None, 5)[5].strip() if len(line.split(None, 5)) > 5 else ""
            if path.startswith("/") and any(k in os.path.basename(path).lower() or k in path.lower() for k in LIB_PAT):
                out.add(path)
    except OSError as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return sorted(out)


def process_tree(pid=None):
    """PIDs of all descendants of `pid` (default: this process), from /proc/*/stat."""
    pid = pid or os.getpid(); kids = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                stat = open(f"/proc/{d}/stat").read(); ppid = int(stat[stat.rindex(")") + 2:].split()[1]); kids.setdefault(ppid, []).append(int(d))
            except (OSError, ValueError):
                pass
    seen, todo = [], list(kids.get(pid, []))
    while todo:
        c = todo.pop(); seen.append(c); todo += kids.get(c, [])
    return sorted(seen)


def tree_loaded_libs():
    """{pid: libs} for this process and every descendant (vLLM EngineCore / Worker processes) -- the runtime-truth for CUDA/NCCL/Flash libs."""
    return {str(p): loaded_libs(p) for p in [os.getpid(), *process_tree()]}


def toolkit_versions():
    """CUDA Toolkit (nvcc) and pip-visible NVIDIA runtime packages, observed -- including BOTH nccl-cu12/cu13 if both are installed."""
    import importlib.metadata as md
    nv = {d.metadata["Name"]: d.version for d in md.distributions() if d.metadata["Name"] and d.metadata["Name"].lower().startswith(("nvidia-nccl", "nvidia-cuda-runtime", "nvidia-cuda-nvcc"))}
    nvcc = sh("nvcc --version | tail -1") if sh("which nvcc") else "nvcc not found"
    return {"nvcc": nvcc, "nvidia_packages": nv}


def model_revision(model):
    """Commit hash of the cached snapshot: refs/main in the HF cache (works offline), else the snapshot dir name. None if unknown."""
    try:
        from huggingface_hub import constants
        d = pathlib.Path(constants.HF_HUB_CACHE) / ("models--" + model.replace("/", "--"))
        ref = d / "refs" / "main"
        if ref.is_file():
            return ref.read_text().strip()
        snaps = sorted((d / "snapshots").iterdir()) if (d / "snapshots").is_dir() else []
        return snaps[-1].name if snaps else None
    except Exception:  # noqa: BLE001
        return None


def insufficient(results, name, detail=""):
    results.append({"check": name, "status": "INSUFFICIENT_EVIDENCE", "detail": str(detail)})
    print(f"INSUFFICIENT_EVIDENCE {name} -- {detail}")


def proc_detail(pid):
    """{pid, state, cmd} from /proc (state 'Z' = zombie awaiting reap). Never raises."""
    try:
        stat = open(f"/proc/{pid}/stat").read(); state = stat[stat.rindex(")") + 2:].split()[0]
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()[:160]
        return {"pid": pid, "state": state, "cmd": cmd}
    except OSError:
        return {"pid": pid, "state": "gone", "cmd": ""}


def live_engine_children(pids):
    """Children that are genuinely alive engine processes. Excluded, and reported separately: zombies (exited, not yet reaped) and the
    multiprocessing resource_tracker helper (by design it lives until the parent exits)."""
    d = [proc_detail(p) for p in pids]
    live = [x for x in d if x["state"] not in ("Z", "gone") and "resource_tracker" not in x["cmd"]]
    return live, [x for x in d if x not in live]


def wait_children_gone(timeout=20.0):
    """Poll (instead of a fixed sleep) until no live engine child remains. Returns (live, ignored_detail, waited_seconds)."""
    t0 = time.time()
    while True:
        live, ignored = live_engine_children(process_tree())
        if not live or time.time() - t0 > timeout:
            return live, ignored, round(time.time() - t0, 1)
        time.sleep(0.5)
