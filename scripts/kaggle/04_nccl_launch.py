"""Stage 04 LAUNCHER (parent). One process per GPU via torch.distributed.run --standalone (it picks a free rendezvous port itself).

* stale outputs of earlier runs are removed first, so a rerun can never be judged on old JSON;
* finite parent timeout (STOKASTISK_NCCL_PARENT_TIMEOUT_S, default 300 s); on timeout the whole process group is killed;
* full stdout+stderr of torchrun and every NCCL log are preserved; NCCL transport selection is NOT overridden;
* the parent never initialises CUDA (so it does not hold GPU memory while the workers run)."""
import datetime, glob, json, os, pathlib, signal, subprocess, sys, time
from _common import OUT, save, sha256

HERE = pathlib.Path(__file__).resolve().parent
TIMEOUT = float(os.environ.get("STOKASTISK_NCCL_PARENT_TIMEOUT_S", "300"))
for pat in ("04_nccl_*.json", "04_nccl_*.log", "04_nccl_acceptance.zip", "nccl.*.log"):
    for p in glob.glob(str(OUT / pat)):
        os.remove(p)
PLOG = open(OUT / "04_nccl_parent.log", "w", buffering=1)


def log(msg):
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True); PLOG.write(line + "\n")


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return {"cmd": cmd, "rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr}


worker = HERE / "04_nccl_dual_gpu.py"
log(f"worker {worker} sha256={sha256(worker)}")
probe = ("import json,platform,torch;print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
         "'nccl':list(torch.cuda.nccl.version()),'device_count':torch.cuda.device_count()}))")
env_info = {"torch_probe": sh([sys.executable, "-c", probe]), "nvidia_smi": sh(["nvidia-smi"]),
            "nvidia_smi_query": sh(["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,compute_cap", "--format=csv"])}
save("04_nccl_environment.json", env_info)
save("04_nccl_topology.json", {"nvidia_smi_topo_m": sh(["nvidia-smi", "topo", "-m"])})

env = dict(os.environ, PYTHONPATH=str(HERE) + ":" + os.environ.get("PYTHONPATH", ""))
for k, v in (("NCCL_DEBUG", "INFO"), ("NCCL_DEBUG_SUBSYS", "INIT,ENV,GRAPH,NET,P2P,SHM,COLL"), ("TORCH_DISTRIBUTED_DEBUG", "DETAIL")):
    env.setdefault(k, v)
env["NCCL_DEBUG_FILE"] = str(OUT / "nccl.%h.%p.log")           # unique per host+pid, so ranks never overwrite each other
overrides = {k: env[k] for k in ("NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME", "NCCL_P2P_LEVEL", "NCCL_NET") if k in env}
if overrides:
    log(f"WARNING: transport-affecting NCCL variables are set in the environment -> this is NOT NCCL's default selection: {overrides}")
cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2", "--max_restarts=0", str(worker)]
log("command: " + " ".join(cmd))
t0 = time.time(); timed_out = False
with open(OUT / "04_nccl_torchrun.log", "w") as out:
    p = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        rc = p.wait(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        timed_out = True; log(f"TIMEOUT after {TIMEOUT}s: killing process group {p.pid}")
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                break
            try:
                rc = p.wait(timeout=10); break
            except subprocess.TimeoutExpired:
                continue
        else:
            rc = None
        rc = p.poll() if rc is None else rc
dur = round(time.time() - t0, 2)
log(f"torchrun finished rc={rc} timed_out={timed_out} duration={dur}s")
save("04_nccl_launch.json", {"cmd": cmd, "returncode": rc, "timed_out": timed_out, "timeout_s": TIMEOUT, "duration_s": dur, "transport_overrides_in_env": overrides,
                             "worker_sha256": sha256(worker), "control_mode": env.get("STOKASTISK_NCCL_CONTROL")})
sys.exit(0 if (rc == 0 and not timed_out) else 1)
