"""Stage 04 SUMMARY: judge the launch + both ranks + NCCL logs. Never reports PASS if a worker failed, the parent timed out,
a collective was wrong, a rank file is missing/incomplete, or the run was a Gloo control experiment."""
import glob, json, re, sys, zipfile
from _common import OK_STATUSES, OUT, check, save, warn

R, EXTRA = [], {}
try:
    launch = json.load(open(OUT / "04_nccl_launch.json"))
except FileNotFoundError:
    launch = None
check(R, "launcher ran and wrote 04_nccl_launch.json", launch is not None)
if launch:
    check(R, "torchrun exit code 0", launch["returncode"] == 0, f"rc={launch['returncode']}")
    check(R, "parent did not time out", not launch["timed_out"], f"timeout {launch['timeout_s']}s, took {launch['duration_s']}s")
    if launch.get("transport_overrides_in_env"):
        warn(R, "NCCL transport variables were set; result is not NCCL's default transport selection", launch["transport_overrides_in_env"])
ranks = {}
for p in sorted(glob.glob(str(OUT / "04_nccl_rank*.json"))):
    j = json.load(open(p)); ranks[j.get("rank")] = j
check(R, "exactly ranks 0 and 1 reported", sorted(ranks) == [0, 1], sorted(ranks))
tb = {}
for rk, j in sorted(ranks.items()):
    check(R, f"rank{rk} result schema_version >= 2 (installed SDK is current)", j.get("schema_version", 0) >= 2, f"schema_version={j.get('schema_version')}; an older installed stokastisk wheel lacks the required fields")
    check(R, f"rank{rk} backend is nccl (a Gloo control run is never acceptance)", j.get("backend") == "nccl" and j.get("acceptance_valid") is True, j.get("backend"))
    dev = j.get("device") if isinstance(j.get("device"), dict) else {}      # old-format files carry a plain string here
    check(R, f"rank{rk} on a Tesla T4 SM75 (local_rank {j.get('local_rank')})", "T4" in str(dev.get("name")) and dev.get("compute_capability") == [7, 5], j.get("device"))
    for op in j.get("collectives", []):
        check(R, f"rank{rk} {op['name']}", op["status"] == "PASS", {k: v for k, v in op.items() if k not in ("name",)})
    for need in ("all_reduce_sum", "all_gather", "reduce_scatter", "broadcast", "barrier", "distinct_devices"):
        check(R, f"rank{rk} ran {need}", any(o["name"] == need and o["status"] == "PASS" for o in j.get("collectives", [])), "missing or not PASS")
    check(R, f"rank{rk} process group destroyed", j.get("cleanup") == "process group destroyed" and j.get("dist_initialized_after_teardown") is False, j.get("cleanup"))
    base, after = j.get("gpu_mem_used_MiB_baseline"), j.get("gpu_mem_used_MiB_after_teardown")
    if base is None or after is None:
        check(R, f"rank{rk} GPU memory fields present", False, f"baseline={base} after={after}")
    else:
        # Recorded, not judged: no empirically validated threshold exists, so a number is never turned into PASS. A large increase is a WARNING only.
        EXTRA.setdefault("gpu_memory_after_teardown_MiB", {})[f"rank{rk}"] = {"baseline": base, "after": after, "delta": after - base, "status": "MEASURED"}
        if after - base > 256:
            warn(R, f"rank{rk} GPU memory after teardown is {after - base} MiB above its pre-init baseline (informational; no validated threshold)", f"baseline {base} MiB -> {after} MiB")
    if j.get("tracebacks"):
        tb[f"rank{rk}"] = j["tracebacks"]
save("04_nccl_tracebacks.json", tb)
save("04_nccl_collectives.json", {f"rank{k}": v.get("collectives") for k, v in ranks.items()})

# transport evidence: only what the NCCL logs literally say
logs = {p: open(p, errors="replace").read() for p in sorted(glob.glob(str(OUT / "nccl.*.log")))}
text = "".join(logs.values())
channels = sorted(set(re.findall(r"Channel \d+(?:/\d+)? : \d+\[\w+\] -> \d+\[\w+\] via (\S+)", text)))
evidence = {"log_files": list(logs), "channel_transports": channels, "transport_env_in_nccl_log": sorted(set(re.findall(r"(NCCL_\w+) set by environment to (\S+?)\.?\n", text))),
            "net_plugin": sorted(set(re.findall(r"Using network (\S+)", text))), "nccl_version_in_log": sorted(set(re.findall(r"NCCL version (\S+)", text))),
            "p2p_check_lines": sorted(set(re.findall(r"Check P2P Type [^\n]+", text)))[:4],
            "peer_access_matrix_cuda": (ranks.get(0) or {}).get("peer_access"),
            "interpretation": ("unknown: no 'Channel ... via <transport>' lines captured" if not channels else
                               "NCCL itself reports these per-channel transports (default selection unless 'transport_env_in_nccl_log' lists an override): " + ", ".join(channels)),
            "note": "CUDA peer capability, NCCL transport selection and actual direct GPU-to-GPU traffic are three different things; only the 'channel_transports' lines are NCCL's own statement."}
torch_nccl = (ranks.get(0) or {}).get("nccl")
evidence["torch_reported_nccl_version"] = torch_nccl
evidence["torch_reported_nccl_note"] = "torch.cuda.nccl.version() is the version torch was COMPILED against; 'nccl_version_in_log' is what NCCL itself printed for the library actually loaded."
evidence["loaded_libraries_per_rank"] = {f"rank{k}": v.get("loaded_libraries") for k, v in ranks.items()}
save("04_nccl_transport_evidence.json", evidence)
_logged = {v.split("+")[0] for v in evidence["nccl_version_in_log"]}
if _logged and torch_nccl and torch_nccl not in _logged:
    warn(R, "runtime NCCL differs from the NCCL torch was compiled against", f"torch compiled {torch_nccl}; NCCL log says {sorted(_logged)}. Correctness checks still ran on the runtime library.")
check(R, "NCCL logs captured (transport evidence)", bool(logs), f"{len(logs)} file(s)")
if not channels:
    warn(R, "NCCL transport could not be established from the logs", "reported as unknown")
print("NCCL transports observed:", channels or "unknown")
EXTRA.update(nccl_transports=channels, launch=launch)
status = "PASS" if R and all(r["status"] in OK_STATUSES for r in R) else "FAIL"
doc = {"stage": "04_nccl_dual_gpu", "status": status, "checks": R, **EXTRA}
save("04_nccl_summary.json", doc)
save("04_nccl_dual_gpu.json", doc)                          # consumed by 07_summary.py
with zipfile.ZipFile(OUT / "04_nccl_acceptance.zip", "w", zipfile.ZIP_DEFLATED) as z:       # evidence archive (excludes itself)
    for p in sorted(OUT.glob("*")):
        if p.is_file() and p.name != "04_nccl_acceptance.zip":
            z.write(p, p.name)
for r in R:
    print(r["status"] + " " + r["check"])
print(f"\n[04_nccl_dual_gpu] OVERALL: {status}")
sys.exit(0 if status == "PASS" else 1)
