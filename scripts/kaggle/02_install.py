#!/usr/bin/env python3
"""Stage 02: SAFE offline-first installer for Stokastisk 0.3.0 (Kaggle dual T4, Python 3.13, torch 2.11.0+cu130).

Design (root causes of the failed first release are called out):
 * NO dependency resolver is ever run against the Kaggle environment. The previous installer ran `pip install <40 reqs>`;
   pip backtracked (xgrammar) and replaced transformers 5.20.2 -> 4.57.6, plus 15 other preinstalled packages, and pulled
   flashinfer-python + CUDA 13.2 compiler wheels. Here: both wheels `--no-deps`, then ONLY the exact pins in
   requirements/kaggle-t4-core.txt, also `--no-deps`.
 * Every decision is made BEFORE installing from a before-snapshot; any unsatisfied requirement aborts untouched.
 * After installing, every preinstalled distribution must be byte-for-byte the same version; otherwise FAIL.
 * Idempotent: a second run finds everything satisfied and calls pip zero times.
usage: 02_install.py [BUNDLE_DIR] [--dry-run] [--report PATH]"""
import argparse, glob, hashlib, importlib.metadata as md, json, os, pathlib, platform, re, subprocess, sys, tempfile, time, traceback, zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PROTECTED_EXACT = {"torch": "2.11.0+cu130", "triton": "3.6.0"}
PROHIBITED = {"flash-attn", "flashinfer", "flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache", "vllm-flash-attn",
              "cuda-tile", "nvidia-cuda-tileiras", "humming-kernels", "nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base",
              "nvidia-cutlass-dsl-libs-cu13", "quack-kernels", "tokenspeed-mla", "tokenspeed-triton"}
PROHIBITED_WHEEL_PATHS = ("vllm_flash_attn", "flashinfer", "_vllm_fa", "flashmla")
norm = lambda s: re.sub(r"[-_.]+", "-", s).lower()  # noqa: E731


# ---------------- pure decision functions (unit-tested in tests/test_installer.py) ----------------
def snapshot():
    return {norm(d.metadata["Name"]): d.version for d in md.distributions() if d.metadata["Name"]}


def parse_pins(text):
    pins = {}
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if line:
            n, _, v = line.partition("==")
            pins[norm(n)] = v.strip()
    return pins


def vllm_core_requirements(metadata_text):
    """Non-extra Requires-Dist of the wheel -> list of packaging Requirement."""
    from packaging.requirements import Requirement
    out = []
    for l in metadata_text.splitlines():
        if l.startswith("Requires-Dist: "):
            r = Requirement(l[15:])
            if r.marker is None or r.marker.evaluate({"extra": ""}):
                if not (r.marker and "extra" in str(r.marker)):
                    out.append(r)
    return out


def make_plan(installed, pins, requirements):
    """Return {install:{name:ver}, satisfied:[...], conflicts:[...]}. Never modifies anything."""
    from packaging.version import Version
    plan = {"install": {}, "satisfied": [], "compatible_kept": [], "conflicts": []}
    for n, v in pins.items():
        if n in PROHIBITED:
            plan["conflicts"].append(f"core pin {n} is prohibited")
        elif installed.get(n) == v:
            plan["satisfied"].append(f"{n}=={v}")
        elif n not in installed:
            plan["install"][n] = v
        else:
            plan["compatible_kept"].append(f"{n} {installed[n]} (pin {v}; kept, will be checked against requirement)")
    for r in requirements:
        n = norm(r.name)
        if n in PROHIBITED:
            plan["conflicts"].append(f"wheel requires prohibited package {n}")
        have = installed.get(n) or (pins.get(n) if n in plan["install"] else None)
        if have is None:
            plan["conflicts"].append(f"requirement {r} is not installed and not in the core pin list")
        elif r.specifier and not r.specifier.contains(Version(have), prereleases=True):
            plan["conflicts"].append(f"requirement {r} not satisfied by {have}")
    return plan


def check_baseline(installed):
    errs = []
    for n, v in PROTECTED_EXACT.items():
        if installed.get(n) != v:
            errs.append(f"{n} must be {v}, found {installed.get(n)}")
    tv = installed.get("transformers")
    if tv is None:
        errs.append("transformers not installed")
    else:
        from packaging.version import Version
        if Version(tv).major < 5 or Version(tv) < Version("5.5.3"):
            errs.append(f"transformers {tv} rejected: vLLM 0.25.1 needs >=5.5.3 (Transformers 4.x removed in vLLM 0.24)")
    bad = sorted(PROHIBITED & set(installed))
    if bad:
        errs.append(f"prohibited packages present: {bad}")
    return errs


def diff_snapshots(before, after, expected_new):
    changed = {n: (before[n], after.get(n)) for n in before if after.get(n) != before[n]}
    added = sorted(set(after) - set(before))
    return {"preinstalled_changed_or_removed": changed, "added": added,
            "unexpected_added": sorted(set(added) - {norm(x) for x in expected_new}),
            "ok": not changed and not (set(added) - {norm(x) for x in expected_new})}


def inspect_wheel(path):
    z = zipfile.ZipFile(path); bad = [n for n in z.namelist() if any(k in n.lower() for k in PROHIBITED_WHEEL_PATHS)
                                      and not n.endswith(".py") ]
    pyflash = [n for n in z.namelist() if "vllm_flash_attn" in n]
    meta = next((z.read(n).decode() for n in z.namelist() if n.endswith(".dist-info/METADATA")), "")
    return {"prohibited_binaries_or_dirs": bad + pyflash, "metadata": meta}


# ---------------- driver ----------------
def find_bundle(arg):
    cands = [arg] if arg else []
    cands += [os.environ.get("STOKASTISK_BUNDLE", "")]
    cands += [str(pathlib.Path(p).parent.parent) for p in glob.glob("/kaggle/input/**/manifests/SHA256SUMS", recursive=True)]
    cands += [str(pathlib.Path(__file__).resolve().parents[2])]
    for c in cands:
        if c and (pathlib.Path(c) / "manifests" / "SHA256SUMS").is_file():
            return pathlib.Path(c)
    sys.exit("bundle not found: pass its directory (containing manifests/SHA256SUMS)")


def pip(*a, **kw):
    return subprocess.run([sys.executable, "-m", "pip", *a], capture_output=True, text=True, **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("bundle", nargs="?"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report"); a = ap.parse_args(argv)
    from _common import OUT, check, finish
    R = []; rep = {"started": time.strftime("%FT%T%z"), "dry_run": a.dry_run}; t0 = time.time()
    b = find_bundle(a.bundle); rep["bundle"] = str(b)
    bad = [n for h, n in (l.split(None, 1) for l in (b / "manifests" / "SHA256SUMS").read_text().splitlines())
           if not (b / n.strip().lstrip("*")).is_file() or hashlib.sha256((b / n.strip().lstrip("*")).read_bytes()).hexdigest() != h]
    check(R, "SHA256SUMS verified before any install", not bad, bad or "all files match")
    if bad: finish("02_install", R, rep)
    check(R, "python 3.13", sys.version_info[:2] == (3, 13), platform.python_version())
    before = snapshot(); rep["before_count"] = len(before)
    (OUT / "02_packages_before.json").write_text(json.dumps(before, indent=0, sort_keys=True))
    errs = check_baseline(before)
    for e in errs: check(R, "baseline: " + e, False)
    check(R, "baseline protected (torch/triton exact, transformers>=5.5.3, no prohibited pkgs)", not errs, {k: before.get(k) for k in ("torch", "triton", "transformers", "numpy", "nvidia-cuda-nvcc", "nvidia-nvvm")})
    try:
        import torch
        n = torch.cuda.device_count(); caps = [torch.cuda.get_device_capability(i) for i in range(n)]
        ok = (n == 2 and all(c == (7, 5) for c in caps)) or a.dry_run
        check(R, "two Tesla T4 (SM75)" + (" [dry-run: not enforced]" if a.dry_run and not (n == 2) else ""), ok, f"{n} gpu(s) {caps}")
    except Exception as e:
        check(R, "torch/cuda probe", a.dry_run, repr(e))
    wheels = sorted(glob.glob(str(b / "wheels" / "*.whl")))
    check(R, "exactly two wheels", len(wheels) == 2, [os.path.basename(w) for w in wheels])
    vw = next((w for w in wheels if os.path.basename(w).startswith("vllm-")), None)
    sw = next((w for w in wheels if os.path.basename(w).startswith("stokastisk-")), None)
    wi = {w: inspect_wheel(w) for w in wheels}
    badbin = {os.path.basename(w): i["prohibited_binaries_or_dirs"] for w, i in wi.items() if i["prohibited_binaries_or_dirs"]}
    check(R, "wheels contain no FlashAttention/FlashInfer/FlashMLA files", not badbin, badbin or "none")
    pins = parse_pins((b / "requirements" / "kaggle-t4-core.txt").read_text())
    reqs = vllm_core_requirements(wi[vw]["metadata"]) if vw else []
    plan = make_plan(before, pins, reqs); rep["plan"] = plan
    for c in plan["conflicts"]: check(R, "plan: " + c, False)
    check(R, "dependency plan has no conflicts (nothing is installed if this fails)", not plan["conflicts"], {"install": plan["install"]})
    (OUT / "02_install_plan.json").write_text(json.dumps(plan, indent=1))
    if not all(r["status"] == "PASS" for r in R):
        rep["seconds"] = time.time() - t0; finish("02_install", R, rep)
    # idempotence: wheels already installed?
    def wheel_installed(name, w):
        try:
            return md.distribution(name).read_text("METADATA") == wi[w]["metadata"]
        except md.PackageNotFoundError:
            return False
    wheel_todo = [w for n, w in (("stokastisk", sw), ("vllm", vw)) if w and not wheel_installed(n, w)]
    rep["wheels_to_install"] = [os.path.basename(w) for w in wheel_todo]
    dl_bytes = 0; tmp = pathlib.Path(tempfile.mkdtemp(prefix="stk-pins-"))
    if plan["install"]:
        specs = [f"{n}=={v}" for n, v in plan["install"].items()]
        r = pip("download", "--no-deps", "-d", str(tmp), *specs); rep["pip_download_rc"] = r.returncode
        check(R, "download exact pins (no resolver, --no-deps)", r.returncode == 0, r.stderr[-300:] if r.returncode else f"{len(specs)} pins")
        dl_bytes = sum(p.stat().st_size for p in tmp.iterdir()); rep["downloaded_bytes"] = dl_bytes
    if a.dry_run:
        rep["would_install"] = {"wheels": rep["wheels_to_install"], "pins": plan["install"]}
        rep["seconds"] = time.time() - t0; finish("02_install_dryrun", R, rep)
        return
    if wheel_todo:
        r = pip("install", "--no-deps", "--no-index", *wheel_todo); check(R, "install wheels --no-deps", r.returncode == 0, r.stderr[-400:])
    if plan["install"]:
        r = pip("install", "--no-deps", "--no-index", "--find-links", str(tmp), *[f"{n}=={v}" for n, v in plan["install"].items()])
        check(R, "install core pins --no-deps", r.returncode == 0, r.stderr[-400:])
    after = snapshot(); (OUT / "02_packages_after.json").write_text(json.dumps(after, indent=0, sort_keys=True))
    d = diff_snapshots(before, after, ["stokastisk", "vllm", *plan["install"]]); rep["diff"] = d
    check(R, "no preinstalled package changed or removed", not d["preinstalled_changed_or_removed"], d["preinstalled_changed_or_removed"] or "identical")
    check(R, "only expected packages were added", not d["unexpected_added"], d["added"])
    errs = check_baseline(after); check(R, "protected baseline still intact", not errs, errs or "ok")
    # pip check: only NEW conflicts count (Kaggle has pre-existing unrelated ones)
    pc = pip("check").stdout.strip().splitlines(); rep["pip_check"] = pc
    mine = [l for l in pc if re.match(r"(vllm|stokastisk) ", l)]
    check(R, "pip check: no conflicts involving vllm/stokastisk", not mine, mine or f"{len(pc)} unrelated pre-existing lines")
    imports = ["transformers", "tokenizers", "vllm", "stokastisk", "vllm._C_stable_libtorch", "vllm._moe_C_stable_libtorch", "vllm.cumem_allocator",
               "vllm.spinloop", "vllm.fs_io_C", "vllm.engine.arg_utils", "vllm.v1.attention.backends.registry", "vllm.v1.attention.selector",
               "vllm.v1.attention.backends.triton_attn", "vllm.v1.engine.llm_engine", "vllm.v1.executor.multiproc_executor",
               "vllm.distributed.device_communicators.cuda_communicator", "vllm.model_executor.models.qwen2", "vllm.entrypoints.llm"]
    tb = {}
    for m in imports:
        try: __import__(m)
        except BaseException:
            tb[m] = traceback.format_exc()[-1500:]
    check(R, "imports (full traceback captured on failure)", not tb, tb or f"{len(imports)} modules")
    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        cls = AttentionBackendEnum.TRITON_ATTN.get_class(); check(R, "TRITON_ATTN backend class resolves", cls.__name__ == "TritonAttentionBackend", cls.__name__)
    except BaseException:
        check(R, "TRITON_ATTN backend class resolves", False, traceback.format_exc()[-800:])
    import transformers
    check(R, "transformers is 5.x and unchanged", transformers.__version__ == before["transformers"] and int(transformers.__version__.split(".")[0]) >= 5, transformers.__version__)
    import stokastisk
    pr = stokastisk.prohibited_report(); rep["prohibited"] = pr
    check(R, "no FlashAttention/FlashInfer distributions, modules or binaries installed", pr["clean"], pr)
    check(R, "versions", stokastisk.__version__ == "0.3.0", stokastisk.__version__)
    rep["seconds"] = time.time() - t0; rep["downloaded_bytes"] = dl_bytes
    finish("02_install", R, rep)


if __name__ == "__main__":
    main()
