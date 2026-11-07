#!/usr/bin/env python3.13
"""Assemble release candidate rc5 from the (untouched) rc4 bundle + repaired scripts/notebooks/tests + a FRESHLY BUILT stokastisk wheel.

rc5 rebuilds the pure-Python stokastisk wheel from sdk/ AND derives the vLLM wheel from the rc4 wheel by patch 0003 (two pure-Python default-path imports of removed FlashInfer modules); every native .so is asserted byte-identical to rc4 (no native rebuild), so native-binary / ELF / SM75 / Flash* audits are carried over
(no native rebuild was performed).
usage: assemble_rc5.py        (rc1/rc2/rc3 dirs and zips are never modified)"""
import ast, hashlib, json, os, pathlib, re, shutil, subprocess, sys, time, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "dist-rc"; RC1 = "stokastisk-0.3.0-kaggle-t4-sm75-rc4"; RC2 = "stokastisk-0.3.0-kaggle-t4-sm75-rc5"
S, B = D / RC1, D / RC2
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()  # noqa: E731
assert S.is_dir() and not B.exists(), "rc4 must exist and rc5 must not (never overwrite a release)"
shutil.copytree(S, B)
ign = shutil.ignore_patterns("__pycache__", "*.pyc")

# overlay the repaired sources (authoritative: repo files)
shutil.rmtree(B / "scripts/kaggle"); shutil.copytree(ROOT / "scripts/kaggle", B / "scripts/kaggle", ignore=ign)
shutil.rmtree(B / "scripts/release"); shutil.copytree(ROOT / "scripts/release", B / "scripts/release", ignore=ign)
shutil.rmtree(B / "scripts/audit", ignore_errors=True); shutil.copytree(ROOT / "scripts/audit", B / "scripts/audit", ignore=ign)
shutil.copy2(ROOT / "scripts/make_notebooks.py", B / "scripts")
shutil.rmtree(B / "tests"); shutil.copytree(ROOT / "tests", B / "tests", ignore=ign)
shutil.rmtree(B / "notebooks"); shutil.copytree(ROOT / "notebooks", B / "notebooks", ignore=ign)
shutil.rmtree(B / "patches", ignore_errors=True); shutil.copytree(ROOT / "patches", B / "patches", ignore=ign)
for f in ("README.md", "INSTALL.md", "BUILD.md", "RELEASE.md"): shutil.copy2(ROOT / f, B / f)

# wheels: vllm byte-identical to the base; stokastisk rebuilt from sdk/ and verified against it
import tempfile as _tf
sys.path.insert(0, str(ROOT / "tests")); from test_wheel_matches_source import wheel_drift  # noqa: E402
base_w = {w.name: sha(w) for w in sorted((S / "wheels").glob("*.whl"))}
with _tf.TemporaryDirectory() as td:
    r = subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "-o", td, str(ROOT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    nw = next(pathlib.Path(td).glob("stokastisk-*.whl"))
    drift = wheel_drift(nw); assert not drift, drift
    for old_w in (B / "wheels").glob("stokastisk-*.whl"): old_w.unlink()
    shutil.copy2(nw, B / "wheels" / nw.name)
# vLLM wheel: derived from the rc4 wheel by patches/0003 (pure Python only); native objects byte-identical (asserted by the patch script)
vname = next(n for n in base_w if n.startswith("vllm-"))
pr = subprocess.run([sys.executable, str(ROOT / "scripts/release/patch_vllm_wheel_t4.py"), str(S / "wheels" / vname), str(B / "wheels" / vname), str(B / "patches/0003-t4-flashinfer-free-default-path.patch")], capture_output=True, text=True)
assert pr.returncode == 0, pr.stderr[-1500:]
assert (B / "patches/0003-t4-flashinfer-free-default-path.patch").read_text() == (ROOT / "patches/0003-t4-flashinfer-free-default-path.patch").read_text(), "committed patch differs from regenerated patch"
patch_info = json.loads(pr.stdout)
wheels = {w.name: sha(w) for w in sorted((B / "wheels").glob("*.whl"))}
assert wheels[vname] != base_w[vname], "vllm wheel unexpectedly unchanged"
sname = next(n for n in wheels if n.startswith("stokastisk-")); assert wheels[sname] != base_w[sname], "stokastisk wheel unexpectedly unchanged"
os.environ["STOKASTISK_VLLM_WHEEL"] = str(B / "wheels" / vname)
# evidence: namespace audit + wheel audit
(B / "evidence").mkdir(exist_ok=True)
r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/audit/namespace_audit.py"), str(ROOT / "stage.zip"), str(B / "evidence"), *map(str, sorted((B / "wheels").glob("*.whl")))], capture_output=True, text=True)
assert r.returncode == 0, r.stderr
(B / "evidence/wheel-vs-source.json").write_text(json.dumps({"stokastisk_wheel": sname, "sha256": wheels[sname], "previous_wheel_sha256": base_w[sname], "drift_vs_sdk": drift,
        "previous_wheel_drift_vs_sdk": wheel_drift(next((S / "wheels").glob("stokastisk-*.whl")))}, indent=1))

# fresh native audit of the NEW wheel (CUDA 13.0 cuobjdump) + wheel lineage
na = subprocess.run([sys.executable, str(ROOT / "scripts/audit/native_audit.py"), str(B / "wheels" / vname), str(B / "evidence/native-binary-audit-rc5.json")], capture_output=True, text=True)
assert na.returncode == 0, na.stderr[-800:]
naj = json.loads((B / "evidence/native-binary-audit-rc5.json").read_text())["summary"]
assert naj["all_x86_64"] and not naj["any_needs_cuda12_soname"] and not naj["any_cuda12_toolkit_string"] and not naj["any_older_than_sm75_cubin"] and not naj["any_flash_markers"], naj
(B / "manifests/WHEEL-LINEAGE.json").write_text(json.dumps({"vllm_wheel": vname, "rc1_to_rc4_sha256": base_w[vname], "rc5_sha256": wheels[vname], "change": "patches/0003-t4-flashinfer-free-default-path.patch (pure Python: " + ", ".join(patch_info["changed"]) + ")",
    "native_objects_identical_to_rc4": patch_info["native_files_identical"], "native_rebuild": False,
    "stale_hash_evidence_note": "evidence/*.json carried over from rc1-rc4 may quote the rc1-rc4 vLLM wheel hash above; they describe the native build, which is unchanged. The authoritative hash for rc5 is manifests/SHA256SUMS and RELEASE-MANIFEST.json."}, indent=1))

# tests (CPU only)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
tests = {}
for t in ("test_sdk_cpu.py", "test_installer.py", "test_notebooks.py", "test_nccl.py", "test_wheel_matches_source.py", "test_notebook05_spawn.py", "test_flash_free_default_path.py"):
    r = subprocess.run([sys.executable, str(ROOT / "tests" / t)], capture_output=True, text=True, env=env)
    m = re.search(r"Ran (\d+) tests", r.stderr)
    sk = re.search(r"skipped=(\d+)", r.stderr)
    tests[t] = {"returncode": r.returncode, "ran": int(m.group(1)) if m else None, "skipped": int(sk.group(1)) if sk else 0, "tail": r.stderr.strip().splitlines()[-2:]}
assert all(v["skipped"] == 0 for v in tests.values()), f"skipped tests would be reported as NOT_RUN, not PASS: {tests}"
assert all(v["returncode"] == 0 for v in tests.values()), tests

# notebook structural validation
nbv = []
for f in sorted((B / "notebooks").glob("*.ipynb")):
    j = json.loads(f.read_text()); ok = True
    for c in j["cells"]:
        if c["cell_type"] == "code":
            try: ast.parse("\n".join(l for l in "".join(c["source"]).splitlines() if not l.startswith("!")))
            except SyntaxError: ok = False
    nbv.append({"notebook": f.name, "valid_json": True, "cells": len(j["cells"]), "code_cells_parse": ok, "outputs_present": any(c.get("outputs") for c in j["cells"] if c["cell_type"] == "code")})
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()  # noqa: E731
now = time.strftime("%FT%T%z")
ev = {"timestamp": now, "kind": "local CPU tests + static checks; NOT Kaggle execution", "git_head_at_build": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
      "git_dirty": bool(git("status", "--porcelain")), "tests": tests, "notebook_validation": nbv, "wheels": wheels,
      "03_script_sha256": sha(B / "scripts/kaggle/03_single_gpu.py"),
      "kaggle_acceptance": {"01": "PASSED (user-executed, rc3 dataset)", "02": "PASSED (user-executed, rc4 dataset)", "03": "PASSED (user-executed, rc3 dataset)", "04": "PASSED with WARNING (user-executed, rc4 dataset): collectives+teardown OK; NCCL runtime 2.32.3+cuda12.9 loaded from Kaggle's own nvidia-nccl-cu12 (see docs)", "05": "FAILED on rc4 (TP1: no vllm.v1.attention.backends.flashinfer; TP2: no ...flashinfer_all_reduce). rc5 fix UNTESTED on Kaggle", "06": "NOT_RUN"}}
(B / "evidence/rc5-test-results.json").write_text(json.dumps(ev, indent=1))
(B / "evidence/notebook-validation.json").write_text(json.dumps({"timestamp": now, "kind": "static (JSON + syntax). NOT executed.", "notebooks": nbv, "execution_status": "PENDING USER on Kaggle"}, indent=1))

rm = json.loads((S / "manifests/RELEASE-MANIFEST.json").read_text())
rm.update(name=RC2, timestamp=now, supersedes=RC1, status="RELEASE CANDIDATE rc5 — LOCALLY TESTED; KAGGLE NOTEBOOKS 05 AND 06 NOT YET EXECUTED WITH THIS FIX. NOT PRODUCTION-READY",
          changes=["vLLM wheel patch 0003: sampler and CudaCommunicator no longer import FlashInfer modules that this Flash-free build does not ship (rc4 Kaggle 05 root causes, TP1 and TP2)",
                   "sdk config: VLLM_USE_FLASHINFER_SAMPLER=0 and VLLM_ALLREDUCE_USE_FLASHINFER=0 pinned (user values win)",
                   "notebooks: explicit release identity + manifest-name check (no hits[0])", "05/06: main()->int + freeze_support, loaded-library (/proc maps) evidence, model revision, shutdown check, INSUFFICIENT_EVIDENCE status",
                   "04: GPU-memory number recorded as MEASURED evidence (no arbitrary PASS threshold); per-rank loaded libraries recorded",
                   "tests/test_flash_free_default_path.py; scripts/audit/native_audit.py; vLLM native objects byte-identical to rc4 (NO native rebuild)"],
          wheel_lineage="manifests/WHEEL-LINEAGE.json", native_audit_rc5=naj,
          kaggle_acceptance=ev["kaggle_acceptance"], wheels={n: {"size": (B / "wheels" / n).stat().st_size, "sha256": h} for n, h in wheels.items()})
for p in (B / "manifests/RELEASE-MANIFEST.json", B / "evidence/RELEASE-MANIFEST.json"): p.write_text(json.dumps(rm, indent=1))

# checksums
(B / "manifests/SHA256SUMS").unlink()
files = sorted(p for p in B.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
(B / "manifests/SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(B)}\n" for p in files))

zp = D / f"{RC2}.zip"
assert not zp.exists()
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in sorted(B.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            zi = zipfile.ZipInfo(f"{RC2}/{p.relative_to(B)}", (2026, 11, 7, 0, 0, 0)); zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = (0o755 if p.suffix in (".py", ".sh") else 0o644) << 16; z.writestr(zi, p.read_bytes())
# verify from a fresh extraction
import tempfile
with tempfile.TemporaryDirectory() as td:
    zipfile.ZipFile(zp).extractall(td); root = pathlib.Path(td) / RC2
    bad = [l for l in (root / "manifests/SHA256SUMS").read_text().splitlines() if sha(root / l.split(None, 1)[1].strip()) != l.split()[0]]
    names = zipfile.ZipFile(zp).namelist()
    forbidden = [n for n in names if any(k in n.lower() for k in ("tokens.env", ".env", ".git/", "__pycache__", ".pyc", "id_rsa", "notebooks-executed", "flashinfer-python", "vllm_flash_attn"))]
    secrets = [n for n in names if n.endswith((".py", ".md", ".json", ".txt", ".ipynb", ".sh", ".patch")) and re.search("gh" + "p_|github" + "_pat_|hf" + "_[A-Za-z0-9]{20}", zipfile.ZipFile(zp).read(n).decode(errors="ignore"))]
    nbs = sorted(p.name for p in (root / "notebooks").glob("*.ipynb"))
    sc = sha(root / "scripts/kaggle/03_single_gpu.py") == sha(ROOT / "scripts/kaggle/03_single_gpu.py")
integ = {"zip": zp.name, "size": zp.stat().st_size, "sha256": sha(zp), "testzip": zipfile.ZipFile(zp).testzip(), "entries": len(names), "hash_mismatches": bad,
         "forbidden": forbidden, "secret_hits": secrets, "notebooks": nbs, "script03_matches_repo": sc}
assert not (bad or forbidden or secrets) and len(nbs) == 6 and sc, integ
(D / f"{RC2}.zip.sha256").write_text(f"{sha(zp)}  {zp.name}\n")
print(json.dumps(integ, indent=1))
