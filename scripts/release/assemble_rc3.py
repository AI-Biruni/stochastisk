#!/usr/bin/env python3.13
"""Assemble release candidate rc3 from the (untouched) rc2 bundle + repaired scripts/notebooks/tests + a FRESHLY BUILT stokastisk wheel.

rc1/rc2 shipped a stale stokastisk wheel (nccl.py older than sdk/). rc3 rebuilds that pure-Python wheel from sdk/ and asserts byte-equality of
its contents with sdk/. The vLLM wheel is byte-identical to rc1/rc2 (asserted), so native-binary / ELF / SM75 / Flash* audits are carried over
(no native rebuild was performed).
usage: assemble_rc3.py        (rc1/rc2 dirs and zips are never modified)"""
import ast, hashlib, json, pathlib, re, shutil, subprocess, sys, time, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "dist-rc"; RC1 = "stokastisk-0.3.0-kaggle-t4-sm75-rc2"; RC2 = "stokastisk-0.3.0-kaggle-t4-sm75-rc3"
S, B = D / RC1, D / RC2
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()  # noqa: E731
assert S.is_dir() and not B.exists(), "rc2 must exist and rc3 must not (never overwrite a release)"
shutil.copytree(S, B)
ign = shutil.ignore_patterns("__pycache__", "*.pyc")

# overlay the repaired sources (authoritative: repo files)
shutil.rmtree(B / "scripts/kaggle"); shutil.copytree(ROOT / "scripts/kaggle", B / "scripts/kaggle", ignore=ign)
shutil.rmtree(B / "scripts/release"); shutil.copytree(ROOT / "scripts/release", B / "scripts/release", ignore=ign)
shutil.rmtree(B / "scripts/audit", ignore_errors=True); shutil.copytree(ROOT / "scripts/audit", B / "scripts/audit", ignore=ign)
shutil.copy2(ROOT / "scripts/make_notebooks.py", B / "scripts")
shutil.rmtree(B / "tests"); shutil.copytree(ROOT / "tests", B / "tests", ignore=ign)
shutil.rmtree(B / "notebooks"); shutil.copytree(ROOT / "notebooks", B / "notebooks", ignore=ign)
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
wheels = {w.name: sha(w) for w in sorted((B / "wheels").glob("*.whl"))}
vname = next(n for n in wheels if n.startswith("vllm-")); assert wheels[vname] == base_w[vname], "vllm wheel changed"
sname = next(n for n in wheels if n.startswith("stokastisk-")); assert wheels[sname] != base_w[sname], "stokastisk wheel unexpectedly unchanged"
# evidence: namespace audit + wheel audit
(B / "evidence").mkdir(exist_ok=True)
r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/audit/namespace_audit.py"), str(ROOT / "stage.zip"), str(B / "evidence"), *map(str, sorted((B / "wheels").glob("*.whl")))], capture_output=True, text=True)
assert r.returncode == 0, r.stderr
(B / "evidence/wheel-vs-source.json").write_text(json.dumps({"stokastisk_wheel": sname, "sha256": wheels[sname], "previous_wheel_sha256": base_w[sname], "drift_vs_sdk": drift,
        "previous_wheel_drift_vs_sdk": wheel_drift(next((S / "wheels").glob("stokastisk-*.whl")))}, indent=1))

# tests (CPU only)
import os
env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
tests = {}
for t in ("test_sdk_cpu.py", "test_installer.py", "test_notebooks.py", "test_nccl.py", "test_wheel_matches_source.py"):
    r = subprocess.run([sys.executable, str(ROOT / "tests" / t)], capture_output=True, text=True, env=env)
    m = re.search(r"Ran (\d+) tests", r.stderr)
    tests[t] = {"returncode": r.returncode, "ran": int(m.group(1)) if m else None, "tail": r.stderr.strip().splitlines()[-2:]}
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
      "kaggle_acceptance": {"01": "PASSED on rc1 dataset (user-executed)", "02": "PASSED on rc1 dataset (user-executed)", "03": "PASSED 20/20 on Kaggle using rc1-dataset script (user-executed); rc3 script UNTESTED", "04": "ran on rc1-dataset content: NCCL collectives OK, summary FAIL due to stale SDK wheel (fixed in rc3, UNTESTED)", "05": "PENDING", "06": "PENDING"}}
(B / "evidence/rc2-test-results.json").write_text(json.dumps(ev, indent=1))
(B / "evidence/notebook-validation.json").write_text(json.dumps({"timestamp": now, "kind": "static (JSON + syntax). NOT executed.", "notebooks": nbv, "execution_status": "PENDING USER on Kaggle"}, indent=1))

rm = json.loads((S / "manifests/RELEASE-MANIFEST.json").read_text())
rm.update(name=RC2, timestamp=now, supersedes=RC1, status="RELEASE CANDIDATE rc3 — LOCALLY TESTED, KAGGLE RE-RUN OF 03-06 PENDING USER EXECUTION",
          changes=["stokastisk wheel REBUILT from sdk/ (rc1/rc2 shipped a stale stokastisk/nccl.py; the only drifted file)", "sdk/stokastisk/nccl.py: schema v2, no NCCL transport overrides, finite timeout, broadcast+barrier, teardown evidence",
                   "04_nccl_launch.py (parent, timeout, stale-output cleanup), 04_nccl_dual_gpu.py (worker), 04_nccl_summary.py rewritten; notebooks regenerated", "tests: test_nccl.py, test_wheel_matches_source.py (wheel must equal sdk/)",
                   "scripts/audit/namespace_audit.py + evidence/stage-*.json (velocita is an unused stage.zip leftover; in no wheel)", "vllm wheel, native binaries and their audits: unchanged from rc1/rc2 (carried over; no native rebuild)"],
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
