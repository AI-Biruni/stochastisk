#!/usr/bin/env python3.13
"""Assemble release candidate rc2 from the (untouched) rc1 bundle + the repaired scripts/notebooks/tests.

rc2 changes ONLY Python scripts, notebooks, tests and docs. Both wheels are byte-identical to rc1 (asserted below), so the
native-binary / ELF / SM75 / Flash* audits are carried over from rc1 (provenance recorded; no native rebuild was performed).
usage: assemble_rc2.py        (rc1 dir and zip are never modified)"""
import ast, hashlib, json, pathlib, re, shutil, subprocess, sys, time, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "dist-rc"; RC1 = "stokastisk-0.3.0-kaggle-t4-sm75"; RC2 = RC1 + "-rc2"
S, B = D / RC1, D / RC2
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()  # noqa: E731
assert S.is_dir() and not B.exists(), "rc1 must exist and rc2 must not (never overwrite a release)"
shutil.copytree(S, B)
ign = shutil.ignore_patterns("__pycache__", "*.pyc")

# overlay the repaired sources (authoritative: repo files)
shutil.rmtree(B / "scripts/kaggle"); shutil.copytree(ROOT / "scripts/kaggle", B / "scripts/kaggle", ignore=ign)
shutil.rmtree(B / "scripts/release"); shutil.copytree(ROOT / "scripts/release", B / "scripts/release", ignore=ign)
shutil.copy2(ROOT / "scripts/make_notebooks.py", B / "scripts")
shutil.rmtree(B / "tests"); shutil.copytree(ROOT / "tests", B / "tests", ignore=ign)
shutil.rmtree(B / "notebooks"); shutil.copytree(ROOT / "notebooks", B / "notebooks", ignore=ign)
for f in ("README.md", "INSTALL.md", "BUILD.md", "RELEASE.md"): shutil.copy2(ROOT / f, B / f)

# wheels must be byte-identical to rc1
wheels = {w.name: sha(w) for w in sorted((B / "wheels").glob("*.whl"))}
assert wheels == {w.name: sha(w) for w in sorted((S / "wheels").glob("*.whl"))}, "wheels changed"

# tests (CPU only)
import os
env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
tests = {}
for t in ("test_sdk_cpu.py", "test_installer.py", "test_notebooks.py"):
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
      "git_dirty": bool(git("status", "--porcelain")), "tests": tests, "notebook_validation": nbv, "wheels_identical_to_rc1": wheels,
      "03_script_sha256": sha(B / "scripts/kaggle/03_single_gpu.py"),
      "kaggle_acceptance": {"01": "PASSED on rc1 (user-executed)", "02": "PASSED on rc1 (user-executed)", "03": "FAILED on rc1: ModuleNotFoundError stokastisk (fresh session, no install); rc2 UNTESTED", "04": "PENDING", "05": "PENDING", "06": "PENDING"}}
(B / "evidence/rc2-test-results.json").write_text(json.dumps(ev, indent=1))
(B / "evidence/notebook-validation.json").write_text(json.dumps({"timestamp": now, "kind": "static (JSON + syntax). NOT executed.", "notebooks": nbv, "execution_status": "PENDING USER on Kaggle"}, indent=1))

rm = json.loads((S / "manifests/RELEASE-MANIFEST.json").read_text())
rm.update(name=RC2, timestamp=now, supersedes=RC1, status="RELEASE CANDIDATE rc2 — LOCALLY TESTED, KAGGLE 03-06 PENDING USER EXECUTION",
          changes=["03_single_gpu.py: per-stage isolation + tracebacks, per-extension/per-GPU checks, recursive ELF audit, readelf required, cuobjdump limitation reported as WARN",
                   "_common.py: require_installed() guard, WARN status, sha256()", "notebooks 03-06: idempotent install cell + script/wheel SHA256 provenance cell",
                   "tests/test_notebooks.py added", "wheels, native binaries and their audits: unchanged from rc1 (carried over; no native rebuild)"],
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
