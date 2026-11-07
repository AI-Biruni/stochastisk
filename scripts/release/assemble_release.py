#!/usr/bin/env python3.13
"""Assemble a release candidate bundle from the (untouched) base bundle + current repo scripts/notebooks/tests/docs.
The vLLM wheel (all native code) is carried over BYTE-IDENTICAL from the base (asserted); the stokastisk wheel is rebuilt from sdk/. Never overwrites an existing release; verifies the final ZIP
from a fresh extraction. usage: assemble_release.py NEW_NAME BASE_NAME
  e.g. assemble_release.py stokastisk-0.3.0-kaggle-t4-sm75-rc6 stokastisk-0.3.0-kaggle-t4-sm75-rc5"""
import ast, hashlib, json, os, pathlib, re, shutil, subprocess, sys, time, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "dist-rc"; RC2, RC1 = sys.argv[1], sys.argv[2]            # new release, base release (never modified)
S, B = D / RC1, D / RC2
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()  # noqa: E731
assert S.is_dir() and not B.exists(), "base release must exist and the new one must not (never overwrite a release)"
shutil.copytree(S, B)
ign = shutil.ignore_patterns("__pycache__", "*.pyc")

# overlay the repaired sources (authoritative: repo files)
shutil.rmtree(B / "scripts/kaggle"); shutil.copytree(ROOT / "scripts/kaggle", B / "scripts/kaggle", ignore=ign)
shutil.rmtree(B / "scripts/release"); shutil.copytree(ROOT / "scripts/release", B / "scripts/release", ignore=ign)
shutil.rmtree(B / "scripts/audit", ignore_errors=True); shutil.copytree(ROOT / "scripts/audit", B / "scripts/audit", ignore=ign)
shutil.copy2(ROOT / "scripts/make_notebooks.py", B / "scripts"); shutil.copy2(ROOT / "scripts/fetch_release.py", B / "scripts")
shutil.rmtree(B / "tests"); shutil.copytree(ROOT / "tests", B / "tests", ignore=ign)
shutil.rmtree(B / "notebooks"); shutil.copytree(ROOT / "notebooks", B / "notebooks", ignore=ign)
shutil.rmtree(B / "patches", ignore_errors=True); shutil.copytree(ROOT / "patches", B / "patches", ignore=ign)
for f in ("README.md", "INSTALL.md", "BUILD.md", "RELEASE.md"): shutil.copy2(ROOT / f, B / f)

# wheels: vllm byte-identical to the base; stokastisk rebuilt from sdk/ and verified against it
import tempfile as _tf
sys.path.insert(0, str(ROOT / "tests")); from test_wheel_matches_source import wheel_drift  # noqa: E402
base_w = {w.name: sha(w) for w in sorted((S / "wheels").glob("*.whl"))}
# vLLM wheel (all native code): byte-identical to the base. stokastisk wheel (pure Python): rebuilt from sdk/ and checked against it.
import tempfile as _tf
sys.path.insert(0, str(ROOT / "tests")); from test_wheel_matches_source import wheel_drift  # noqa: E402
vname = next(n for n in base_w if n.startswith("vllm-")); sname = next(n for n in base_w if n.startswith("stokastisk-"))
shutil.copy2(S / "wheels" / vname, B / "wheels" / vname)
with _tf.TemporaryDirectory() as td:
    r = subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "-o", td, str(ROOT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    nw = next(pathlib.Path(td).glob("stokastisk-*.whl")); drift = wheel_drift(nw); assert not drift, drift
    for old_w in (B / "wheels").glob("stokastisk-*.whl"): old_w.unlink()
    shutil.copy2(nw, B / "wheels" / nw.name)
wheels = {w.name: sha(w) for w in sorted((B / "wheels").glob("*.whl"))}
assert wheels[vname] == base_w[vname], "vllm wheel must be byte-identical to the base release"
patch_info = {"changed": ["(carried over from the base release)"], "native_files_identical": "see native-binary-audit.json"}
assert (B / "patches/0003-t4-flashinfer-free-default-path.patch").read_text() == (ROOT / "patches/0003-t4-flashinfer-free-default-path.patch").read_text()
os.environ["STOKASTISK_VLLM_WHEEL"] = str(B / "wheels" / vname)
# evidence: namespace audit + wheel audit
(B / "evidence").mkdir(exist_ok=True)
ns = {w.name: [n for n in zipfile.ZipFile(w).namelist() if re.search(r"velocita|stochastisk", n, re.I)] for w in (B / "wheels").glob("*.whl")}
assert not any(ns.values()), f"obsolete/misspelled namespace inside a wheel: {ns}"
(B / "evidence/namespace-audit.json").write_text(json.dumps({"canonical_package": "stokastisk", "wheel_members_with_obsolete_or_misspelled_names": ns}, indent=1))
(B / "evidence/wheel-vs-source.json").write_text(json.dumps({"stokastisk_wheel": sname, "sha256": wheels[sname], "previous_wheel_sha256": base_w[sname], "drift_vs_sdk": drift}, indent=1))

# fresh native audit of the NEW wheel (CUDA 13.0 cuobjdump) + wheel lineage
na = subprocess.run([sys.executable, str(ROOT / "scripts/audit/native_audit.py"), str(B / "wheels" / vname), str(B / "evidence/native-binary-audit.json")], capture_output=True, text=True)
assert na.returncode == 0, na.stderr[-800:]
naj = json.loads((B / "evidence/native-binary-audit.json").read_text())["summary"]
assert naj["all_x86_64"] and not naj["any_needs_cuda12_soname"] and not naj["any_cuda12_toolkit_string"] and not naj["any_older_than_sm75_cubin"] and not naj["any_flash_markers"], naj
# manifests/WHEEL-LINEAGE.json is inherited unchanged from the base release (same wheels).

# tests (CPU only)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
tests = {}
for t in ("test_sdk_cpu.py", "test_installer.py", "test_notebooks.py", "test_nccl.py", "test_wheel_matches_source.py", "test_notebook05_spawn.py", "test_flash_free_default_path.py", "test_kaggle_limits.py", "test_fetch_release.py"):
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
      "kaggle_acceptance": {"01": "PASSED (user-executed, earlier release)", "02": "PASSED (user-executed)", "03": "PASSED (user-executed, earlier release)", "04": "PASSED with WARNING on rc5 (user-executed): collectives+teardown OK; runtime NCCL 2.32.3 (the cu12-built package) comes from Kaggle's own nvidia-nccl-cu12", "05": "rc5: TP1 and TP2 completed real TRITON_ATTN generation; ONLY the (since corrected) shutdown check failed. This release UNTESTED on Kaggle", "06": "PASSED on rc5 (user-executed): TP1 1032.9 tok/s, TP2 545.1 tok/s (0.53x, measured). This release UNTESTED on Kaggle"}}
(B / "evidence/test-results.json").write_text(json.dumps(ev, indent=1))
(B / "evidence/notebook-validation.json").write_text(json.dumps({"timestamp": now, "kind": "static (JSON + syntax). NOT executed.", "notebooks": nbv, "execution_status": "PENDING USER on Kaggle"}, indent=1))

rm = json.loads((S / "manifests/RELEASE-MANIFEST.json").read_text())
rm.update(name=RC2, timestamp=now, supersedes=RC1, status="RELEASE CANDIDATE — LOCALLY TESTED. Kaggle 04 and 06 passed on the previous candidate, 05 completed inference there; this exact bundle has NOT been executed on Kaggle. NOT production-ready until notebooks 01-06 pass on it",
          changes=["shutdown check made state-aware (zombies / multiprocessing resource_tracker reported separately, never counted as live engine processes)",
                   "model_revision read from the HF cache refs/main", "stokastisk.limits: Kaggle hard limits (4 CPU, 30 GiB RAM, 20 GiB /kaggle/working, 2xT4 SM75 15 GiB) enforced/reported; CPU thread budget split across TP workers",
                   "scripts/fetch_release.py: stdlib HF CDN downloader (parallel ranges, SHA256-verified); notebooks fall back to it when the dataset is not attached",
                   "repository cleanup: stale evidence/manifests/one-off scripts removed; project name unified to 'stokastisk'", "vLLM wheel byte-identical to the base release (no native rebuild); stokastisk wheel rebuilt from sdk/"],
          wheel_lineage="manifests/WHEEL-LINEAGE.json", native_audit=naj,
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
