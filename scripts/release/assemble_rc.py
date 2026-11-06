#!/usr/bin/env python3.13
"""Assemble the release candidate bundle + zip + remaining evidence. usage: assemble_rc.py <scratch_dir>"""
import hashlib, io, json, os, pathlib, shutil, subprocess, sys, time, unittest, zipfile
ROOT = pathlib.Path(__file__).resolve().parents[2]; scratch = pathlib.Path(sys.argv[1])
NAME = "stokastisk-0.3.0-kaggle-t4-sm75"; D = ROOT / "dist-rc"; B = D / NAME; EV = ROOT / "evidence-rc"
now = lambda: time.strftime("%FT%T%z"); sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
if B.exists(): shutil.rmtree(B)
for d in ("wheels", "manifests", "LICENSES", "scripts", "notebooks", "docs", "cuda", "requirements", "patches", "evidence"): (B / d).mkdir(parents=True)
for w in (D / "wheels").glob("*.whl"): shutil.copy2(w, B / "wheels")
ign = shutil.ignore_patterns("__pycache__", "*.pyc")
shutil.copytree(ROOT / "scripts/kaggle", B / "scripts/kaggle", ignore=ign); shutil.copytree(ROOT / "scripts/release", B / "scripts/release", ignore=ign)
shutil.copytree(ROOT / "scripts/audit", B / "scripts/audit", ignore=ign); shutil.copy2(ROOT / "scripts/make_notebooks.py", B / "scripts"); shutil.copy2(ROOT / "scripts/build_vllm_sm75.sh", B / "scripts")
shutil.copytree(ROOT / "tests", B / "tests", ignore=ign)
for n in (ROOT / "notebooks").glob("*.ipynb"): shutil.copy2(n, B / "notebooks")
for n in (ROOT / "docs").glob("*.md"): shutil.copy2(n, B / "docs")
for n in (ROOT / "requirements").glob("*.txt"): shutil.copy2(n, B / "requirements")
for n in (ROOT / "patches").glob("*"): shutil.copy2(n, B / "patches")
for f in ("README.md", "INSTALL.md", "BUILD.md", "RELEASE.md"): shutil.copy2(ROOT / f, B)
shutil.copy2(ROOT / "LICENSE", B / "LICENSES/vllm-Apache-2.0.txt")
(B / "LICENSES/NOTICE.txt").write_text("vLLM (Apache-2.0) derivative; FlashAttention/FlashInfer/FlashMLA excluded. Third-party packages installed by the installer carry their own licenses (xgrammar, llguidance, msgspec, cbor2, pybase64, openai-harmony, apache-tvm-ffi, partial-json-parser).\n")
(B / "cuda/README.md").write_text("No CUDA libraries are bundled by design (libcudart.so.13/libtorch from Kaggle, libcuda.so.1 from the driver). Native files are inside the vLLM wheel; see manifests/ELF-DEPENDENCY-MANIFEST.json and CUDA-ARCH-MANIFEST.json.\n")
def write_sums():
    files = sorted(p for p in B.rglob("*") if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts)
    (B / "manifests/SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(B)}\n" for p in files))
write_sums()
tools = {"python": sys.version.split()[0]}
def rep(name, kind, cmds, body, failures=(), warnings=()):
    d = {"timestamp": now(), "kind": kind, "source_revision": {"vllm_tag": "v0.25.1", "vllm_commit": "752a3a504485790a2e8491cacbb35c137339ad34", "stokastisk_git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()},
         "tool_versions": tools, "input_hashes": {"stage.zip": sha(ROOT / "stage.zip"), **{w.name: sha(w) for w in (D / "wheels").glob("*.whl")}}, "commands": cmds, "failures": list(failures), "warnings": list(warnings), **body}
    (EV / name).write_text(json.dumps(d, indent=1)); return d
# --- SDK tests
def run_unittest(path):
    res = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=path))
    return {"ran": res.testsRun, "failures": [str(f[0]) for f in res.failures], "errors": [str(e[0]) for e in res.errors], "skipped": [str(s[0]) for s in res.skipped]}
env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
def sub_unittest(f):
    r = subprocess.run([sys.executable, str(ROOT / "tests" / f)], capture_output=True, text=True, env=env); last = r.stderr.strip().splitlines()[-3:]
    import re; m = re.search(r"Ran (\d+) tests", r.stderr)
    return {"returncode": r.returncode, "ran": int(m.group(1)) if m else None, "tail": last}
sdk = sub_unittest("test_sdk_cpu.py"); ins = sub_unittest("test_installer.py")
rep("sdk-test-results.json", "local (CPU-only; no GPU; mocks are logic tests, not GPU evidence)", ["CUDA_VISIBLE_DEVICES= python3.13 tests/test_sdk_cpu.py"], sdk, failures=[] if sdk["returncode"] == 0 else sdk["tail"])
# --- installer dry run on this (non-Kaggle) machine
sc = pathlib.Path(scratch) / "dryrun-out"; shutil.rmtree(sc, ignore_errors=True)
dr = subprocess.run([sys.executable, str(B / "scripts/kaggle/02_install.py"), str(B), "--dry-run"], capture_output=True, text=True, env=dict(os.environ, STOKASTISK_OUT=str(sc), PYTHONDONTWRITEBYTECODE="1"))
drj = next((json.load(open(sc / n)) for n in ("02_install_dryrun.json", "02_install.json") if (sc / n).exists()), {})
rep("installer-test.json", "local (unit tests on synthetic snapshots + dry-run on a NON-Kaggle machine)", ["python3.13 tests/test_installer.py", "02_install.py <bundle> --dry-run"],
    {"unit_tests": ins, "dry_run_returncode": dr.returncode, "dry_run_status": drj.get("status"), "dry_run_checks": drj.get("checks"), "dry_run_plan": drj.get("plan"),
     "dry_run_downloaded_bytes": drj.get("downloaded_bytes"), "dry_run_seconds": drj.get("seconds"),
     "interpretation": "On this development machine the baseline gate is expected to refuse (no Kaggle torch/transformers 5.x); that refusal-before-any-change IS the tested behaviour. The Kaggle outcome is PENDING USER."},
    failures=[] if ins["returncode"] == 0 else ins["tail"])
rep("dependency-install-plan.json", "static + local", ["make_plan() on synthetic Kaggle-baseline snapshot (versions from executed notebook logs) with the real candidate wheel metadata"],
    {"core_pins": [l.split("#")[0].strip() for l in (ROOT / "requirements/kaggle-t4-core.txt").read_text().splitlines() if l.strip() and not l.startswith("#")], "download_bytes_measured": 57259125,
     "download_note": "pip download --no-deps of the 8 pins, measured locally; previous failed install fetched ~178 files (see 02-stokastisk-installation.ipynb)", "installs_with_resolver": False})
rep("protected-package-audit.json", "local (unit tests on synthetic snapshots built from Kaggle log versions)", ["tests/test_installer.py"],
    {"protected_exact": {"torch": "2.11.0+cu130", "triton": "3.6.0"}, "floors": {"transformers": ">=5.5.3 and major>=5"}, "after_install_rule": "no preinstalled distribution may change version or disappear",
     "scenarios_covered": ["transformers 4.57.6 rejected", "torch/triton change rejected", "preinstalled flashinfer rejected", "CUDA 13.2 nvvm/nvcc replacement detected by diff", "removed package detected", "idempotent empty plan", "unsatisfiable requirement aborts before install"],
     "unit_test_returncode": ins["returncode"], "kaggle_verification": "PENDING USER (notebook 02)"})
src = json.load(open(EV / "source-archive-audit.json")); sweep = json.load(open(scratch / "sweep.json")) if (scratch / "sweep.json").exists() else {}
usage = json.load(open(scratch / "absent-usage.json")) if (scratch / "absent-usage.json").exists() else {}
import re as _re
_n = lambda x: _re.sub(r"[-_.]+", "-", _re.match(r"[A-Za-z0-9_.\-]+", x).group(0)).lower()
_unmet = {_n(u) for u in json.load(open(ROOT / "tests/data/kaggle_notebook02_unmet_requirements.json"))["unmet"]}
_core = [l[15:] for l in zipfile.ZipFile(next((D / "wheels").glob("vllm-*.whl"))).read("vllm-0.25.1.dist-info/METADATA").decode().splitlines() if l.startswith("Requires-Dist: ") and "extra ==" not in l]
kx = {"core_requirements_unmet_on_kaggle": sorted(_n(c) for c in _core if _n(c) in _unmet), "pins_in_core_file": sorted(_n(l.split("#")[0].strip()) for l in (ROOT / "requirements/kaggle-t4-core.txt").read_text().splitlines() if l.strip() and not l.startswith("#")),
      "prohibited_in_core_requirements": sorted(_n(c) for c in _core if _n(c) in {"flashinfer-python", "flashinfer-cubin", "nvidia-cutlass-dsl", "quack-kernels", "humming-kernels", "tilelang"})}
kx["documented_spec_deviations"] = {"fastapi": "upstream pin <0.137 was the only reason it was unmet on Kaggle (0.141.1 installed); wheel requires >=0.133.0; import sweep passes with 0.141.1"}
kx["unmet_equals_pins_excluding_deviations"] = set(kx["core_requirements_unmet_on_kaggle"]) - set(kx["documented_spec_deviations"]) == set(kx["pins_in_core_file"])
rep("dependency-audit.json", "static + local import sweep", ["scripts/audit/import_sweep.py (CPU)", "AST usage scan of absent-on-Kaggle packages"],
    {"core_import_sweep": {"ok": sweep.get("ok"), "third_party_dists_imported": sweep.get("third_party_dists_imported")}, "absent_on_kaggle_usage": usage,
     "profiles": {p.name: [l.strip() for l in p.read_text().splitlines() if l.strip() and not l.startswith('#')] for p in (ROOT / "requirements").glob("*.txt")},
     "kaggle_executed_log_cross_check": kx,
     "verified_against_kaggle_versions": {"transformers": "5.20.2", "tokenizers": "0.23.3", "huggingface-hub": "1.33.0", "fastapi": "0.141.1"}})
nb = []
for f in sorted((ROOT / "notebooks").glob("*.ipynb")):
    j = json.load(open(f)); import ast
    ok = True
    for c in j["cells"]:
        if c["cell_type"] == "code":
            try: ast.parse("\n".join(l for l in "".join(c["source"]).splitlines() if not l.startswith("!")))
            except SyntaxError: ok = False
    nb.append({"notebook": f.name, "valid_json": True, "cells": len(j["cells"]), "code_cells_parse": ok, "outputs_present": any(c.get("outputs") for c in j["cells"] if c["cell_type"] == "code")})
refs = {"scripts_referenced_exist": all((ROOT / "scripts/kaggle" / s).exists() for s in ("01_env_audit.py", "02_install.py", "03_single_gpu.py", "04_nccl_dual_gpu.py", "04_nccl_summary.py", "05_inference.py", "06_tp_benchmark.py", "07_summary.py"))}
rep("notebook-validation.json", "static (JSON + syntax + script existence). NOT executed.", ["json.load + ast.parse"], {"notebooks": nb, **refs, "execution_status": "PENDING USER on Kaggle"})
for f in EV.glob("*.json"): shutil.copy2(f, B / "evidence" / f.name)
# manifests
for n in ("ELF-DEPENDENCY-MANIFEST.json", "CUDA-ARCH-MANIFEST.json"):
    shutil.copy2(scratch / ("elf-" + n), B / "manifests" / n)
shutil.copy2(EV / "native-binary-audit.json", B / "manifests/NATIVE-BINARY-AUDIT.json")
dep = json.load(open(EV / "dependency-audit.json")); json.dump({"core_pins": json.load(open(EV / "dependency-install-plan.json"))["core_pins"], "profiles": dep["profiles"], "usage": usage}, open(B / "manifests/DEPENDENCY-MANIFEST.json", "w"), indent=1)
orig = ROOT / "work/build/vllm-wheel/vllm-0.25.1-cp38-abi3-linux_x86_64.whl"
bm = rep("BUILD-MANIFEST.json", "static", ["see BUILD.md"], {"inputs": {"vllm_tarball_sha256": sha(ROOT / "work/downloads/vllm-v0.25.1.tar.gz"), "stage_zip_sha256": sha(ROOT / "stage.zip"), "original_build_wheel": {"name": orig.name, "sha256": sha(orig), "size": orig.stat().st_size}},
    "native_extensions_origin": "existing local build (NVCC 13.0.48, TORCH_CUDA_ARCH_LIST=7.5, CMAKE_CUDA_ARCHITECTURES=75); not rebuilt; FA2/FA3 binaries dropped", "kaggle_nvcc": "13.0.88 (differs)", "patches": sorted(p.name for p in (ROOT / "patches").glob("*.patch")),
    "removed_modules": json.load(open(ROOT / "work/rc-tree/REMOVED.json")), "retained_flash_named_modules": json.load(open(ROOT / "work/rc-tree/KEPT.json")), "clean_native_rebuild_without_fa": "NOT PERFORMED", "build_exit_code_of_original": "not recorded in log; wheel assembled; vllm-server (Rust) failed: protoc missing"})
shutil.copy2(EV / "BUILD-MANIFEST.json", B / "manifests/BUILD-MANIFEST.json")
rm = rep("RELEASE-MANIFEST.json", "static", [], {"name": NAME, "status": "RELEASE CANDIDATE — STATICALLY VERIFIED, LOCALLY TESTED, KAGGLE TEST PENDING", "kaggle_acceptance": "PENDING USER EXECUTION",
    "wheels": {w.name: {"size": w.stat().st_size, "sha256": sha(w)} for w in sorted((B / "wheels").glob("*.whl"))}})
shutil.copy2(EV / "RELEASE-MANIFEST.json", B / "manifests/RELEASE-MANIFEST.json")
for f in EV.glob("*.json"): shutil.copy2(f, B / "evidence" / f.name)
write_sums()
# zip (stdlib), fixed order and timestamps
zp = D / f"{NAME}.zip"
if zp.exists(): zp.unlink()
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in sorted(B.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            zi = zipfile.ZipInfo(f"{NAME}/{p.relative_to(B)}", (2026, 11, 6, 0, 0, 0)); zi.compress_type = zipfile.ZIP_DEFLATED; zi.external_attr = (0o755 if p.suffix in (".py", ".sh") else 0o644) << 16
            z.writestr(zi, p.read_bytes())
# integrity: extract fresh, verify every hash, discoverability, forbidden content
x = pathlib.Path(scratch) / "zip-verify"; shutil.rmtree(x, ignore_errors=True)
zipfile.ZipFile(zp).extractall(x); root = x / NAME
bad = [l for l in (root / "manifests/SHA256SUMS").read_text().splitlines() if sha(root / l.split(None, 1)[1].strip()) != l.split()[0]]
names = zipfile.ZipFile(zp).namelist()
forbidden = [n for n in names if any(k in n.lower() for k in ("tokens.env", ".env", ".git/", "__pycache__", ".pyc", "id_rsa", ".so", "vllm_flash_attn", "flashinfer-python"))]
disc = [str(p.parent.parent.name) for p in x.glob("**/manifests/SHA256SUMS")]
secret_hits = []
for n in names:
    if n.endswith((".py", ".md", ".json", ".txt", ".ipynb", ".sh", ".patch")):
        t = zipfile.ZipFile(zp).read(n).decode(errors="ignore")
        if any(k in t for k in ("gh" + "p_", "github" + "_pat_", "hf" + "_" + "A", "GITHUB" + "_TOKEN=", "HF" + "_TOKEN=")): secret_hits.append(n)
ri = rep("release-integrity.json", "static", ["zipfile.testzip", "extract + sha256 verify", "forbidden-content scan"],
    {"zip": zp.name, "zip_size": zp.stat().st_size, "zip_sha256": sha(zp), "zip_test": zipfile.ZipFile(zp).testzip(), "entries": len(names), "hash_mismatches": bad,
     "discoverable_roots_for_glob_/kaggle/input/**/manifests/SHA256SUMS": disc, "scripts_kaggle_present": (root / "scripts/kaggle/02_install.py").exists(),
     "forbidden_names_in_zip": forbidden, "possible_secret_strings": secret_hits, "wheels": {w.name: w.stat().st_size for w in sorted((root / "wheels").glob("*.whl"))}},
    failures=bad + forbidden + secret_hits)
(zp.with_suffix(".zip.sha256")).write_text(f"{sha(zp)}  {zp.name}\n")
print(json.dumps({"zip": str(zp), "size": zp.stat().st_size, "sha256": sha(zp), "mismatch": bad, "forbidden": forbidden, "secret": secret_hits, "sdk": sdk["ran"], "installer_tests": ins["ran"], "dry_run_rc": dr.returncode, "dry_run_status": drj.get("status")}, indent=1))
