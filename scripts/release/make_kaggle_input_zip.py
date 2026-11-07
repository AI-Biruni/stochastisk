#!/usr/bin/env python3.13
"""Build the Kaggle dataset ZIP for a release bundle. Reproducible (fixed timestamps/permissions, sorted entries) and idempotent.
usage: make_kaggle_input_zip.py BUNDLE_NAME [OUT_DIR]      e.g. stokastisk-0.3.0-kaggle-t4-sm75-rc4
 * verifies manifests/SHA256SUMS before zipping, the required script/wheel/notebook inventory, that notebooks are UNEXECUTED,
   and that no credential / executed-notebook / Flash* file is selected;
 * if the output ZIP already exists: identical bytes -> reports 'unchanged' (exit 0); different bytes -> the old ZIP is renamed
   to <name>.superseded-<sha12> (never overwritten, never deleted) and the new one is written."""
import hashlib, json, pathlib, re, shutil, sys, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
name = sys.argv[1]
BUNDLE = ROOT / "dist-rc" / name
OUT = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "stokastisk-kaggle-input"
ZIP = OUT / f"{name}-kaggle-input.zip"
SCRIPTS = ["01_env_audit", "02_install", "03_single_gpu", "04_nccl_launch", "04_nccl_dual_gpu", "04_nccl_summary", "05_inference",
           "05_backend_evidence", "06_tp_benchmark", "07_summary", "_common"]
NOTEBOOKS = sorted(p.stem for p in (BUNDLE / "notebooks").glob("*.ipynb"))   # optional: end-user notebooks are authored separately
REQUIRED = ["README.md", "BUILD.md", "INSTALL.md", "RELEASE.md", "manifests/SHA256SUMS", "requirements/kaggle-t4-core.txt",
            "wheels/stokastisk-0.3.0-py3-none-any.whl", "wheels/vllm-0.25.1-cp38-abi3-linux_x86_64.whl",
            *[f"scripts/kaggle/{s}.py" for s in SCRIPTS]]
BAD_PARTS = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "notebooks-executed-with-output"}
BAD_NAMES = {"tokens.env", ".env", "id_rsa", "id_ed25519"}
BAD_SUBSTR = ("flashinfer", "vllm_flash_attn", "flash_attn", "flashmla")
SECRET = re.compile(("gh" + "p_[A-Za-z0-9]{20}|github" + "_pat_[A-Za-z0-9_]{20}|hf" + "_[A-Za-z0-9]{30}").encode())   # split so this file never matches itself
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()  # noqa: E731

assert BUNDLE.is_dir(), f"bundle not found: {BUNDLE}"
missing = [r for r in REQUIRED if not (BUNDLE / r).is_file()]
assert not missing, f"missing required files: {missing}"
bad = []
for line in (BUNDLE / "manifests/SHA256SUMS").read_text().splitlines():
    h, rel = line.split(None, 1); rel = rel.strip().lstrip("*")
    if ".." in pathlib.PurePosixPath(rel).parts or not (BUNDLE / rel).is_file() or sha(BUNDLE / rel) != h: bad.append(rel)
assert not bad, f"SHA256SUMS mismatches: {bad[:10]}"
for n in NOTEBOOKS:
    for c in json.loads((BUNDLE / "notebooks" / f"{n}.ipynb").read_text())["cells"]:
        assert c["cell_type"] != "code" or (not c["outputs"] and c["execution_count"] is None), f"{n} is executed / has outputs"
files = []
for p in sorted(BUNDLE.rglob("*")):
    if not p.is_file(): continue
    rel = p.relative_to(BUNDLE)
    assert not p.is_symlink(), f"symlink: {rel}"
    if any(x in BAD_PARTS for x in rel.parts) or p.name in BAD_NAMES or p.suffix in (".pyc", ".pyo"): continue
    low = rel.as_posix().lower()
    if p.parts and "wheels" in rel.parts: assert not any(k in p.name.lower() for k in BAD_SUBSTR), f"prohibited wheel: {rel}"
    if p.suffix in (".py", ".md", ".json", ".txt", ".sh", ".ipynb", ".patch", ".yml", ".toml"):
        assert not SECRET.search(p.read_bytes()), f"credential-like string in {rel}"
    files.append(p)

def build(dest):
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            zi = zipfile.ZipInfo(f"{name}/{p.relative_to(BUNDLE).as_posix()}", (2026, 1, 1, 0, 0, 0)); zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = (0o755 if p.suffix in (".py", ".sh") else 0o644) << 16
            z.writestr(zi, p.read_bytes())

OUT.mkdir(parents=True, exist_ok=True)
tmp = OUT / (ZIP.name + ".tmp"); build(tmp)
new = sha(tmp)
if ZIP.exists() and sha(ZIP) == new:
    tmp.unlink(); print("unchanged:", ZIP, new); status = "unchanged"
else:
    if ZIP.exists():
        old = sha(ZIP); keep = OUT / f"{ZIP.name}.superseded-{old[:12]}"; ZIP.rename(keep); print("previous ZIP preserved as", keep.name)
    tmp.rename(ZIP); status = "written"
with zipfile.ZipFile(ZIP) as z:
    assert z.testzip() is None; names = z.namelist(); assert len(names) == len(set(names)), "duplicate paths"
    assert all(n.startswith(name + "/") for n in names)
pathlib.Path(str(ZIP) + ".sha256").write_text(f"{new}  {ZIP.name}\n")
print(json.dumps({"status": status, "zip": str(ZIP), "sha256": new, "size": ZIP.stat().st_size, "entries": len(names)}, indent=1))
