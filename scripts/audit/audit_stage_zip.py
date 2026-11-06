#!/usr/bin/env python3.13
"""Forensic audit of stage.zip vs canonical upstream vLLM source. usage: audit_stage_zip.py stage.zip upstream_dir extract_dir out_dir"""
import hashlib, json, os, re, stat, sys, time, zipfile, collections, pathlib
zp, up, ex, out = map(pathlib.Path, sys.argv[1:5]); out.mkdir(parents=True, exist_ok=True)
sha = lambda b: hashlib.sha256(b).hexdigest()
z = zipfile.ZipFile(zp); infos = z.infolist(); names = [i.filename for i in infos]
dups = [n for n, c in collections.Counter(names).items() if c > 1]
trav = [n for n in names if n.startswith("/") or ".." in pathlib.PurePosixPath(n).parts]
links = [i.filename for i in infos if stat.S_ISLNK(i.external_attr >> 16)]
bins = [i.filename for i in infos if i.filename.endswith((".so", ".a", ".o", ".cubin", ".whl", ".exe"))]
susp = [n for n in names if re.search(r"(\.env$|token|secret|id_rsa|\.pem$)", n, re.I)]
if ex.exists(): raise SystemExit(f"{ex} exists; refusing to reuse")
z.extractall(ex)
st = ex / "stage"
files = {str(p.relative_to(st)): p for p in st.rglob("*") if p.is_file()}
vers = (st / "vllm/version.py").read_text() if (st / "vllm/version.py").exists() else None
a1 = {"zip_sha256": sha(zp.read_bytes()), "zip_size": zp.stat().st_size, "entries": len(names), "files": len(files),
      "duplicate_paths": dups, "path_traversal": trav, "symlinks": links, "binary_files": bins, "suspicious_names": susp,
      "has_setup_py": "setup.py" in files, "has_csrc": any(f.startswith("csrc/") for f in files), "top_level": sorted({f.split('/')[0] for f in files}),
      "version_file": vers.strip()[:300] if vers else None}
# diff vs upstream python tree
ups = {str(p.relative_to(up)): p for p in (up / "vllm").rglob("*") if p.is_file() and "__pycache__" not in p.parts}
ups = {k: v for k, v in ups.items()}
same, mod, only_stage, removed = 0, [], [], []
for k, p in files.items():
    if k.startswith("vllm/"):
        u = ups.get(k)
        if u is None: only_stage.append(k)
        elif sha(u.read_bytes()) == sha(p.read_bytes()): same += 1
        else: mod.append(k)
removed = sorted(k for k in ups if k not in files and k.endswith((".py", ".json", ".yaml", ".yml", ".txt")))
rm_dirs = collections.Counter("/".join(k.split("/")[:3]) for k in removed)
fa = lambda s: re.search(r"flash_?attn|flashinfer|flashmla|fa2_C|fa3_C|quack", s, re.I)
flagged_paths = sorted(k for k in files if fa(k))
flagged_imports = {}
for k, p in files.items():
    if k.endswith(".py"):
        for ln, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if re.match(r"\s*(import|from)\s+(vllm\.vllm_flash_attn|flashinfer|flash_attn)\b", line) and not line.startswith("    "):  # top-level only
                flagged_imports.setdefault(k, []).append((ln, line.strip()))
triton = {k: (k in files) for k in ("vllm/v1/attention/backends/triton_attn.py", "vllm/v1/attention/ops/triton_unified_attention.py",
          "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py", "vllm/v1/attention/backends/registry.py", "vllm/v1/attention/selector.py")}
diff = {"identical_files": same, "modified_vs_upstream": mod, "only_in_stage": only_stage, "removed_from_upstream_count": len(removed),
        "removed_by_dir_top20": rm_dirs.most_common(20), "stage_flash_named_paths": flagged_paths, "toplevel_flash_imports": flagged_imports,
        "triton_required_files_present": triton}
meta = {"timestamp": time.strftime("%FT%T%z"), "kind": "static", "tool": "audit_stage_zip.py", "commands": ["unzip(extractall)", "sha256 compare vs upstream/vllm"]}
(out / "source-archive-audit.json").write_text(json.dumps({**meta, **a1}, indent=1))
(out / "source-diff-report.json").write_text(json.dumps({**meta, **diff}, indent=1))
print(json.dumps({**a1, "mod": len(mod), "only_stage": len(only_stage), "same": same, "removed": len(removed), "flash_paths": len(flagged_paths), "toplevel_flash_import_files": len(flagged_imports), "triton": triton}, indent=1))
