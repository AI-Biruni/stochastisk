#!/usr/bin/env python3.13
"""Static evidence: wheel-audit, native-binary-audit, cuda-architecture-audit, flashattention-flashinfer-audit.
usage: make_binary_evidence.py <vllm.whl> <stokastisk.whl> <orig.whl> <tree_dir> <evidence_dir> <scratch>"""
import ast, csv, hashlib, base64, io, json, pathlib, re, subprocess, sys, time, zipfile
vw, sw, ow, tree, ev, scratch = map(pathlib.Path, sys.argv[1:7]); ev.mkdir(exist_ok=True, parents=True)
now = time.strftime("%FT%T%z"); sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
tools = {"python": sys.version.split()[0], "nvcc": subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.splitlines()[-2],
         "readelf": subprocess.run(["readelf", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
         "cuobjdump": subprocess.run(["cuobjdump", "--version"], capture_output=True, text=True).stdout.splitlines()[-1]}
meta = lambda kind, cmds: {"timestamp": now, "kind": kind, "tool_versions": tools, "commands": cmds, "source": {"tag": "v0.25.1", "commit": "752a3a504485790a2e8491cacbb35c137339ad34"}}
def wheel_info(w):
    z = zipfile.ZipFile(w); names = z.namelist(); rn = next(n for n in names if n.endswith("RECORD"))
    rows = list(csv.reader(io.StringIO(z.read(rn).decode()))); bad = 0
    for n, h, sz in rows:
        if n == rn: continue
        d = z.read(n)
        if "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(d).digest()).rstrip(b"=").decode() != h or str(len(d)) != sz: bad += 1
    m = z.read(next(n for n in names if n.endswith("/METADATA"))).decode()
    return {"file": w.name, "size": w.stat().st_size, "sha256": sha(w), "zip_test": z.testzip(), "entries": len(names), "record_rows": len(rows),
            "record_mismatches": bad, "unlisted_files": sorted(set(names) - {r[0] for r in rows}),
            "wheel_file": z.read(next(n for n in names if n.endswith("/WHEEL"))).decode().splitlines(),
            "requires_dist": [l[15:] for l in m.splitlines() if l.startswith("Requires-Dist: ")], "requires_python": next((l for l in m.splitlines() if l.startswith("Requires-Python")), None),
            "native_extensions": [{"name": i.filename, "size": i.file_size} for i in z.infolist() if i.filename.endswith(".so")],
            "python_files": sum(n.endswith(".py") for n in names)}
wa = {"timestamp": now, **meta("static", ["zipfile.testzip", "RECORD sha256 verify"]), "vllm_wheel": wheel_info(vw), "stokastisk_wheel": wheel_info(sw), "original_build_wheel_for_comparison": wheel_info(ow)}
o, n = set(zipfile.ZipFile(ow).namelist()), set(zipfile.ZipFile(vw).namelist())
wa["vs_original_build"] = {"only_in_original": len(o - n), "only_in_t4": len(n - o), "only_in_original_native": sorted(x for x in o - n if x.endswith(".so"))}
(ev / "wheel-audit.json").write_text(json.dumps(wa, indent=1))
# native
ex = scratch / "wheel-extract"
if ex.exists(): subprocess.run(["rm", "-rf", str(ex)])
zipfile.ZipFile(vw).extractall(ex)
out = subprocess.run([sys.executable, str(pathlib.Path(__file__).parent / "audit_elf.py"), str(ex), str(scratch / "elf-")], capture_output=True, text=True)
elf = json.load(open(scratch / "elf-ELF-DEPENDENCY-MANIFEST.json")); ca = json.load(open(scratch / "elf-CUDA-ARCH-MANIFEST.json"))
def vkey(s): return tuple(int(x) for x in s.split(".")) if s else (0,)
KAGGLE_GLIBC = "2.39"
for k, v in elf.items():
    sig = subprocess.run(["grep", "-aciE", "flashinfer|flash_attn|vllm_fa[234]|flash_fwd|FlashAttn|flashmla", str(ex / k)], capture_output=True, text=True).stdout.strip()
    v["flash_signature_hits"] = int(sig or 0); v["glibc_ok_for_kaggle_2_39"] = vkey(v["glibc_max"]) <= vkey(KAGGLE_GLIBC) if v["glibc_max"] else True
    ptx = subprocess.run(["cuobjdump", "--list-ptx", str(ex / k)], capture_output=True, text=True).stdout
    v["embedded_ptx"] = re.findall(r"compute_\d+", ptx)
na = {**meta("static", ["readelf -d/-V", "ldd (torch lib path)", "cuobjdump --list-elf/--list-ptx", "grep -aci <flash signatures>"]),
      "target_glibc": KAGGLE_GLIBC, "binaries": elf,
      "summary": {"count": len(elf), "any_cuda12_link": any(v["links_cuda12"] for v in elf.values()), "all_glibc_ok": all(v["glibc_ok_for_kaggle_2_39"] for v in elf.values()),
                  "unresolved_deps": {k: v["unresolved_with_torch_libpath"] for k, v in elf.items() if v["unresolved_with_torch_libpath"]},
                  "flash_signature_total": sum(v["flash_signature_hits"] for v in elf.values())},
      "libs_provided_by": {"libtorch*/libc10*/libtorch_cuda": "Kaggle PyTorch 2.11.0+cu130", "libcudart.so.13": "Kaggle CUDA 13.0 runtime (torch wheel's nvidia-cuda-runtime)", "libcuda.so.1": "NVIDIA driver 580.178.04", "libstdc++/libgcc_s/libc": "system (glibc 2.39)", "statically_linked_cuda": "not determined per-symbol; NEEDED lists only the dynamic ones shown", "triton_jit": "Triton 3.6.0 compiles attention kernels at runtime, no cubin shipped"}}
(ev / "native-binary-audit.json").write_text(json.dumps(na, indent=1))
cu = {**meta("static", ["cuobjdump --list-elf", "cuobjdump --list-ptx"]), "target": ["sm_75"], "per_file": ca["per_file"], "non_target_architectures": ca["non_sm75"],
      "ptx_present": {k: v["embedded_ptx"] for k, v in elf.items() if v["embedded_ptx"]}, "note": "static cubin inspection only; execution on a T4 is PENDING USER (Kaggle notebook 03)"}
(ev / "cuda-architecture-audit.json").write_text(json.dumps(cu, indent=1))
# FA / FI audit
BADN = re.compile(r"(flash_?attn|flashinfer|flashmla|_fa[234]_C|flash_mla|cutedsl_C)", re.I)
z = zipfile.ZipFile(vw); wnames = z.namelist()
named = sorted(x for x in wnames if BADN.search(x))
def modlevel(tree_):
    hits = []
    def walk(nodes, lazy):
        for nd in nodes:
            if isinstance(nd, (ast.FunctionDef, ast.AsyncFunctionDef)): walk(nd.body, True); continue
            if isinstance(nd, ast.If) and "TYPE_CHECKING" in ast.dump(nd.test): continue
            mods = [a.name for a in nd.names] if isinstance(nd, ast.Import) else ([nd.module] if isinstance(nd, ast.ImportFrom) and nd.module and nd.level == 0 else [])
            for m in mods:
                if re.match(r"(flashinfer|flash_attn|vllm\.vllm_flash_attn|vllm\.third_party\.flashmla)(\.|$)", m) and not lazy: hits.append(m)
            for f in ("body", "orelse", "finalbody"):
                s = getattr(nd, f, None)
                if s: walk(s, lazy)
            for h in getattr(nd, "handlers", []): walk(h.body, lazy)
    walk(tree_.body, False); return hits
ml = {}
for p in (tree / "vllm").rglob("*.py"):
    try: h = modlevel(ast.parse(p.read_text()))
    except SyntaxError: continue
    if h: ml[str(p.relative_to(tree))] = h
lazy_refs = {}
for p in (tree / "vllm").rglob("*.py"):
    t = p.read_text(errors="replace")
    c = len(re.findall(r"^\s+(?:from|import)\s+(?:flashinfer|flash_attn|vllm\.vllm_flash_attn)", t, re.M))
    if c: lazy_refs[str(p.relative_to(tree))] = c
reg = (tree / "vllm/v1/attention/backends/registry.py").read_text()
reqs = [l for f in pathlib.Path(sys.argv[0]).resolve().parents[2].glob("requirements/*.txt") if "prohibited" not in f.name for l in f.read_text().splitlines() if re.search(r"flash", l, re.I) and not l.strip().startswith("#")]
fa = {**meta("static", ["zip namelist regex", "AST module-level import scan", "grep -aci on ELF", "metadata Requires-Dist scan"]),
      "wheel_prohibited_native_binaries": [x for x in wnames if x.endswith(".so") and BADN.search(x)],
      "wheel_paths_with_flash_names": named, "wheel_dist_info_requires_prohibited": [r for r in wa["vllm_wheel"]["requires_dist"] if re.search(r"flash|cutlass-dsl|quack|humming|tilelang|cuda-tile", r, re.I)],
      "vllm_flash_attn_dir_in_wheel": any("vllm_flash_attn" in x for x in wnames),
      "elf_flash_signature_hits": {k: v["flash_signature_hits"] for k, v in elf.items()},
      "module_level_imports_of_prohibited_packages_in_tree": ml,
      "lazy_function_level_references_files": len(lazy_refs), "lazy_reference_note": "inside functions of retained inert wrappers/backends; reached only if FlashInfer is installed or a flash backend is selected; SDK rejects those and installer forbids the packages",
      "retained_flash_named_modules": [x for x in named],
      "registry_enum_mentions_flash_backends": bool(re.search(r"FLASH_ATTN|FLASHINFER", reg)),
      "registry_note": "enum strings are lazy qualnames; selecting them raises ImportError (modules removed); Stokastisk config rejects them",
      "requirements_profiles_mentioning_flash_outside_prohibited": reqs,
      "pass": not [x for x in wnames if x.endswith(".so") and BADN.search(x)] and not ml and not any("vllm_flash_attn" in x for x in wnames)
              and all(v["flash_signature_hits"] == 0 for v in elf.values())}
(ev / "flashattention-flashinfer-audit.json").write_text(json.dumps(fa, indent=1))
print(json.dumps({"fa_pass": fa["pass"], "elf_flash_hits": fa["elf_flash_signature_hits"], "module_level": ml, "named_in_wheel": len(named), "summary": na["summary"], "arch": {k: v for k, v in ca["per_file"].items()}}, indent=1))
