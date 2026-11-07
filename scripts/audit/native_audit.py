#!/usr/bin/env python3.13
"""Audit every native object in a wheel: ELF class/arch, NEEDED, RUNPATH, GLIBC/GLIBCXX floors, CUDA runtime linkage, SM cubins/PTX
(cuobjdump from CUDA 13.0), and CUDA-12 / Flash markers. Read-only. usage: native_audit.py WHEEL OUT.json"""
import hashlib, json, pathlib, re, shutil, subprocess, sys, tempfile, zipfile
whl, out = sys.argv[1:3]
CUOBJ = shutil.which("cuobjdump") or "/usr/local/cuda-13.0/bin/cuobjdump"
def run(*a): r = subprocess.run(a, capture_output=True, text=True); return r.stdout + r.stderr
def ver(items, pre): return sorted({m for m in items}, key=lambda s: [int(x) for x in re.findall(r"\d+", s)])[-1:] 
res = {"wheel": pathlib.Path(whl).name, "wheel_sha256": hashlib.sha256(pathlib.Path(whl).read_bytes()).hexdigest(), "cuobjdump": run(CUOBJ, "--version").strip().splitlines()[-1:] , "objects": []}
with tempfile.TemporaryDirectory() as d:
    z = zipfile.ZipFile(whl)
    names = [n for n in z.namelist() if n.endswith(".so") or ".so." in n]
    z.extractall(d, names)
    for n in sorted(names):
        p = pathlib.Path(d, n); data = p.read_bytes()
        hdr = run("readelf", "-h", str(p)); dyn = run("readelf", "-d", str(p))
        syms = run("objdump", "-T", str(p))
        glibc = sorted(set(re.findall(r"GLIBC_(\d+(?:\.\d+)+)", syms)), key=lambda s: [int(x) for x in s.split(".")])
        glibcxx = sorted(set(re.findall(r"GLIBCXX_(\d+(?:\.\d+)+)", syms)), key=lambda s: [int(x) for x in s.split(".")])
        needed = re.findall(r"NEEDED\)\s+Shared library: \[([^\]]+)\]", dyn)
        elf = run(CUOBJ, "--list-elf", str(p)); ptx = run(CUOBJ, "--list-ptx", str(p))
        sms = sorted(set(re.findall(r"sm_(\d+\w?)", elf))); ptx_sms = sorted(set(re.findall(r"sm_(\d+\w?)", ptx)))
        low = data.lower()
        res["objects"].append({"member": n, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
            "class": re.search(r"Class:\s+(\S+)", hdr).group(1), "machine": re.search(r"Machine:\s+(.+)", hdr).group(1).strip(),
            "needed": needed, "runpath": re.findall(r"(?:RUNPATH|RPATH)\)\s+Library \w+path: \[([^\]]*)\]", dyn),
            "max_glibc": glibc[-1] if glibc else None, "max_glibcxx": glibcxx[-1] if glibcxx else None,
            "cuda_runtime_needed": [x for x in needed if re.match(r"libcu(dart|blas|rand|sparse|solver|fft)|libnvrtc|libnccl", x)],
            "needs_cuda12_soname": [x for x in needed if re.search(r"\.so\.12(\.|$)", x)],
            "cubin_sms": sms, "ptx_sms": ptx_sms, "has_sm75_cubin": "75" in sms, "has_sm50_or_older_cubin": [s for s in sms if s.isdigit() and int(s) < 75],
            "markers": {"cuda_12_toolkit_string": bool(re.search(rb"cuda_12\.|release 12\.\d|CUDA 12\.\d", data)),
                        "flashattention_symbols": low.count(b"flash_fwd") + low.count(b"vllm_flash_attn"), "flashinfer": low.count(b"flashinfer"), "flashmla": low.count(b"flash_mla") + low.count(b"flashmla")}})
o = res["objects"]
res["summary"] = {"native_objects": len(o), "all_x86_64": all("X86-64" in x["machine"] for x in o), "any_needs_cuda12_soname": any(x["needs_cuda12_soname"] for x in o),
  "any_cuda12_toolkit_string": any(x["markers"]["cuda_12_toolkit_string"] for x in o), "any_older_than_sm75_cubin": any(x["has_sm50_or_older_cubin"] for x in o),
  "objects_with_cubins": sum(bool(x["cubin_sms"]) for x in o), "objects_with_sm75": sum(x["has_sm75_cubin"] for x in o),
  "max_glibc": max((x["max_glibc"] for x in o if x["max_glibc"]), key=lambda s: [int(v) for v in s.split(".")], default=None),
  "any_flash_markers": any(x["markers"]["flashattention_symbols"] or x["markers"]["flashinfer"] or x["markers"]["flashmla"] for x in o)}
pathlib.Path(out).write_text(json.dumps(res, indent=1)); print(json.dumps(res["summary"], indent=1))
for x in o: print(f'{x["member"]:55s} cubins={x["cubin_sms"]} ptx={x["ptx_sms"]} glibc={x["max_glibc"]} cuda12_needed={x["needs_cuda12_soname"]} cudart={x["cuda_runtime_needed"]}')
