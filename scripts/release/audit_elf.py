#!/usr/bin/env python3.13
"""Audit every ELF in an extracted wheel dir: NEEDED, RUNPATH, GLIBC/GLIBCXX max, CUDA archs, FA presence.
usage: audit_elf.py <extracted-dir> <out-prefix>   (locally generated artifacts only; uses readelf/cuobjdump/ldd)"""
import json, os, re, subprocess, sys, hashlib
root, out = sys.argv[1], sys.argv[2]
def sh(*a, env=None):
    return subprocess.run(a, capture_output=True, text=True, env=env).stdout + ""
def vkey(s): return tuple(int(x) for x in re.findall(r"\d+", s))
try:
    import torch; tlib = os.path.join(os.path.dirname(torch.__file__), "lib")
except Exception: tlib = ""
env = dict(os.environ, LD_LIBRARY_PATH=tlib + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
elf, archs = {}, {}
for d, _, fs in os.walk(root):
    for f in sorted(fs):
        p = os.path.join(d, f)
        if os.path.islink(p) or not os.path.isfile(p): continue
        if open(p, "rb").read(4) != b"\x7fELF": continue
        rel = os.path.relpath(p, root)
        dyn = sh("readelf", "-d", p); ver = sh("readelf", "-V", p)
        glibc = sorted(set(re.findall(r"GLIBC_([\d.]+)", ver)), key=vkey)
        cxx = sorted(set(re.findall(r"GLIBCXX_([\d.]+)", ver)), key=vkey)
        cxxabi = sorted(set(re.findall(r"CXXABI_([\d.]+)", ver)), key=vkey)
        needed = re.findall(r"NEEDED.*\[(.*)\]", dyn)
        ldd = [l.strip() for l in sh("ldd", p, env=env).splitlines() if "not found" in l]
        cu = sh("cuobjdump", "--list-elf", p) + sh("cuobjdump", "--list-ptx", p)
        a = sorted(set(re.findall(r"(?:sm|compute)_\d+[a-z]?", cu)))
        elf[rel] = dict(size=os.path.getsize(p), sha256=hashlib.sha256(open(p, "rb").read()).hexdigest(),
            file=sh("file", "-b", p).strip(), soname=re.findall(r"SONAME.*\[(.*)\]", dyn),
            needed=needed, runpath=re.findall(r"(?:RUNPATH|RPATH).*\[(.*)\]", dyn),
            glibc_max=glibc[-1] if glibc else None, glibcxx_max=cxx[-1] if cxx else None,
            cxxabi_max=cxxabi[-1] if cxxabi else None, links_cuda12=any("cuda" in n and ".so.12" in n for n in needed),
            links_cudart13=any(n.startswith("libcudart.so.13") for n in needed),
            unresolved_with_torch_libpath=ldd, cuda_archs=a)
        archs[rel] = a
json.dump(elf, open(out + "ELF-DEPENDENCY-MANIFEST.json", "w"), indent=1)
json.dump({"target": "sm_75", "per_file": archs,
           "non_sm75": {k: v for k, v in archs.items() if any(x not in ("sm_75", "compute_75") for x in v)}},
          open(out + "CUDA-ARCH-MANIFEST.json", "w"), indent=1)
print(json.dumps({k: dict(archs=v["cuda_archs"], cuda12=v["links_cuda12"], cudart13=v["links_cudart13"],
      glibc=v["glibc_max"], glibcxx=v["glibcxx_max"], unresolved=v["unresolved_with_torch_libpath"]) for k, v in elf.items()}, indent=1))
