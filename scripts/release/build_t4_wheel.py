#!/usr/bin/env python3.13
"""Build the T4-only vLLM wheel from work/rc-tree with rewritten, minimal metadata.
usage: build_t4_wheel.py tree_dir orig_wheel out_dir"""
import base64, csv, hashlib, io, pathlib, re, sys, zipfile
tree, orig, outdir = map(pathlib.Path, sys.argv[1:4])
oz = zipfile.ZipFile(orig); ometa = oz.read("vllm-0.25.1.dist-info/METADATA").decode()
head, _, rest = ometa.partition("\nRequires-Dist:")
head_lines = [l for l in head.splitlines() if not l.startswith("Requires-Dist")]
upstream = {}
for l in ometa.splitlines():
    if l.startswith("Requires-Dist: "):
        r = l[15:]; nm = re.match(r"[A-Za-z0-9_.\-]+", r).group(0); upstream.setdefault(re.sub(r"[-_.]+", "-", nm).lower(), r)
def req(n):
    r = upstream[n]; return r.split(";")[0].strip()
# import-proven & preinstalled-on-Kaggle (ranges from upstream metadata, never pinned by us)
PRESENT = ["regex", "cachetools", "psutil", "sentencepiece", "numpy", "requests", "tqdm", "py-cpuinfo", "transformers", "tokenizers",
           "safetensors", "fastapi", "starlette", "aiohttp", "openai", "pydantic", "prometheus-client", "pillow", "tiktoken", "typing-extensions",
           "filelock", "pyzmq", "pyyaml", "einops", "cloudpickle", "uvloop", "opentelemetry-sdk", "opentelemetry-api"]
ABSENT = ["apache-tvm-ffi", "xgrammar", "llguidance", "cbor2", "msgspec", "openai-harmony", "partial-json-parser", "pybase64"]
core, missing = [], []
for n in PRESENT + ABSENT:
    (core if n in upstream else missing).append(req(n) if n in upstream else n)
core += missing  # names imported by vLLM but absent from upstream metadata (e.g. uvloop via fastapi[standard])
core = [c.replace("fastapi[standard]", "fastapi").replace("fastapi<0.137.0,>=0.133.0", "fastapi>=0.133.0") for c in core]  # deviation: tested with Kaggle fastapi 0.141.1
extras = {"server": ["fastapi-cli", "sse-starlette", "pydantic-settings", "model-hosting-container-standards", "prometheus-fastapi-instrumentator"],
          "observability": ["opentelemetry-exporter-otlp", "opentelemetry-semantic-conventions-ai"],
          "models": ["compressed-tensors", "mistral_common[image]", "lm-format-enforcer", "outlines_core", "diskcache", "interegular", "depyf", "ijson", "setproctitle"]}
lines = head_lines + [f"Requires-Dist: {c}" for c in core]
for ex, names in extras.items():
    lines.append(f"Provides-Extra: {ex}")
    for n in names:
        key = re.sub(r"[-_.]+", "-", re.sub(r"\[.*", "", n)).lower()
        r = upstream.get(key, n).split(";")[0].strip()
        lines.append(f'Requires-Dist: {r} ; extra == "{ex}"')
meta = "\n".join(lines) + "\n\nStokastisk T4/Triton-only derivative of vLLM 0.25.1 (SM75, CUDA 13.0). FlashAttention and FlashInfer are not included.\n"
files = {}
for p in sorted((tree / "vllm").rglob("*")):
    if p.is_file() and "__pycache__" not in p.parts: files[str(p.relative_to(tree))] = p.read_bytes()
di = "vllm-0.25.1.dist-info"
for n in oz.namelist():
    if n.startswith(di + "/") and n.split("/")[-1] in ("entry_points.txt", "top_level.txt") or n.startswith(di + "/licenses/"):
        files[n] = oz.read(n)
files[f"{di}/METADATA"] = meta.encode()
files[f"{di}/WHEEL"] = b"Wheel-Version: 1.0\nGenerator: stokastisk-build_t4_wheel\nRoot-Is-Purelib: false\nTag: cp38-abi3-linux_x86_64\n"
def h(b): return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(b).digest()).rstrip(b"=").decode()
buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
for n, b in files.items(): w.writerow((n, h(b), len(b)))
w.writerow((f"{di}/RECORD", "", ""))
files[f"{di}/RECORD"] = buf.getvalue().encode()
outdir.mkdir(parents=True, exist_ok=True); dst = outdir / "vllm-0.25.1-cp38-abi3-linux_x86_64.whl"
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n, b in files.items():
        zi = zipfile.ZipInfo(n, (2026, 11, 6, 0, 0, 0)); zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = (0o755 if n.endswith(".so") else 0o644) << 16; z.writestr(zi, b)
print(dst, dst.stat().st_size, "core:", len(core), "missing-from-upstream:", missing)
