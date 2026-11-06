#!/usr/bin/env python3.13
"""Assemble the T4-only vllm tree.
 stage python tree
  - vllm/vllm_flash_attn/, vllm/third_party/flashmla/            (removed wholesale)
  - any module with a MODULE-LEVEL import of flashinfer / flash_attn / vllm.vllm_flash_attn / flashmla  (removed; AST-detected)
  + our CUDA13/SM75 .so files
  + patches/*.patch applied with `patch -p1`
Inert wrappers that only probe lazily (e.g. vllm/utils/flashinfer.py) are kept and listed in KEPT.json.
usage: build_t4_tree.py stage_vllm_dir ours_wheel_extract_dir out_dir patches_dir"""
import ast, json, pathlib, shutil, subprocess, sys
stage, ours, out, patches = map(pathlib.Path, sys.argv[1:5])
if out.exists(): raise SystemExit(f"{out} exists")
shutil.copytree(stage, out / "vllm", ignore=shutil.ignore_patterns("__pycache__", "*.so"))
for so in ("_C_stable_libtorch", "_moe_C_stable_libtorch", "cumem_allocator", "spinloop", "fs_io_C", "_rust_tool_parser"):
    shutil.copy2(ours / "vllm" / f"{so}.abi3.so", out / "vllm")
BAD = ("flashinfer", "flash_attn", "vllm.vllm_flash_attn", "flashmla", "vllm.third_party.flashmla", "flash_mla")
def bad(mod): return mod and any(mod == b or mod.startswith(b + ".") for b in BAD)
def module_level_bad(tree):
    hits = []
    # note: try/except guarded imports at module level are still counted (conservative)
    def visit(nodes):
        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
            if isinstance(n, ast.If) and "TYPE_CHECKING" in ast.dump(n.test): continue
            if isinstance(n, ast.Import):
                hits.extend(a.name for a in n.names if bad(a.name))
            elif isinstance(n, ast.ImportFrom):
                if bad(n.module): hits.append(n.module)
            for f in ("body", "orelse", "finalbody", "handlers"):
                sub = getattr(n, f, None)
                if sub: visit([h for h in sub] if f != "handlers" else [x for h in sub for x in h.body])
    visit(tree.body); return hits
removed, kept_flash_named = [], []
patched = set()
for pf in sorted(patches.glob("*.patch")):
    subprocess.run(["patch", "-p1", "-d", str(out), "-i", str(pf.resolve())], check=True, capture_output=True)
    patched |= {l.split()[1][len("b/vllm/"):] for l in pf.read_text().splitlines() if l.startswith("+++ ")}
for d in ("vllm_flash_attn", "third_party/flashmla"):
    if (out / "vllm" / d).exists():
        removed += [str(q.relative_to(out)) for q in (out / "vllm" / d).rglob("*") if q.is_file()]
        shutil.rmtree(out / "vllm" / d)
for p in sorted((out / "vllm").rglob("*.py")):
    try: hits = module_level_bad(ast.parse(p.read_text()))
    except SyntaxError: hits = []
    if hits and str(p.relative_to(out / 'vllm')) in patched: kept_flash_named.append(str(p.relative_to(out)) + '  (patched, guarded)')
    elif hits: removed.append(str(p.relative_to(out)) + "  <- " + ",".join(sorted(set(hits)))); p.unlink()
    elif any(k in p.name for k in ("flash", "fa_utils")): kept_flash_named.append(str(p.relative_to(out)))
(out / "REMOVED.json").write_text(json.dumps(removed, indent=1)); (out / "KEPT.json").write_text(json.dumps(kept_flash_named, indent=1))
print(len(removed), "removed;", len(kept_flash_named), "flash-named modules kept (inert)")
