"""Static scan: module-level `import vllm.x` / `from vllm.x import y` that point at files absent from the tree.
usage: unresolved_imports.py tree_dir  -> JSON {module_level: {...}, lazy: {...}}"""
import ast, json, pathlib, sys
root = pathlib.Path(sys.argv[1]); present = set()
for p in (root / "vllm").rglob("*.py"):
    rel = p.relative_to(root).with_suffix("")
    present.add(".".join(rel.parts[:-1] if rel.name == "__init__" else rel.parts))
for p in (root / "vllm").rglob("*"):
    if p.is_dir(): present.add(".".join(p.relative_to(root).parts))
def ok(m, names=()):
    if m in present: return True
    return False
res = {"module_level": {}, "lazy": {}}
for p in (root / "vllm").rglob("*.py"):
    try: tree = ast.parse(p.read_text())
    except SyntaxError: continue
    def walk(nodes, lazy):
        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)): walk(n.body, True); continue
            if isinstance(n, ast.If) and "TYPE_CHECKING" in ast.dump(n.test): continue
            mods = []
            if isinstance(n, ast.Import): mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                mods = [n.module] + [f"{n.module}.{a.name}" for a in n.names]
                # `from pkg import name` is fine if pkg exists and name is an attribute; only flag pkg itself
                mods = [n.module] if all(not ok(f"{n.module}.{a.name}") for a in n.names) else mods[:1]
            for m in mods:
                if m.split(".")[0] == "vllm" and not ok(m):
                    res["lazy" if lazy else "module_level"].setdefault(str(p.relative_to(root)), []).append(m)
            for f in ("body", "orelse", "finalbody"):
                s = getattr(n, f, None)
                if s: walk(s, lazy)
            for h in getattr(n, "handlers", []): walk(h.body, lazy)
    walk(tree.body, False)
print(json.dumps(res, indent=1))
