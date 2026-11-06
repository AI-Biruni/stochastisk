#!/usr/bin/env python3.13
"""Audit duplicate files/directory trees. Read-only. Writes JSON report + summary.

usage: audit_duplicates.py ROOT [ROOT...] --canonical DIR --report out.json [--manifest-dir DIR]
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from collections import defaultdict
from pathlib import Path

PROTECTED_NAMES = {"stage.zip"}

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def walk_files(root: Path, project: Path):
    """Yield regular files; never follow symlinks; skip anything resolving outside project."""
    for dp, dns, fns in os.walk(root, followlinks=False):
        dns[:] = [d for d in dns if not (Path(dp, d).is_symlink())]
        for fn in fns:
            p = Path(dp, fn)
            if p.is_symlink() or not p.is_file():
                continue
            if project not in p.resolve().parents:
                continue
            yield p

def tree_hash(entries: dict[str, str]) -> str:
    h = hashlib.sha256()
    for rel in sorted(entries):
        h.update(f"{rel}\0{entries[rel]}\n".encode())
    return h.hexdigest()

def build_inventory(root: Path, project: Path):
    inv = {}
    for p in walk_files(root, project):
        st = p.stat()
        inv[str(p.relative_to(root))] = {"size": st.st_size, "sha256": sha256(p), "mtime": st.st_mtime}
    return inv

def classify(roots: dict[str, dict], canonical: str):
    can = {k: v["sha256"] for k, v in roots[canonical].items()}
    out = {}
    for name, inv in roots.items():
        if name == canonical:
            continue
        same_path_same = [k for k, v in inv.items() if k in can and can[k] == v["sha256"]]
        same_path_diff = [k for k, v in inv.items() if k in can and can[k] != v["sha256"]]
        by_hash = defaultdict(list)
        for k, v in can.items():
            by_hash[v].append(k)
        diff_path_same = {k: by_hash[v["sha256"]] for k, v in inv.items()
                          if k not in can and v["sha256"] in by_hash}
        unique = [k for k, v in inv.items() if k not in can and v["sha256"] not in by_hash]
        out[name] = {
            "files": len(inv),
            "A_identical_path_and_hash": len(same_path_same),
            "A_modified_same_path": same_path_diff,
            "B_same_hash_different_path": diff_path_same,
            "E_unique_files": unique,
            "exact_duplicate_tree": (not same_path_diff and not diff_path_same and not unique
                                     and len(same_path_same) == len(inv) == len(can)),
            "subset_of_canonical": (not same_path_diff and not diff_path_same and not unique),
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="tree roots to compare (name=path or path)")
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--project", default=".")
    a = ap.parse_args()
    project = Path(a.project).resolve()
    roots = {}
    for r in a.roots:
        p = Path(r).resolve()
        roots[str(p)] = build_inventory(p, project)
    canonical = str(Path(a.canonical).resolve())
    if canonical not in roots:
        sys.exit("canonical must be among roots")
    rep = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "canonical": canonical,
        "tree_hashes": {k: tree_hash({r: v["sha256"] for r, v in inv.items()}) for k, inv in roots.items()},
        "file_counts": {k: len(v) for k, v in roots.items()},
        "comparison": classify(roots, canonical),
    }
    Path(a.report).write_text(json.dumps(rep, indent=2))
    print(f"canonical: {canonical} ({len(roots[canonical])} files)")
    for k, c in rep["comparison"].items():
        print(f"- {k}: {c['files']} files; exact_tree={c['exact_duplicate_tree']} subset={c['subset_of_canonical']} "
              f"unique={len(c['E_unique_files'])} modified={len(c['A_modified_same_path'])} "
              f"renamed_dups={len(c['B_same_hash_different_path'])}")

if __name__ == "__main__":
    main()
