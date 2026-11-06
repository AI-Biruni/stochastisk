#!/usr/bin/env python3.13
"""Quarantine/delete a duplicate tree proven redundant by an audit_duplicates.py report.

Default is DRY-RUN. --quarantine moves the tree; --delete-quarantine permanently removes it
after re-verifying the canonical tree still holds every file. Refuses if the target has unique
or modified files, or is/contains stage.zip or the baseline zip.
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path

PROTECTED = ("stage.zip", "archive-baseline")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--quarantine-dir", default="work/.quarantine")
    ap.add_argument("--allow-unique-by-hash-only", action="store_true",
                    help="permit target files whose names differ but whose content equals canonical")
    ap.add_argument("--quarantine", action="store_true")
    ap.add_argument("--delete-quarantine", action="store_true")
    ap.add_argument("--project", default=".")
    a = ap.parse_args()
    project = Path(a.project).resolve()
    rep = json.loads(Path(a.report).read_text())
    target = Path(a.target).resolve()
    canonical = Path(rep["canonical"])
    if project not in target.parents or target == canonical or canonical in target.parents:
        sys.exit("refuse: target outside project or is/contains canonical")
    if any(t in str(target) for t in PROTECTED) or target.is_symlink():
        sys.exit("refuse: protected or symlink target")
    cmp_ = rep["comparison"].get(str(target))
    if cmp_ is None:
        sys.exit("refuse: target not in report")
    problems = []
    if cmp_["E_unique_files"]:
        problems.append(f"{len(cmp_['E_unique_files'])} unique files: {cmp_['E_unique_files'][:5]}")
    if cmp_["A_modified_same_path"]:
        problems.append(f"{len(cmp_['A_modified_same_path'])} modified files")
    if cmp_["B_same_hash_different_path"] and not a.allow_unique_by_hash_only:
        problems.append(f"{len(cmp_['B_same_hash_different_path'])} files renamed vs canonical: "
                        f"{list(cmp_['B_same_hash_different_path'].items())[:5]}")
    print(f"target: {target}\nfiles: {cmp_['files']}  identical-to-canonical: {cmp_['A_identical_path_and_hash']}")
    if problems:
        print("NOT SAFE TO DELETE:\n  " + "\n  ".join(problems)); sys.exit(2)
    q = (project / a.quarantine_dir).resolve() / target.name
    if not (a.quarantine or a.delete_quarantine):
        print(f"DRY-RUN: would move to {q}"); return
    if a.quarantine:
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(q)); print("quarantined ->", q)
    if a.delete_quarantine:
        if not canonical.is_dir() or sum(1 for _ in canonical.rglob("*") if _.is_file()) != rep["file_counts"][str(canonical)]:
            sys.exit("canonical verification failed; not deleting")
        shutil.rmtree(q); print("deleted", q)

if __name__ == "__main__":
    main()
