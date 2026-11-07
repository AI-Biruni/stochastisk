"""Forensic summary of executed Kaggle notebooks + acceptance ZIPs (read-only on the originals).
usage: python3.13 -I forensic_notebooks.py EXECUTED_DIR CANONICAL_DIR OUT_JSON EXTRACT_ROOT"""
import hashlib, json, re, sys, zipfile, pathlib, collections

ANSI = re.compile(r"\x1b\[[0-9;]*m")
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def nb_report(path, canon_dir):
    nb = json.loads(pathlib.Path(path).read_text())
    cells, errors, first_fail, texts = [], [], None, []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code": continue
        cells.append({"index": i, "execution_count": c.get("execution_count"), "has_output": bool(c.get("outputs"))})
        for o in c.get("outputs", []):
            if o["output_type"] == "error":
                e = {"cell": i, "execution_count": c.get("execution_count"), "ename": o["ename"], "evalue": o["evalue"],
                     "traceback": ANSI.sub("", "\n".join(o["traceback"]))}
                errors.append(e); first_fail = first_fail or e
            elif o["output_type"] == "stream": texts.append(ANSI.sub("", "".join(o["text"])))
    full = "\n".join(texts)
    after = [c for c in cells if first_fail and c["index"] > first_fail["cell"] and c["execution_count"] is not None]
    # code comparison with the canonical notebook: normalise by stripping leading '!' magics is unnecessary; compare sources
    stem = pathlib.Path(path).name.split("-")[0]
    canon = next(iter(pathlib.Path(canon_dir).glob(f"{stem}_*.ipynb")), None)
    cmp = None
    if canon:
        cs = [("".join(c["source"])) for c in json.loads(canon.read_text())["cells"] if c["cell_type"] == "code"]
        es = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
        cmp = {"canonical": canon.name, "code_cells_executed": len(es), "code_cells_canonical": len(cs),
               "identical_code": cs == es,
               "differing_cells": [i for i, (a, b) in enumerate(zip(cs, es)) if a != b]}
    return {"file": pathlib.Path(path).name, "sha256": sha(path), "code_cells": len(cells),
            "errors": errors, "first_failing_cell": first_fail["cell"] if first_fail else None,
            "executed_cells_after_first_failure": [c["index"] for c in after],
            "bundle_paths": sorted(set(re.findall(r"/kaggle/input/\S*?kaggle-input/[^\s/]+", full)))[:5],
            "wheel_sha256_printed": sorted(set(re.findall(r"\b[0-9a-f]{64}\b(?=\s+\S+\.whl)", full))),
            "overall_lines": re.findall(r"\[\w+\] OVERALL: (\w+)", full),
            "fail_lines": [l for l in full.splitlines() if l.startswith(("FAIL", "ERROR"))][:50],
            "canonical_comparison": cmp}

def zip_report(path, extract_root):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    dups = [n for n, k in collections.Counter(names).items() if k > 1]
    out = {"file": pathlib.Path(path).name, "sha256": sha(path), "size": pathlib.Path(path).stat().st_size,
           "members": len(names), "testzip_bad_member": z.testzip(), "duplicate_paths": dups, "json": {}, "logs": []}
    dest = pathlib.Path(extract_root) / pathlib.Path(path).stem
    dest.mkdir(parents=True, exist_ok=False)
    z.extractall(dest)
    for n in names:
        if n.endswith("/"): continue
        if n.endswith(".json"):
            try:
                d = json.loads(z.read(n)); out["json"][n] = {"status": d.get("status", d.get("overall")) if isinstance(d, dict) else None,
                    "ok": d.get("ok") if isinstance(d, dict) else None}
            except Exception as e: out["json"][n] = {"parse_error": str(e)}
        elif n.endswith(".log"): out["logs"].append({"name": n, "bytes": len(z.read(n))})
    out["extracted_to"] = str(dest)
    return out

if __name__ == "__main__":
    ex, canon, out, root = sys.argv[1:5]
    res = {"notebooks": [], "zips": []}
    for p in sorted(pathlib.Path(ex).glob("*.ipynb")): res["notebooks"].append(nb_report(p, canon))
    for p in sorted(pathlib.Path(ex).glob("*.zip")): res["zips"].append(zip_report(p, root))
    pathlib.Path(out).write_text(json.dumps(res, indent=1))
    for n in res["notebooks"]:
        print(n["file"], "errors:", len(n["errors"]), "first_fail:", n["first_failing_cell"], "overall:", n["overall_lines"], "fails:", len(n["fail_lines"]),
              "canon_identical:", (n["canonical_comparison"] or {}).get("identical_code"))
    for z in res["zips"]: print(z["file"], z["sha256"][:12], "members", z["members"], "bad", z["testzip_bad_member"], "dups", z["duplicate_paths"])
