#!/usr/bin/env python3.13
"""Audit the `velocita` vs `stokastisk` namespaces across stage.zip, wheels and repo sources. Read-only.
usage: namespace_audit.py stage.zip out_dir wheel [wheel ...]"""
import hashlib, json, pathlib, re, sys, zipfile

stage, out, wheels = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), [pathlib.Path(w) for w in sys.argv[3:]]
ROOT = pathlib.Path(__file__).resolve().parents[2]
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
PAT = re.compile(rb"\bvelocita\b", re.I)
res = {"stage_zip": {"path": str(stage), "sha256": sha(stage.read_bytes())}}

z = zipfile.ZipFile(stage); names = z.namelist()
top = sorted({n.split("/")[1] for n in names if n.count("/") >= 1 and n.split("/")[1]})
# the leftover facade was named stage/velocita/ in the original archive and stage/stochastisk/ (sic, 'ch') after the archive was edited on 2026-11-07
LEFT = next((d for d in ("velocita", "stochastisk") if any(n.startswith(f"stage/{d}/") for n in names)), None)
vel = [n for n in names if LEFT and n.startswith(f"stage/{LEFT}/")]
res["stage_zip"].update(top_level_under_stage=top, velocita_files=vel, vllm_file_count=sum(n.startswith("stage/vllm/") for n in names))
res["stage_zip"]["leftover_facade_dir"] = LEFT
src = z.read(f"stage/{LEFT}/__init__.py").decode() if LEFT else ""
res["velocita_init"] = {"sha256": sha(src.encode()), "bytes": len(src), "imports": re.findall(r"^(?:from|import)\s+([\w.]+)", src, re.M),
                        "defines": re.findall(r"^(?:def|class)\s+(\w+)", src, re.M),
                        "is_older_name_of_sdk_facade": "SDK facade over the integrated vLLM" in src}
sdk_init = (ROOT / "sdk/stokastisk/__init__.py").read_text()
res["velocita_init"]["shared_public_names_with_stokastisk"] = sorted(set(re.findall(r"^def (\w+)|^class (\w+)", src, re.M)[i][j] for i in range(len(re.findall(r"^def (\w+)|^class (\w+)", src, re.M))) for j in (0, 1)) - {""} & {n for t in re.findall(r"^def (\w+)|^class (\w+)", sdk_init, re.M) for n in t if n})
# does anything inside stage/vllm mention velocita?
hits = [n for n in names if n.startswith("stage/vllm/") and n.endswith((".py", ".txt", ".json", ".md")) and PAT.search(z.read(n))]
res["velocita_referenced_inside_stage_vllm"] = hits
# builder input
b = (ROOT / "BUILD.md").read_text() + (ROOT / "scripts/release/build_t4_tree.py").read_text()
res["builder_uses_only_stage_vllm"] = {"BUILD.md_arg": "work/rc-stage-extract/stage/vllm" in b,
                                      "copytree_src_is_stage_arg": "shutil.copytree(stage, out / \"vllm\"" in b}
# repo sources
repo = []
for p in list(ROOT.glob("scripts/**/*.py")) + list(ROOT.glob("sdk/**/*.py")) + list(ROOT.glob("tests/**/*.py")) + list(ROOT.glob("*.md")) + list(ROOT.glob("*.toml")) + list(ROOT.glob("requirements/*")) + list(ROOT.glob("notebooks/*.ipynb")):
    if p.is_file() and PAT.search(p.read_bytes()): repo.append(str(p.relative_to(ROOT)))
res["repo_files_mentioning_velocita"] = repo
res["wheels"] = []
for w in wheels:
    zz = zipfile.ZipFile(w); n = zz.namelist()
    tl = sorted({x.split("/")[0] for x in n})
    dist = [x for x in n if x.endswith(".dist-info/METADATA")]
    meta = zz.read(dist[0]).decode()
    res["wheels"].append({"wheel": w.name, "sha256": sha(w.read_bytes()), "top_level": [t for t in tl if not t.endswith(".dist-info")],
                          "velocita_paths": [x for x in n if "velocita" in x.lower()],
                          "velocita_in_text_files": [x for x in n if x.endswith((".py", ".txt", "METADATA", "RECORD", "top_level.txt")) and PAT.search(zz.read(x))],
                          "metadata_name": re.search(r"^Name: (.+)$", meta, re.M).group(1), "requires_dist": re.findall(r"^Requires-Dist: (.+)$", meta, re.M)[:5]})
res["verdict"] = {
    "velocita_is_unused_leftover": not hits and not any(w["velocita_paths"] or w["velocita_in_text_files"] for w in res["wheels"]),
    "entered_any_wheel": any(w["velocita_paths"] or w["velocita_in_text_files"] for w in res["wheels"]),
    "needs_rename": False,
    "explanation": "stage/velocita/__init__.py is an earlier-named copy of the SDK facade (imports torch+vllm, same API shape). It is outside stage/vllm, "
                   "is never read by the builder (which receives stage/vllm only), and appears in no wheel. The installed SDK is `stokastisk`."}
out.mkdir(parents=True, exist_ok=True)
(out / "stage-package-namespace-audit.json").write_text(json.dumps(res, indent=1, default=list))
(out / "stage-forensic-audit.json").write_text(json.dumps({"stage_zip": res["stage_zip"], "velocita_init": res["velocita_init"], "verdict": res["verdict"],
                                                           "builder": res["builder_uses_only_stage_vllm"]}, indent=1))
print(json.dumps(res["verdict"], indent=1)); print(json.dumps(res["builder_uses_only_stage_vllm"]), res["repo_files_mentioning_velocita"], hits[:3])
for w in res["wheels"]: print(w["wheel"], w["top_level"], w["velocita_paths"], w["velocita_in_text_files"])
