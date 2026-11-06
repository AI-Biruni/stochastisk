"""Shared helpers for the Stokastisk Kaggle acceptance scripts."""
import json, os, pathlib, subprocess, sys, time

OUT = pathlib.Path(os.environ.get("STOKASTISK_OUT", "/kaggle/working/stokastisk-acceptance"))
OUT.mkdir(parents=True, exist_ok=True)


def save(name, obj):
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, default=str))
    return p


def sh(cmd, **kw):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, **kw)
    return (r.stdout + r.stderr).strip()


def check(results, name, ok, detail=""):
    results.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)})
    print(("PASS " if ok else "FAIL ") + name + (f" -- {detail}" if detail else ""))
    return ok


def finish(stage, results, extra=None):
    status = "PASS" if results and all(r["status"] == "PASS" for r in results) else "FAIL"
    save(f"{stage}.json", {"stage": stage, "status": status, "checks": results, **(extra or {})})
    print(f"\n[{stage}] OVERALL: {status}")
    if status != "PASS":
        sys.exit(1)
