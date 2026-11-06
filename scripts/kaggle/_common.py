"""Shared helpers for the Stokastisk Kaggle acceptance scripts."""
import hashlib, importlib, json, os, pathlib, subprocess, sys, time, traceback

# Statuses: PASS FAIL ERROR SKIPPED WARNING NOT_RUN. Only these never fail a stage; FAIL/ERROR/NOT_RUN do.
OK_STATUSES = ("PASS", "WARNING", "SKIPPED")
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
    # WARNING/SKIPPED = a documented limitation (e.g. cuobjdump absent); it never hides a FAIL and never counts as a verified PASS.
    status = "PASS" if results and all(r["status"] in OK_STATUSES for r in results) else "FAIL"
    save(f"{stage}.json", {"stage": stage, "status": status, "checks": results, **(extra or {})})
    print(f"\n[{stage}] OVERALL: {status}")
    if status != "PASS":
        sys.exit(1)


def warn(results, name, detail=""):
    results.append({"check": name, "status": "WARNING", "detail": str(detail)})
    print(f"WARNING {name} -- {detail}")


def skipped(results, name, detail=""):
    results.append({"check": name, "status": "SKIPPED", "detail": str(detail)})
    print(f"SKIPPED {name} -- {detail}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_installed(stage, modules=("stokastisk", "vllm")):
    """Kaggle runs every notebook in a FRESH session: packages installed by notebook 02 do not exist in notebooks 03-06.
    Fail early, with machine-readable evidence and the fix, instead of an anonymous ModuleNotFoundError."""
    R, missing = [], {}
    for m in modules:
        try:
            importlib.import_module(m)
            check(R, f"import {m}", True)
        except Exception as e:  # noqa: BLE001
            missing[m] = traceback.format_exc()
            check(R, f"import {m}", False, f"{type(e).__name__}: {e}")
    if missing:
        hint = ("Kaggle starts each notebook in a fresh session; install first with "
                "`python scripts/kaggle/02_install.py <bundle>` (idempotent). The generated notebook 03-06 install cell does this.")
        print("\n" + hint)
        finish(stage, R, {"missing_modules": missing, "hint": hint, "script": os.path.abspath(sys.argv[0]),
                          "script_sha256": sha256(sys.argv[0])})
