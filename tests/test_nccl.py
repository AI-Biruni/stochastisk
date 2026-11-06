#!/usr/bin/env python3.13
"""Notebook 04 regression tests (CPU only). NCCL itself needs two T4s and can only be accepted on Kaggle; here we test the harness:
the Gloo *control* run proves launcher/worker/summary plumbing, timeouts, and that a control run can never count as acceptance."""
import json, os, pathlib, shutil, subprocess, sys, tempfile, unittest, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
K = ROOT / "scripts" / "kaggle"
BASE = dict(os.environ, PYTHONPATH=str(ROOT / "sdk"), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")


def run(script, out, **env):
    return subprocess.run([sys.executable, str(K / script)], cwd=K, env=dict(BASE, STOKASTISK_OUT=str(out), **env), capture_output=True, text=True, timeout=240)


class NcclHarness(unittest.TestCase):
    def test_gloo_control_run_is_correct_but_never_acceptance(self):
        with tempfile.TemporaryDirectory() as d:
            r = run("04_nccl_launch.py", d, STOKASTISK_NCCL_CONTROL="gloo")
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            for rk in (0, 1):
                j = json.load(open(f"{d}/04_nccl_rank{rk}.json"))
                self.assertEqual(j["schema_version"], 2); self.assertFalse(j["acceptance_valid"]); self.assertEqual(j["backend"], "gloo")
                by = {o["name"]: o["status"] for o in j["collectives"]}
                for n in ("all_reduce_sum", "all_gather", "broadcast", "barrier"):
                    self.assertEqual(by[n], "PASS", (n, j["collectives"]))      # real Gloo collectives with exact expected values
                self.assertEqual(j["dist_initialized_after_teardown"], False)
            s = run("04_nccl_summary.py", d)
            self.assertEqual(s.returncode, 1)                                       # control run must FAIL acceptance
            doc = json.load(open(f"{d}/04_nccl_dual_gpu.json"))
            self.assertEqual(doc["status"], "FAIL")
            self.assertTrue(any("never acceptance" in c["check"] and c["status"] == "FAIL" for c in doc["checks"]))
            self.assertTrue(zipfile.ZipFile(f"{d}/04_nccl_acceptance.zip").namelist())

    def test_parent_timeout_is_a_failure_and_kills_workers(self):
        with tempfile.TemporaryDirectory() as d:
            r = run("04_nccl_launch.py", d, STOKASTISK_NCCL_CONTROL="gloo", STOKASTISK_NCCL_PARENT_TIMEOUT_S="0.5")
            self.assertEqual(r.returncode, 1)
            l = json.load(open(f"{d}/04_nccl_launch.json"))
            self.assertTrue(l["timed_out"])
            s = run("04_nccl_summary.py", d); self.assertEqual(s.returncode, 1)
            self.assertTrue(any(c["check"] == "parent did not time out" and c["status"] == "FAIL" for c in json.load(open(f"{d}/04_nccl_dual_gpu.json"))["checks"]))

    def test_summary_rejects_the_real_kaggle_rc2_output(self):
        """The exact rank JSONs from the failed Kaggle run (stale SDK wheel): collectives ok but required fields absent -> FAIL, with the cause named."""
        with tempfile.TemporaryDirectory() as d:
            for rk in (0, 1):
                shutil.copy(ROOT / "tests/data" / f"kaggle_rc2_nccl_rank{rk}.json", f"{d}/04_nccl_rank{rk}.json")
            json.dump({"returncode": 0, "timed_out": False, "timeout_s": 300, "duration_s": 5, "transport_overrides_in_env": {}}, open(f"{d}/04_nccl_launch.json", "w"))
            r = run("04_nccl_summary.py", d); self.assertEqual(r.returncode, 1)
            checks = {c["check"]: c["status"] for c in json.load(open(f"{d}/04_nccl_dual_gpu.json"))["checks"]}
            self.assertEqual(checks["rank0 result schema_version >= 2 (installed SDK is current)"], "FAIL")

    def test_no_forced_nccl_transport_anywhere_in_the_primary_path(self):
        for f in (K / "04_nccl_dual_gpu.py", K / "04_nccl_launch.py", ROOT / "sdk/stokastisk/nccl.py"):
            src = f.read_text()
            for var in ("NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_IB_DISABLE"):
                for line in src.splitlines():
                    self.assertFalse(var in line and ("setdefault" in line or "environ[" in line and "=" in line.split("environ[")[1]) and not line.lstrip().startswith("#"), f"{f.name}: {line}")

    def test_sdk_source_has_required_teardown_fields(self):
        s = (ROOT / "sdk/stokastisk/nccl.py").read_text()
        for k in ("dist_initialized_after_teardown", "gpu_mem_used_MiB_after_teardown", "gpu_mem_used_MiB_baseline", "SCHEMA_VERSION"):
            self.assertIn(k, s)


if __name__ == "__main__":
    unittest.main()
