#!/usr/bin/env python3.13
"""The Kaggle hard limits (4 CPU, 30 GiB RAM, 20 GiB /kaggle/working, 2 x T4 SM75 with 15 GiB VRAM) live in stokastisk.limits and are enforced/reported."""
import os, pathlib, sys, tempfile, unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdk"))
from stokastisk import limits as L  # noqa: E402
from stokastisk.config import StokastiskConfig  # noqa: E402


class Limits(unittest.TestCase):
    def test_constants_match_kaggle(self):
        self.assertEqual((L.LIMITS.cpu_cores, L.LIMITS.system_ram_gib, L.LIMITS.working_dir_gib, L.LIMITS.gpu_count, L.LIMITS.gpu_vram_gib), (4, 30.0, 20.0, 2, 15.0))
        self.assertEqual(L.LIMITS.gpu_compute_capability, (7, 5)); self.assertEqual(L.LIMITS.working_dir, "/kaggle/working")

    def test_cpu_budget_never_exceeds_four_and_follows_nproc(self):
        self.assertLessEqual(L.cpu_budget(), 4); self.assertGreaterEqual(L.cpu_budget(), 1)
        self.assertEqual(L.available_cpus(), len(os.sched_getaffinity(0)))

    def test_threads_split_across_tp_workers(self):
        orig = L.cpu_budget
        try:
            L.cpu_budget = lambda: 4
            self.assertEqual(L.threads_per_worker(1), 4); self.assertEqual(L.threads_per_worker(2), 2)
            self.assertEqual(set(L.cpu_thread_env(2).values()), {"2"})
        finally:
            L.cpu_budget = orig

    def test_config_exports_thread_env_but_user_wins(self):
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.pop(k, None)
        StokastiskConfig(tensor_parallel_size=2).apply_environment()
        self.assertEqual(os.environ["OMP_NUM_THREADS"], str(L.threads_per_worker(2)))
        os.environ["OMP_NUM_THREADS"] = "7"; StokastiskConfig().apply_environment(); self.assertEqual(os.environ["OMP_NUM_THREADS"], "7")
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ.pop(k, None)

    def test_working_dir_status_and_quota(self):
        with tempfile.TemporaryDirectory() as d:
            pathlib.Path(d, "a.bin").write_bytes(b"x" * 1024)
            s = L.working_dir_status(d); self.assertEqual(s["status"], "PASS"); self.assertTrue(s["present"])
            self.assertEqual(L.working_dir_status(d + "/missing")["status"], "SKIPPED")

    def test_vram_utilisation_fits_a_15gib_t4(self):
        self.assertLessEqual(StokastiskConfig().gpu_memory_utilization * L.LIMITS.gpu_vram_gib, 15.0)

    def test_check_resources_reports_every_limit(self):
        r = L.check_resources()
        for k in ("limits", "cpu", "ram", "working_dir", "gpus"): self.assertIn(k, r)


if __name__ == "__main__":
    unittest.main()
