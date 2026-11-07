#!/usr/bin/env python3.13
"""Regression tests for the Kaggle notebook 05 failure (both TP=1 and TP=2 exited 1, no result JSON).

Root cause (from vllm_tp1.log): vLLM forces the `spawn` start method; spawn re-imports the launching script as `__mp_main__`;
05_inference.py / 06_tp_benchmark.py ran the engine at MODULE level, so the child tried to start its own processes -> RuntimeError
'An attempt has been made to start a new process before the current process has finished its bootstrapping phase'."""
import ast, json, os, pathlib, subprocess, sys, tempfile, unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
K = ROOT / "scripts" / "kaggle"
ENV = dict(os.environ, PYTHONPATH=str(K), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
sys.path.insert(0, str(K))
os.environ.setdefault("STOKASTISK_OUT", tempfile.mkdtemp(prefix="stk-test-"))   # _common creates its output dir at import


def top_level_nodes(name):
    return ast.parse((K / name).read_text()).body


class SpawnSafety(unittest.TestCase):
    def test_engine_code_only_under_main_guard(self):
        for name in ("05_inference.py", "06_tp_benchmark.py"):
            for node in top_level_nodes(name):
                ok = isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)) or \
                    (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)) or \
                    (isinstance(node, ast.If) and "__name__" in ast.unparse(node.test))
                self.assertTrue(ok, f"{name}: module-level statement would re-run in every spawned vLLM child: {ast.unparse(node)[:80]}")
            src = (K / name).read_text()
            self.assertIn('if __name__ == "__main__":', src)

    def test_spawn_reimport_has_no_side_effects(self):
        """Simulate what multiprocessing.spawn does to the parent script: import it as __mp_main__. It must do nothing."""
        for name in ("05_inference.py", "06_tp_benchmark.py"):
            with tempfile.TemporaryDirectory() as d:
                r = subprocess.run([sys.executable, "-c", f"import runpy; runpy.run_path({str(K / name)!r}, run_name='__mp_main__')"],
                                   env=dict(ENV, STOKASTISK_OUT=d), capture_output=True, text=True, cwd=d, timeout=120)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertEqual([p.name for p in pathlib.Path(d).iterdir()], [], "re-import created files (e.g. wiped/created the Triton cache)")

    def test_crash_always_leaves_error_json_with_traceback(self):
        with tempfile.TemporaryDirectory() as d:
            code = ("import _common\n"
                    "def boom(): raise RuntimeError('simulated engine failure')\n"
                    "_common.run_guarded('05_inference_tp9', boom)")
            r = subprocess.run([sys.executable, "-c", code], env=dict(ENV, STOKASTISK_OUT=d), capture_output=True, text=True, cwd=d)
            self.assertEqual(r.returncode, 1)
            j = json.load(open(f"{d}/05_inference_tp9.json"))
            self.assertEqual(j["status"], "ERROR"); self.assertIn("simulated engine failure", j["traceback"]); self.assertIn("Traceback", j["traceback"])


class BackendEvidence(unittest.TestCase):
    def setUp(self):
        import importlib; self.mod = importlib.import_module("05_backend_evidence")

    def test_real_vllm_0_25_1_lines(self):
        a = self.mod.analyse("INFO 11-07 11:27:55 [cuda.py:416] Using AttentionBackendEnum.TRITON_ATTN backend.\n")   # vllm/platforms/cuda.py:416
        self.assertEqual(len(a["triton_lines"]), 1); self.assertEqual(a["other_backend_lines"], [])
        a = self.mod.analyse("Using TRITON_ATTN attention backend out of potential backends: ['TRITON_ATTN'].")      # auto-select form
        self.assertEqual(len(a["triton_lines"]), 1)

    def test_other_backends_cannot_pass_as_triton(self):
        for line in ("Using AttentionBackendEnum.FLASH_ATTN backend.", "Using FLASHINFER attention backend out of potential backends: ['FLASHINFER']."):
            a = self.mod.analyse(line)
            self.assertEqual(a["triton_lines"], [], line); self.assertTrue(a["other_backend_lines"], line)
        # a KV-cache-layout line that merely mentions TRITON_ATTN is not a selection line
        self.assertEqual(self.mod.analyse("Using NHD KV cache layout for TRITON_ATTN backend.")["triton_lines"], [])

    def test_missing_selection_line_is_insufficient_evidence_not_pass(self):
        with tempfile.TemporaryDirectory() as d:
            pathlib.Path(d, "vllm_tp1.log").write_text("engine crashed before backend selection\n")
            r = subprocess.run([sys.executable, str(K / "05_backend_evidence.py"), "1"], env=dict(ENV, STOKASTISK_OUT=d), capture_output=True, text=True, cwd=d)
            self.assertEqual(r.returncode, 1); self.assertIn("INSUFFICIENT", r.stdout)
            self.assertEqual(json.load(open(f"{d}/05_backend_evidence.json"))["status"], "FAIL")

    def test_pass_only_with_real_selection_line(self):
        with tempfile.TemporaryDirectory() as d:
            pathlib.Path(d, "vllm_tp1.log").write_text("Using AttentionBackendEnum.TRITON_ATTN backend.\n")
            r = subprocess.run([sys.executable, str(K / "05_backend_evidence.py"), "1"], env=dict(ENV, STOKASTISK_OUT=d), capture_output=True, text=True, cwd=d)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


class SummaryAndNotebooks(unittest.TestCase):
    def test_summary_is_per_notebook_and_never_passes_without_benchmarks(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, str(K / "07_summary.py")], env=dict(ENV, STOKASTISK_OUT=d), capture_output=True, text=True, cwd=d)
            self.assertEqual(r.returncode, 1); self.assertIn("NOT_RUN", r.stdout)
            for tp in (1, 2): pathlib.Path(d, f"06_bench_tp{tp}.json").write_text(json.dumps({"status": "PASS", "tok_per_s_best": 100.0 * tp}))
            r = subprocess.run([sys.executable, str(K / "07_summary.py")], env=dict(ENV, STOKASTISK_OUT=d), capture_output=True, text=True, cwd=d)
            self.assertEqual(r.returncode, 0, r.stdout); self.assertIn("NOTEBOOK 06 TP BENCHMARK: PASS", r.stdout)


if __name__ == "__main__":
    unittest.main()
