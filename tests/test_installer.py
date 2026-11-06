"""Installer decision-logic tests. Static/local: synthetic package snapshots, no pip, no network, no GPU."""
import importlib.util, os, pathlib, sys, tempfile, unittest
HERE = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("STOKASTISK_OUT", tempfile.mkdtemp())
sys.path.insert(0, str(HERE.parent / "scripts" / "kaggle"))
spec = importlib.util.spec_from_file_location("inst", HERE.parent / "scripts/kaggle/02_install.py")
inst = importlib.util.module_from_spec(spec); spec.loader.exec_module(inst)
from packaging.requirements import Requirement

# Kaggle baseline values observed in executed notebooks 01/02 (versions of preinstalled packages are from the pip log)
KAGGLE = {"torch": "2.11.0+cu130", "triton": "3.6.0", "numpy": "2.1.3", "transformers": "5.20.2", "tokenizers": "0.23.3",
          "huggingface-hub": "1.33.0", "fastapi": "0.141.1", "nvidia-cuda-nvcc": "13.0.88", "nvidia-nvvm": "13.0.88",
          "opentelemetry-api": "1.42.1", "opentelemetry-sdk": "1.42.1", "lark": "1.3.1", "numba": "0.61.2", "safetensors": "0.7.0"}
PINS = inst.parse_pins((HERE.parent / "requirements/kaggle-t4-core.txt").read_text())
REQS = [Requirement(r) for r in ("transformers>=5.5.3", "tokenizers>=0.21.1", "numpy", "fastapi>=0.133.0", "xgrammar<1.0.0,>=0.2.1", "msgspec")]


class InstallerTests(unittest.TestCase):
    def test_baseline_ok(self):
        self.assertEqual(inst.check_baseline(KAGGLE), [])

    def test_rejects_transformers_4(self):          # the exact failure seen on Kaggle
        b = dict(KAGGLE, transformers="4.57.6")
        self.assertTrue(any("transformers" in e for e in inst.check_baseline(b)))

    def test_rejects_transformers_below_floor(self):
        self.assertTrue(inst.check_baseline(dict(KAGGLE, transformers="5.5.2")))

    def test_rejects_changed_torch_triton(self):
        self.assertTrue(inst.check_baseline(dict(KAGGLE, torch="2.10.0+cu128")))
        self.assertTrue(inst.check_baseline(dict(KAGGLE, triton="3.5.0")))

    def test_rejects_preinstalled_flashinfer(self):
        self.assertTrue(any("prohibited" in e for e in inst.check_baseline(dict(KAGGLE, **{"flashinfer-python": "0.6.13"}))))

    def test_plan_installs_only_absent_pins_and_never_touches_present(self):
        p = inst.make_plan(KAGGLE, PINS, REQS)
        self.assertTrue(set(p["install"]) <= set(PINS))
        self.assertNotIn("transformers", p["install"]); self.assertNotIn("fastapi", p["install"])
        self.assertNotIn("flashinfer-python", p["install"]); self.assertNotIn("nvidia-cuda-nvcc", p["install"])
        self.assertEqual(p["conflicts"], [])

    def test_plan_is_empty_when_everything_present_idempotent(self):
        have = dict(KAGGLE, **PINS)
        p = inst.make_plan(have, PINS, REQS)
        self.assertEqual(p["install"], {}); self.assertEqual(p["conflicts"], [])

    def test_plan_conflict_when_requirement_unsatisfiable_without_touching_env(self):
        p = inst.make_plan(dict(KAGGLE, fastapi="0.100.0"), PINS, REQS)
        self.assertTrue(any("fastapi" in c for c in p["conflicts"]))

    def test_plan_flags_requirement_missing_and_not_pinned(self):
        p = inst.make_plan(KAGGLE, PINS, REQS + [Requirement("some-missing-lib>=1")])
        self.assertTrue(any("some-missing-lib" in c for c in p["conflicts"]))

    def test_plan_flags_prohibited_requirement(self):
        p = inst.make_plan(KAGGLE, PINS, REQS + [Requirement("flashinfer-python==0.6.13")])
        self.assertTrue(any("prohibited" in c for c in p["conflicts"]))

    def test_core_pins_contain_no_prohibited(self):
        self.assertFalse(set(PINS) & inst.PROHIBITED)

    def test_diff_detects_the_kaggle_failure(self):    # reproduces notebook-02 effects
        after = dict(KAGGLE, transformers="4.57.6", **{"nvidia-cuda-nvcc": "13.2.86", "nvidia-nvvm": "13.2.86", "flashinfer-python": "0.6.13"})
        d = inst.diff_snapshots(KAGGLE, after, ["vllm", "stokastisk"])
        self.assertFalse(d["ok"]); self.assertIn("transformers", d["preinstalled_changed_or_removed"])
        self.assertIn("nvidia-nvvm", d["preinstalled_changed_or_removed"]); self.assertIn("flashinfer-python", d["unexpected_added"])

    def test_diff_ok_for_expected_install(self):
        after = dict(KAGGLE, vllm="0.25.1", stokastisk="0.3.0", xgrammar="0.2.5.post1")
        self.assertTrue(inst.diff_snapshots(KAGGLE, after, ["vllm", "stokastisk", "xgrammar"])["ok"])

    def test_diff_detects_removed(self):
        a = dict(KAGGLE); a.pop("numba")
        self.assertFalse(inst.diff_snapshots(KAGGLE, a, [])["ok"])

    def test_wheel_inspection_flags_flash_files(self):
        import zipfile
        t = pathlib.Path(tempfile.mkdtemp()) / "x.whl"
        with zipfile.ZipFile(t, "w") as z:
            z.writestr("vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so", b"x"); z.writestr("vllm-0.25.1.dist-info/METADATA", "Name: vllm\n")
        self.assertTrue(inst.inspect_wheel(t)["prohibited_binaries_or_dirs"])

    def test_real_candidate_wheel_is_clean_and_plans_without_conflict(self):
        w = HERE.parent / "dist-rc/wheels/vllm-0.25.1-cp38-abi3-linux_x86_64.whl"
        if not w.exists(): self.skipTest("candidate wheel not built")
        i = inst.inspect_wheel(w); self.assertEqual(i["prohibited_binaries_or_dirs"], [])
        reqs = inst.vllm_core_requirements(i["metadata"])
        names = {inst.norm(r.name) for r in reqs}
        self.assertFalse(names & inst.PROHIBITED); self.assertIn("transformers", names)
        have = dict(KAGGLE, pydantic="2.12.5", aiohttp="3.13.3", starlette="1.0.1", openai="2.0.0", requests="2.32.0", tqdm="4", regex="1",
                    cachetools="1", psutil="1", sentencepiece="1", pillow="1", tiktoken="0.7", typing_extensions="4.12", filelock="3.20", pyzmq="26",
                    pyyaml="6", einops="0.8", cloudpickle="3", uvloop="0.21", prometheus_client="0.21", **{"py-cpuinfo": "9"})
        have = {inst.norm(k): v for k, v in have.items()}
        p = inst.make_plan(have, PINS, reqs)
        self.assertEqual(p["conflicts"], [], p["conflicts"]); self.assertEqual(set(p["install"]), set(PINS))


if __name__ == "__main__":
    unittest.main(verbosity=1)
