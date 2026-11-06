"""CPU-only tests. Run: PYTHONPATH=sdk python3.13 -m pytest tests (or python3.13 tests/test_sdk_cpu.py)."""
import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk"))

import stokastisk
from stokastisk import StokastiskConfig, errors


class SDKTests(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(stokastisk.__version__, "0.3.0")
        self.assertEqual(stokastisk.VLLM_BASE_VERSION, "0.25.1")
        self.assertEqual(stokastisk.CUDA_BUILD_TARGET, "13.0")
        self.assertEqual(stokastisk.GPU_COMPUTE_CAPABILITY, (7, 5))

    def test_config_defaults(self):
        c = StokastiskConfig()
        kw = c.engine_kwargs()
        self.assertEqual(kw["tensor_parallel_size"], 1)
        self.assertEqual(kw["dtype"], "half")
        self.assertTrue(kw["disable_custom_all_reduce"])
        self.assertEqual(kw["attention_config"], {"backend": "TRITON_ATTN"})

    def test_nccl_env_does_not_override(self):
        os.environ["NCCL_P2P_DISABLE"] = "0"
        os.environ.pop("NCCL_IB_DISABLE", None)
        applied = StokastiskConfig().apply_environment()
        self.assertNotIn("NCCL_P2P_DISABLE", applied)
        self.assertEqual(os.environ["NCCL_P2P_DISABLE"], "0")
        self.assertEqual(os.environ["NCCL_IB_DISABLE"], "1")

    def test_env_override(self):
        os.environ["STOKASTISK_TENSOR_PARALLEL_SIZE"] = "1"
        self.assertEqual(StokastiskConfig.from_env().tensor_parallel_size, 1)
        del os.environ["STOKASTISK_TENSOR_PARALLEL_SIZE"]

    def test_validate_rejects_without_gpu(self):
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2 and \
                torch.cuda.get_device_capability(0) == (7, 5):
            self.skipTest("running on target hardware")
        with self.assertRaises(errors.StokastiskError):
            stokastisk.validate_environment()

    def test_report_never_raises(self):
        r = stokastisk.report()
        self.assertEqual(r["sdk_version"], "0.3.0")
        self.assertIn("native_extensions", r)

    def test_error_hierarchy(self):
        self.assertTrue(issubclass(errors.GPUDiscoveryError, errors.EnvironmentValidationError))
        self.assertTrue(issubclass(errors.EngineError, errors.StokastiskError))

    def test_flash_backends_rejected(self):
        for b in ("FLASH_ATTN", "flash_attn", "FLASHINFER", "FA3"):
            with self.assertRaises(errors.ConfigurationError):
                StokastiskConfig(attention_backend=b)

    def test_unknown_backend_rejected(self):
        with self.assertRaises(errors.ConfigurationError):
            StokastiskConfig(attention_backend="NOPE")

    def test_backend_case_normalised(self):
        self.assertEqual(StokastiskConfig(attention_backend="triton_attn").attention_backend, "TRITON_ATTN")

    def test_invalid_tp_and_dtype(self):
        for kw in ({"tensor_parallel_size": 3}, {"dtype": "bfloat16"}, {"gpu_memory_utilization": 1.5}):
            with self.assertRaises(errors.ConfigurationError):
                StokastiskConfig(**kw)

    def test_env_bool_and_bad_values(self):
        os.environ["STOKASTISK_ENFORCE_EAGER"] = "false"
        try:
            self.assertFalse(StokastiskConfig.from_env().enforce_eager)
            os.environ["STOKASTISK_TENSOR_PARALLEL_SIZE"] = "abc"
            with self.assertRaises(errors.ConfigurationError):
                StokastiskConfig.from_env()
        finally:
            os.environ.pop("STOKASTISK_ENFORCE_EAGER", None); os.environ.pop("STOKASTISK_TENSOR_PARALLEL_SIZE", None)

    def test_engine_kwargs_complete(self):
        kw = StokastiskConfig(tensor_parallel_size=2, max_model_len=None).engine_kwargs()
        self.assertNotIn("max_model_len", kw)
        self.assertEqual(kw["tensor_parallel_size"], 2)
        self.assertEqual(kw["attention_config"], {"backend": "TRITON_ATTN"})
        self.assertTrue(kw["enforce_eager"])

    def test_nccl_env_values(self):
        from stokastisk.config import NCCL_ENV_NO_P2P
        self.assertEqual(NCCL_ENV_NO_P2P["NCCL_P2P_DISABLE"], "1")

    def test_no_flash_native_loaded_and_no_flash_import_in_sdk(self):
        self.assertEqual(stokastisk.report()["flash_attn_native_loaded"], [])
        import pathlib
        for f in (pathlib.Path(stokastisk.__file__).parent).glob("*.py"):
            self.assertNotIn("import flash_attn", f.read_text())

    def test_attention_backend_verification_detects_mismatch(self):
        from types import SimpleNamespace as NS
        from stokastisk.engine import LLM
        llm = object.__new__(LLM); llm.config = StokastiskConfig()
        llm._engine = NS(llm_engine=NS(vllm_config=NS(attention_config=NS(backend=NS(name="FLASH_ATTN")))))
        with self.assertRaises(errors.AttentionBackendError):
            llm.verify_attention_backend()
        llm._engine.llm_engine.vllm_config.attention_config.backend = NS(name="TRITON_ATTN")
        self.assertEqual(llm.verify_attention_backend(), "TRITON_ATTN")   # mocked: logic test only, not GPU evidence

    def test_missing_vllm_reported_as_engine_error(self):
        from stokastisk.engine import LLM
        import importlib.util
        if importlib.util.find_spec("vllm"):
            self.skipTest("vllm importable here")
        with self.assertRaises(errors.EngineError):
            LLM("x")


if __name__ == "__main__":
    unittest.main()
