#!/usr/bin/env python3.13
"""Regression tests for the Kaggle notebook 05 rc4 failures, found AFTER the spawn guard worked:
  TP=1: ModuleNotFoundError 'vllm.v1.attention.backends.flashinfer'  (sampler, EngineCore.load_model)
  TP=2: ModuleNotFoundError 'vllm.distributed.device_communicators.flashinfer_all_reduce' (CudaCommunicator, both Workers)
Both are default-path imports of modules the Flash-free build removed. These tests are CPU-only: they prove the packaging/import
structure and the entry-point plumbing, NOT GPU inference."""
import ast, json, os, pathlib, subprocess, sys, tempfile, textwrap, unittest, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
K = ROOT / "scripts" / "kaggle"
RC4_WHEEL = ROOT / "dist-rc/stokastisk-0.3.0-kaggle-t4-sm75-rc4/wheels/vllm-0.25.1-cp38-abi3-linux_x86_64.whl"
NEW_WHEEL = pathlib.Path(os.environ["STOKASTISK_VLLM_WHEEL"]) if os.environ.get("STOKASTISK_VLLM_WHEEL") else None
# imports of dropped modules that exist in the wheel but are NOT on the dense-Qwen2 TP1/TP2 path (MoE, KV connector, MLA, specific models);
# they fail only if the user selects such a model/backend, which this Flash-free build does not support.
OFF_PATH_ALLOWLIST = {"vllm/model_executor/layers/fused_moe/prepare_finalize/nixl_ep.py", "vllm/model_executor/models/openpangu.py",
                      "vllm/models/deepseek_v4/nvidia/flashmla.py", "vllm/v1/attention/backends/mla/flashmla.py",
                      "vllm/v1/attention/backends/mla/flashmla_sparse.py", "vllm/v1/attention/backends/mla/sparse_swa.py",
                      "vllm/models/deepseek_v4/common/ops/fused_indexer_q.py"}


def patched_wheel():
    w = NEW_WHEEL or next(iter(sorted((ROOT / "dist-rc").glob("*rc5/wheels/vllm-*.whl"))), None)
    if w is None or not pathlib.Path(w).is_file():
        # build it from the rc4 wheel into a temp dir (deterministic, pure Python)
        if not RC4_WHEEL.is_file(): return None
        d = tempfile.mkdtemp(prefix="stk-vllmwheel-"); out = pathlib.Path(d) / "vllm.whl"
        r = subprocess.run([sys.executable, str(ROOT / "scripts/release/patch_vllm_wheel_t4.py"), str(RC4_WHEEL), str(out), str(pathlib.Path(d) / "p.patch")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return out
    return pathlib.Path(w)


def parents(tree):
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n): c._p = n


def guarded(node):
    c = node
    while hasattr(c, "_p"):
        c = c._p
        if isinstance(c, ast.Try) and any(h.type is not None and "ImportError" in ast.unparse(h.type) or "Error" in ast.unparse(h.type or ast.Constant("")) for h in c.handlers): return True
        if isinstance(c, ast.If) and ("TYPE_CHECKING" in ast.unparse(c.test) or "flashinfer" in ast.unparse(c.test).lower()): return True   # explicit opt-in branch
    return False


class DefaultPathImports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wheel = patched_wheel()

    def need_wheel(self):
        if self.wheel is None: self.skipTest("rc4 vLLM wheel not available locally (NOT_RUN)")
        return zipfile.ZipFile(self.wheel)

    def test_sampler_import_is_guarded(self):
        src = self.need_wheel().read("vllm/v1/sample/ops/topk_topp_sampler.py").decode(); t = ast.parse(src); parents(t)
        imps = [n for n in ast.walk(t) if isinstance(n, ast.ImportFrom) and n.module == "vllm.v1.attention.backends.flashinfer"]
        self.assertTrue(imps and all(guarded(n) for n in imps))

    def test_sampler_behaviour_without_flashinfer(self):
        src = self.need_wheel().read("vllm/v1/sample/ops/topk_topp_sampler.py").decode()
        fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "flashinfer_sampler_supported")
        code = compile(ast.Module([fn], []), "sampler", "exec")
        class Env:
            VLLM_USE_FLASHINFER_SAMPLER = True
            def __init__(self, set_): self._s = set_
            def is_set(self, k): return self._s
        class Plat:
            def is_cuda(self): return True
            def get_device_capability(self): raise AssertionError("must return before querying capability")
        class Log:
            def info_once(self, *a, **k): pass
            warning_once = info_once
        def run(set_):
            ns = {"envs": Env(set_), "current_platform": Plat(), "logger": Log()}
            exec(code, ns); return ns["flashinfer_sampler_supported"]()
        self.assertIs(run(False), False)                         # unset: silent fall back to the native sampler
        with self.assertRaises(RuntimeError): run(True)          # explicit opt-in cannot be honoured: loud, not silent

    def test_cuda_communicator_imports_flashinfer_allreduce_only_when_requested(self):
        src = self.need_wheel().read("vllm/distributed/device_communicators/cuda_communicator.py").decode(); t = ast.parse(src); parents(t)
        imps = [n for n in ast.walk(t) if isinstance(n, ast.ImportFrom) and n.module == "vllm.distributed.device_communicators.flashinfer_all_reduce"]
        self.assertEqual(len(imps), 1)
        c = imps[0]; conds = []
        while hasattr(c, "_p"):
            c = c._p
            if isinstance(c, ast.If): conds.append(ast.unparse(c.test))
        self.assertTrue(any("use_flashinfer_allreduce" in x for x in conds), conds)

    def test_no_unguarded_import_of_missing_module_on_default_path(self):
        z = self.need_wheel(); names = set(z.namelist())
        have = {n[:-3].replace("/", ".").removesuffix(".__init__") for n in names if n.endswith(".py")}
        have |= {n.split(".")[0].replace("/", ".") for n in names if ".so" in n}
        offenders = {}
        for n in sorted(names):
            if not (n.startswith("vllm/") and n.endswith(".py")): continue
            try: t = ast.parse(z.read(n)); parents(t)
            except SyntaxError: continue
            for node in ast.walk(t):
                if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 and node.module.startswith("vllm.") and not guarded(node):
                    m = node.module
                    if m not in have and not any(h.startswith(m + ".") for h in have) \
                            and not m.startswith(("vllm.third_party", "vllm.vllm_flash_attn")) and m.split(".")[1] not in ("cumem_allocator", "spinloop", "fs_io_C", "_C"):
                        offenders.setdefault(n, set()).add(m)
        unexpected = {k: sorted(v) for k, v in offenders.items() if k not in OFF_PATH_ALLOWLIST}
        self.assertEqual(unexpected, {}, "default-path file imports a module that this wheel does not ship")

    def test_native_binaries_unchanged_from_rc4(self):
        if not RC4_WHEEL.is_file(): self.skipTest("rc4 wheel missing (NOT_RUN)")
        a, b = zipfile.ZipFile(RC4_WHEEL), self.need_wheel()
        so = [n for n in a.namelist() if ".so" in n]
        self.assertTrue(so)
        self.assertEqual([n for n in so if a.read(n) != b.read(n)], [])


class SdkEnv(unittest.TestCase):
    def test_flashinfer_integrations_pinned_off_but_user_wins(self):
        sys.path.insert(0, str(ROOT / "sdk"))
        from stokastisk.config import StokastiskConfig
        for k in ("VLLM_USE_FLASHINFER_SAMPLER", "VLLM_ALLREDUCE_USE_FLASHINFER"): os.environ.pop(k, None)
        StokastiskConfig().apply_environment()
        self.assertEqual(os.environ["VLLM_USE_FLASHINFER_SAMPLER"], "0"); self.assertEqual(os.environ["VLLM_ALLREDUCE_USE_FLASHINFER"], "0")
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1"; StokastiskConfig().apply_environment()
        self.assertEqual(os.environ["VLLM_USE_FLASHINFER_SAMPLER"], "1")
        for k in ("VLLM_USE_FLASHINFER_SAMPLER", "VLLM_ALLREDUCE_USE_FLASHINFER"): os.environ.pop(k, None)


FAKE = textwrap.dedent('''
    import os, pathlib
    class _Cfg:
        def __init__(self, tensor_parallel_size=1, **k): self.tp = tensor_parallel_size
    class StokastiskConfig(_Cfg): pass
    def validate_environment(require_gpus=1): return {}
    def prohibited_report(): return {"clean": True}
    class _O:
        def __init__(s, t, n): s.text, s.token_ids = t, list(range(n))
    class _R:
        def __init__(s, n): s.prompt_token_ids = list(range(1100 if n > 1 else 5)); s.outputs = [_O(" Paris", n)]
    class LLM:
        def __init__(self, model, config=None):
            if os.environ.get("FAKE_FAIL"): raise RuntimeError("deliberate fake engine failure")
            self.tp = config.tp
        def selected_attention_backend(self): return "TRITON_ATTN"
        def generate(self, prompts, sp=None, **k):
            d = pathlib.Path(os.environ["TRITON_CACHE_DIR"]); d.mkdir(parents=True, exist_ok=True); (d / "kernel_unified_attention.json").write_text("{}")
            n = getattr(sp, "max_tokens", 4); return [_R(n) for _ in prompts]
        def close(self): pass
''')
FAKEVLLM = "class SamplingParams:\n    def __init__(self, max_tokens=4, **k): self.max_tokens = max_tokens\n"


class MockedEntryPoints(unittest.TestCase):
    """Runs the REAL 05/06 scripts as the notebook does (subprocess, cwd=scripts dir) against fake stokastisk/vllm modules.
    Proves plumbing (args -> engine -> JSON -> exit code). It does NOT prove GPU inference."""
    def run_script(self, script, tp, fail=False):
        with tempfile.TemporaryDirectory() as d:
            st = pathlib.Path(d, "stubs"); (st / "stokastisk").mkdir(parents=True); (st / "vllm").mkdir()
            (st / "stokastisk/__init__.py").write_text(FAKE); (st / "vllm/__init__.py").write_text(FAKEVLLM)
            env = dict(os.environ, PYTHONPATH=f"{K}:{st}", STOKASTISK_OUT=d, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
            if fail: env["FAKE_FAIL"] = "1"
            r = subprocess.run([sys.executable, str(K / script), str(tp), "fake/model"], cwd=K, env=env, capture_output=True, text=True, timeout=120)
            stem = {"05_inference.py": f"05_inference_tp{tp}", "06_tp_benchmark.py": f"06_bench_tp{tp}"}[script]
            f = pathlib.Path(d, f"{stem}.json")
            return r, (json.loads(f.read_text()) if f.is_file() else None)

    def test_05_both_tp_paths_write_their_own_json(self):
        for tp in (1, 2):
            r, j = self.run_script("05_inference.py", tp)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr); self.assertEqual(j["tp"], tp); self.assertEqual(j["status"], "PASS")
            self.assertIn("loaded_libraries_by_pid", j); self.assertTrue(j["shutdown"]["clean"])

    def test_06_both_tp_paths_write_their_own_json(self):
        for tp in (1, 2):
            r, j = self.run_script("06_tp_benchmark.py", tp)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr); self.assertEqual(j["tp"], tp)

    def test_engine_exception_gives_error_json_nonzero_exit_full_traceback(self):
        for script in ("05_inference.py", "06_tp_benchmark.py"):
            r, j = self.run_script(script, 2, fail=True)
            self.assertNotEqual(r.returncode, 0); self.assertEqual(j["status"], "ERROR")
            self.assertIn("deliberate fake engine failure", j["traceback"]); self.assertIn("Traceback (most recent call last)", j["traceback"])

    def test_failure_of_one_tp_does_not_erase_the_other(self):
        with tempfile.TemporaryDirectory() as d:
            for tp, fail in ((1, False), (2, True)):
                st = pathlib.Path(d, f"s{tp}"); (st / "stokastisk").mkdir(parents=True); (st / "vllm").mkdir()
                (st / "stokastisk/__init__.py").write_text(FAKE); (st / "vllm/__init__.py").write_text(FAKEVLLM)
                env = dict(os.environ, PYTHONPATH=f"{K}:{st}", STOKASTISK_OUT=d, CUDA_VISIBLE_DEVICES="")
                if fail: env["FAKE_FAIL"] = "1"
                subprocess.run([sys.executable, str(K / "05_inference.py"), str(tp), "m"], cwd=K, env=env, capture_output=True, timeout=120)
            self.assertEqual(json.load(open(f"{d}/05_inference_tp1.json"))["status"], "PASS")
            self.assertEqual(json.load(open(f"{d}/05_inference_tp2.json"))["status"], "ERROR")


class ReleaseIdentity(unittest.TestCase):
    def test_generator_release_matches_assembler_and_no_first_hit(self):
        gen = (ROOT / "scripts/make_notebooks.py").read_text()
        import re
        rel = re.search(r'^RELEASE = "([^"]+)"', gen, re.M).group(1)
        self.assertIn(rel, (ROOT / "scripts/release/assemble_rc5.py").read_text())
        for f in (ROOT / "notebooks").glob("*.ipynb"):
            text = f.read_text(); self.assertIn(rel, text, f.name); self.assertNotIn("hits[0]", text, f.name)
            for old in ("-rc2", "-rc3", "-rc4"): self.assertNotIn("sm75" + old, text, f.name)

    def test_no_cuda12_toolchain_or_py311_in_release_scripts(self):
        import re
        pat = re.compile(r"cuda[-_ ]?12\.[89]|cu12[89]|/cuda-12|python3\.11", re.I)
        allow = {"scripts/kaggle/04_nccl_summary.py", "scripts/kaggle/_common.py", "scripts/release/assemble_rc4.py", "scripts/release/assemble_rc5.py"}   # these only DETECT/describe cu12 runtime libs
        hits = []
        for base in ("scripts", "sdk", "notebooks", "requirements"):
            for f in (ROOT / base).rglob("*"):
                if f.is_file() and f.suffix in (".py", ".sh", ".txt", ".ipynb", ".toml") and "__pycache__" not in f.parts:
                    rel = f.relative_to(ROOT).as_posix()
                    if rel in allow: continue
                    for m in pat.finditer(f.read_text(errors="ignore")): hits.append((rel, m.group(0)))
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
