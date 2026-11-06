#!/usr/bin/env python3.13
"""Regression tests for the Kaggle notebook 03 failure: Kaggle runs each notebook in a FRESH session, so notebooks that
import stokastisk/vllm must install first. Also pins the native-operator call forms checked against vLLM 0.25.1."""
import ast, json, pathlib, re, unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NEEDS_INSTALL = ("03_stokastisk_single_gpu_validation", "04_stokastisk_dual_gpu_nccl_validation",
                 "05_stokastisk_vllm_inference", "06_stokastisk_tensor_parallel_benchmark")


def cells(name):
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


class NotebookTests(unittest.TestCase):
    def test_notebooks_03_to_06_install_and_print_provenance_before_use(self):
        for n in NEEDS_INSTALL:
            src = cells(n)
            i_inst = next(i for i, s in enumerate(src) if "02_install.py" in s and "assert r.returncode == 0" in s)
            i_prov = next(i for i, s in enumerate(src) if "SHA256SUMS" in s and "sha(" in s and "assert" in s)
            i_use = next(i for i, s in enumerate(src) if i > i_prov and re.search(r"03_single_gpu|04_nccl|05_inference|06_tp_benchmark", s))
            self.assertLess(i_inst, i_prov, n); self.assertLess(i_prov, i_use, n)

    def test_all_code_cells_parse(self):
        for f in (ROOT / "notebooks").glob("*.ipynb"):
            for c in cells(f.stem):
                ast.parse("\n".join(l for l in c.splitlines() if not l.startswith("!")))

    def test_03_uses_registered_native_op_forms(self):
        s = (ROOT / "scripts/kaggle/03_single_gpu.py").read_text()
        self.assertIn("torch.ops._C.silu_and_mul(outs, xs)", s)      # vllm._custom_ops has no silu_and_mul wrapper in 0.25.1
        self.assertIn("torch.ops._C.rms_norm(out, xi, w, 1e-6)", s)
        self.assertNotIn("ops.silu_and_mul(", s.replace("torch.ops._C.silu_and_mul(", ""))
        self.assertIn("require_installed", s)

    def test_notebooks_are_regenerated_from_generator(self):
        self.assertIn("INSTALL", (ROOT / "scripts/make_notebooks.py").read_text())


if __name__ == "__main__":
    unittest.main()
