# Install (Kaggle, 2 x T4, Python 3.13, torch 2.11.0+cu130)
1. Upload the bundle directory `stokastisk-0.3.0/` (or the `.tar.gz`, extracted) as a Kaggle Dataset; attach it to a notebook with **GPU T4 x2**.
2. Run notebooks 01 → 06 from `notebooks/`, or manually:
```bash
python /kaggle/input/<dataset>/stokastisk-0.3.0/scripts/kaggle/02_install.py /kaggle/input/<dataset>/stokastisk-0.3.0
```
The installer verifies `manifests/SHA256SUMS`, installs both wheels `--no-deps`, installs only missing third-party requirements with torch/triton/numpy pinned by a constraint file, then checks imports. It never reinstalls PyTorch.
Offline note: missing third-party requirements are fetched from PyPI if the notebook has Internet; vLLM's pure-Python dependency set is not bundled (the Kaggle package list was not available to this build, so it cannot be pre-filtered).
## Use
```python
import stokastisk
llm = stokastisk.LLM("Qwen/Qwen2.5-0.5B-Instruct", config=stokastisk.StokastiskConfig(tensor_parallel_size=1))
print(llm.complete("The capital of France is", max_tokens=16))
```
