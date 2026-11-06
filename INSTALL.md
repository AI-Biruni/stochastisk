# Install on Kaggle (2 × T4, Python 3.13)
1. Upload `stokastisk-0.3.0-kaggle-t4-sm75.zip` as a Kaggle Dataset (Kaggle extracts it; the root contains `manifests/SHA256SUMS` and `scripts/kaggle/`).
2. New notebook → accelerator **GPU T4 ×2** → *Add input* → your dataset. Internet ON for the 57 MB pin download and the Hugging Face model download.
3. Run `notebooks/01` … `06` in order, or by hand:
```bash
python <dataset>/scripts/kaggle/02_install.py <dataset> --dry-run    # plan only
python <dataset>/scripts/kaggle/02_install.py <dataset>              # install; exit code != 0 on any failure
```
```python
import stokastisk
llm = stokastisk.LLM("Qwen/Qwen2.5-0.5B-Instruct", config=stokastisk.StokastiskConfig(tensor_parallel_size=1))
```
If the install plan reports a conflict nothing has been changed; send me `02_install_plan.json`.
