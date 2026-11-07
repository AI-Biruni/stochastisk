# Install on Kaggle (2 × T4, Python 3.13)

**Option A – no dataset (recommended):** open any of `notebooks/01…06`; the first cells download and verify the release from Hugging Face
(`scripts/fetch_release.py`, stdlib only). Needs *Internet ON*.

**Option B – Kaggle dataset:** upload `stokastisk-0.3.0-kaggle-t4-sm75-rc6-kaggle-input.zip` (from the Hub `releases/<name>/`) as a dataset and attach it.
Its root must contain the directory `stokastisk-0.3.0-kaggle-t4-sm75-rc6/`. Notebooks accept only that exact release name.

**Manual:**
```bash
python3.13 scripts/fetch_release.py stokastisk-0.3.0-kaggle-t4-sm75-rc6 /tmp/stokastisk-release
python3.13 /tmp/stokastisk-release/stokastisk-0.3.0-kaggle-t4-sm75-rc6/scripts/kaggle/02_install.py /tmp/stokastisk-release/stokastisk-0.3.0-kaggle-t4-sm75-rc6 --dry-run   # plan only
python3.13 .../02_install.py <bundle>                                                                                       # install
```
The installer changes nothing if the plan has a conflict (`02_install_plan.json`), fails if any protected package (torch, triton, CUDA, NCCL, transformers 5.x…) would change,
and never installs FlashAttention/FlashInfer. Run notebooks in a fresh session each: 01 → 06.
