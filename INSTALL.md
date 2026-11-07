# Install on Kaggle (2 × T4, Python 3.13)

**Option A – Hub CDN, no dataset (recommended):** needs *Internet ON*. `scripts/fetch_release.py` (stdlib only, 4 parallel range requests, SHA256-verified) downloads the ~37 MB release:
```python
import urllib.request, runpy
open('/tmp/fetch_release.py','wb').write(urllib.request.urlopen('https://huggingface.co/waqasm86/stokastisk/resolve/main/scripts/fetch_release.py').read())
bundle = runpy.run_path('/tmp/fetch_release.py')['fetch_release']('stokastisk-0.3.0-kaggle-t4-sm75-rc6')
```

**Option B – Kaggle dataset:** upload `stokastisk-0.3.0-kaggle-t4-sm75-rc6-kaggle-input.zip` (from the Hub `releases/<name>/`) as a dataset and attach it.
Its root must contain the directory `stokastisk-0.3.0-kaggle-t4-sm75-rc6/`. The tooling accepts only that exact release name.

**Manual:**
```bash
python3.13 scripts/fetch_release.py stokastisk-0.3.0-kaggle-t4-sm75-rc6 /tmp/stokastisk-release
python3.13 /tmp/stokastisk-release/stokastisk-0.3.0-kaggle-t4-sm75-rc6/scripts/kaggle/02_install.py /tmp/stokastisk-release/stokastisk-0.3.0-kaggle-t4-sm75-rc6 --dry-run   # plan only
python3.13 .../02_install.py <bundle>                                                                                       # install
```
The installer changes nothing if the plan has a conflict (`02_install_plan.json`), fails if any protected package (torch, triton, CUDA, NCCL, transformers 5.x…) would change,
and never installs FlashAttention/FlashInfer. Kaggle starts every notebook in a fresh session, so each notebook that imports `stokastisk` must run the installer first.
