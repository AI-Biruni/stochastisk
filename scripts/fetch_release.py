#!/usr/bin/env python3.13
"""Download a Stokastisk release bundle from Hugging Face (CDN-backed `resolve/` URLs) - standard library only.

    python3.13 fetch_release.py stokastisk-0.3.0-kaggle-t4-sm75-rc6 [DEST]      # prints the bundle directory

 * ONE file (the release's `<release>-kaggle-input.zip`, ~37 MB) over up to 4 parallel HTTP range requests (the Kaggle CPU budget); falls back
   to a single stream when the server does not support ranges, and to `file://` for offline tests;
 * SHA256 verified against the `.sha256` sidecar BEFORE extraction; path-traversal-safe extraction; the extracted directory must carry
   the release name and `manifests/RELEASE-MANIFEST.json` must say the same name;
 * nothing is installed here: run `<bundle>/scripts/kaggle/02_install.py <bundle>` afterwards (it verifies every file in SHA256SUMS first).
Default destination is /tmp (NOT /kaggle/working, which has a 20 GiB quota)."""
import hashlib, json, os, pathlib, shutil, sys, time, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor

HF_REPO = "waqasm86/stokastisk"
HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve"


def _open(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {"User-Agent": "stokastisk-fetch/1"}), timeout=60)


def _range_ok(url):
    if not url.startswith("http"):
        return None
    with _open(url, {"Range": "bytes=0-0", "User-Agent": "stokastisk-fetch/1"}) as r:
        cr = r.headers.get("Content-Range")                      # "bytes 0-0/37080766"
        return int(cr.rsplit("/", 1)[1]) if r.status == 206 and cr else None


def _download(url, dest, workers=4, retries=3):
    dest = pathlib.Path(dest); total = _range_ok(url)
    t0 = time.time()
    if total and total > (4 << 20):
        chunk = -(-total // workers); parts = [(i * chunk, min(total, (i + 1) * chunk) - 1) for i in range(workers) if i * chunk < total]
        with open(dest, "wb") as f:
            f.truncate(total)

        def part(rng):
            a, b = rng
            for attempt in range(retries):
                try:
                    with _open(url, {"Range": f"bytes={a}-{b}", "User-Agent": "stokastisk-fetch/1"}) as r, open(dest, "r+b") as f:
                        f.seek(a); shutil.copyfileobj(r, f, 1 << 20)
                    return
                except Exception:
                    if attempt == retries - 1: raise
                    time.sleep(2 ** attempt)
        with ThreadPoolExecutor(len(parts)) as ex:
            list(ex.map(part, parts))
        mode = f"{len(parts)} parallel ranges"
    else:
        for attempt in range(retries):
            try:
                with _open(url) as r, open(dest, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
                break
            except Exception:
                if attempt == retries - 1: raise
                time.sleep(2 ** attempt)
        mode = "single stream"
    n = dest.stat().st_size
    print(f"downloaded {n / 1e6:.1f} MB in {time.time() - t0:.1f}s ({n / 1e6 / max(time.time() - t0, 1e-6):.1f} MB/s, {mode})")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def fetch_release(release, dest="/tmp/stokastisk-release", base=None, revision="main", workers=4):
    """Return the verified, extracted bundle directory (…/<release>). `base` overrides the Hub URL prefix (tests use file://)."""
    base = (base or HF_BASE).rstrip("/")
    root = f"{base}/{revision}/releases/{release}" if base.startswith("http") else f"{base}/releases/{release}"
    dest = pathlib.Path(dest); dest.mkdir(parents=True, exist_ok=True)
    zname = f"{release}-kaggle-input.zip"; zpath = dest / zname
    if zpath.exists(): zpath.unlink()
    with _open(f"{root}/{zname}.sha256") as r:
        expected = r.read().decode().split()[0]
    _download(f"{root}/{zname}", zpath, workers=workers)
    got = _sha256(zpath)
    if got != expected:
        zpath.unlink(); raise RuntimeError(f"SHA256 mismatch for {zname}: expected {expected}, got {got}")
    out = dest / release
    if out.exists(): shutil.rmtree(out)
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None
        for n in z.namelist():
            if n.startswith("/") or ".." in pathlib.PurePosixPath(n).parts or not n.startswith(release + "/"):
                raise RuntimeError(f"unsafe or foreign path in archive: {n}")
        z.extractall(dest)
    m = json.loads((out / "manifests" / "RELEASE-MANIFEST.json").read_text())
    if m.get("name") != release:
        raise RuntimeError(f"manifest name {m.get('name')!r} != {release!r}")
    print(f"release {release}: SHA256 {got[:16]}... verified, extracted to {out}")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    print(fetch_release(sys.argv[1], *(sys.argv[2:3] or ["/tmp/stokastisk-release"])))
