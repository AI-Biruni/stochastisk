#!/usr/bin/env python3.13
"""scripts/fetch_release.py (stdlib HF downloader): verified download, refuses tampered / unsafe / mis-named archives. Offline via file://."""
import hashlib, importlib.util, json, pathlib, tempfile, unittest, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch_release", ROOT / "scripts/fetch_release.py")
fr = importlib.util.module_from_spec(spec); spec.loader.exec_module(fr)
REL = "stokastisk-9.9.9-test"


def make_hub(root, members=None, manifest_name=REL, sidecar_hash=None):
    d = pathlib.Path(root) / "releases" / REL; d.mkdir(parents=True)
    z = d / f"{REL}-kaggle-input.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for n, data in (members or {f"{REL}/manifests/RELEASE-MANIFEST.json": json.dumps({"name": manifest_name}), f"{REL}/wheels/x.whl": "w"}).items():
            zf.writestr(n, data)
    (d / (z.name + ".sha256")).write_text(f"{sidecar_hash or hashlib.sha256(z.read_bytes()).hexdigest()}  {z.name}\n")
    return "file://" + str(root)


class Fetch(unittest.TestCase):
    def test_verified_download_and_extract(self):
        with tempfile.TemporaryDirectory() as hub, tempfile.TemporaryDirectory() as dest:
            out = fr.fetch_release(REL, dest, base=make_hub(hub))
            self.assertEqual(out.name, REL); self.assertTrue((out / "wheels/x.whl").is_file())

    def test_sha256_mismatch_is_rejected_and_nothing_extracted(self):
        with tempfile.TemporaryDirectory() as hub, tempfile.TemporaryDirectory() as dest:
            base = make_hub(hub, sidecar_hash="0" * 64)
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"): fr.fetch_release(REL, dest, base=base)
            self.assertFalse((pathlib.Path(dest) / REL).exists())

    def test_path_traversal_and_foreign_prefix_rejected(self):
        for bad in ("../evil.txt", "other/evil.txt", "/abs.txt"):
            with tempfile.TemporaryDirectory() as hub, tempfile.TemporaryDirectory() as dest:
                base = make_hub(hub, members={bad: "x", f"{REL}/manifests/RELEASE-MANIFEST.json": json.dumps({"name": REL})})
                with self.assertRaisesRegex(RuntimeError, "unsafe or foreign"): fr.fetch_release(REL, dest, base=base)

    def test_manifest_name_must_match_release(self):
        with tempfile.TemporaryDirectory() as hub, tempfile.TemporaryDirectory() as dest:
            with self.assertRaisesRegex(RuntimeError, "manifest name"): fr.fetch_release(REL, dest, base=make_hub(hub, manifest_name="stokastisk-0.3.0-other"))

    def test_notebooks_embed_the_same_downloader(self):
        src = (ROOT / "scripts/fetch_release.py").read_text().split('if __name__ == "__main__":')[0].split('"""', 2)[2].strip()
        for f in (ROOT / "notebooks").glob("*.ipynb"):
            cell = "".join(json.loads(f.read_text())["cells"][2]["source"])
            self.assertIn(src, cell, f.name)


if __name__ == "__main__":
    unittest.main()
