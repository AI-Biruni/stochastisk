#!/usr/bin/env python3.13
"""Regression test for the rc1/rc2 defect: the shipped stokastisk wheel contained a stale nccl.py that differed from sdk/.
Builds the wheel from the repo and requires every packaged file to be byte-identical to sdk/stokastisk/; also proves the
detector works by checking the known-stale rc2 wheel (when present locally) IS flagged."""
import pathlib, subprocess, sys, tempfile, unittest, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SDK = ROOT / "sdk" / "stokastisk"


def wheel_drift(wheel):
    """Return {relative_name: reason} for files in the wheel that differ from sdk/ (or are missing/extra)."""
    z = zipfile.ZipFile(wheel); drift = {}
    pk = {n[len("stokastisk/"):]: z.read(n) for n in z.namelist() if n.startswith("stokastisk/") and not n.endswith("/")}
    src = {p.relative_to(SDK).as_posix(): p.read_bytes() for p in SDK.rglob("*.py") if "__pycache__" not in p.parts}
    for n, b in src.items():
        if n not in pk: drift[n] = "missing from wheel"
        elif pk[n] != b: drift[n] = f"differs (wheel {len(pk[n])} B, source {len(b)} B)"
    for n in pk:
        if n not in src and n != "_version.py": drift[n] = "in wheel but not in sdk/"
    return drift


class WheelMatchesSource(unittest.TestCase):
    def test_freshly_built_wheel_matches_sdk(self):
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "-o", td, str(ROOT)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            self.assertEqual(wheel_drift(next(pathlib.Path(td).glob("stokastisk-*.whl"))), {})

    def test_detector_flags_known_stale_rc2_wheel(self):
        w = ROOT / "dist-rc/stokastisk-0.3.0-kaggle-t4-sm75-rc2/wheels/stokastisk-0.3.0-py3-none-any.whl"
        if not w.exists(): self.skipTest("rc2 wheel not present locally")
        self.assertIn("nccl.py", wheel_drift(w))


if __name__ == "__main__":
    unittest.main()
