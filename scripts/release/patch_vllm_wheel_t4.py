#!/usr/bin/env python3.13
"""Derive the rc5 vLLM wheel from the rc4 wheel by fixing TWO default-path imports of modules the Flash-free build removed.
usage: patch_vllm_wheel_t4.py SRC.whl DST.whl PATCH_OUT.patch

Found in Kaggle notebook 05 (rc4 run, after the spawn guard worked):
  TP=1: vllm/v1/sample/ops/topk_topp_sampler.py::flashinfer_sampler_supported imports vllm.v1.attention.backends.flashinfer unconditionally
        (upstream assumes flashinfer is installed) -> ModuleNotFoundError in EngineCore load_model.
  TP=2: vllm/distributed/device_communicators/cuda_communicator.py::__init__ imports ...flashinfer_all_reduce unconditionally
        -> ModuleNotFoundError in both Worker processes.
Only pure-Python text is changed; every other member (including ALL native .so files) is asserted byte-identical; RECORD is regenerated."""
import base64, csv, difflib, hashlib, io, json, sys, zipfile

src, dst, patch_out = sys.argv[1:4]
EDITS = {
 "vllm/v1/sample/ops/topk_topp_sampler.py": [(
'''    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    capability = current_platform.get_device_capability()''',
'''    try:
        from vllm.v1.attention.backends.flashinfer import FlashInferBackend
    except ImportError:
        # Stokastisk T4/Triton-only build: the FlashInfer backend module is intentionally not shipped.
        if envs.is_set("VLLM_USE_FLASHINFER_SAMPLER"):
            raise RuntimeError(
                "FlashInfer top-p/top-k sampling was requested but FlashInfer is not part of "
                "this build. Unset VLLM_USE_FLASHINFER_SAMPLER=1."
            ) from None
        logger.info_once(
            "FlashInfer sampling unavailable (not shipped in this build); using the native sampler."
        )
        return False

    capability = current_platform.get_device_capability()''')],
 "vllm/distributed/device_communicators/cuda_communicator.py": [(
'''        from vllm.distributed.device_communicators.flashinfer_all_reduce import (
            FlashInferAllReduce,
        )
        from vllm.distributed.device_communicators.pynccl''',
'''        from vllm.distributed.device_communicators.pynccl'''),
 ('''        if self.use_flashinfer_allreduce and self.world_size > 1:
            self.fi_ar_comm = FlashInferAllReduce(''',
'''        if self.use_flashinfer_allreduce and self.world_size > 1:
            # Stokastisk T4 build: imported only when explicitly requested (module is not shipped).
            from vllm.distributed.device_communicators.flashinfer_all_reduce import (
                FlashInferAllReduce,
            )

            self.fi_ar_comm = FlashInferAllReduce(''')],
}
def b64(d): return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(d).digest()).rstrip(b"=").decode()
zin = zipfile.ZipFile(src); rec, diffs, changed = [], [], []
recname = next(n for n in zin.namelist() if n.endswith(".dist-info/RECORD"))
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zout:
    for i in zin.infolist():
        if i.filename == recname: continue
        data = zin.read(i.filename)
        if i.filename in EDITS:
            old = data.decode()
            for a, b in EDITS[i.filename]:
                assert old.count(a) == 1, f"{i.filename}: anchor not found exactly once (already patched / different source?)\n{a}"
                old = old.replace(a, b)
            diffs += difflib.unified_diff(data.decode().splitlines(True), old.splitlines(True), f"a/{i.filename}", f"b/{i.filename}")
            data = old.encode(); changed.append(i.filename)
        zi = zipfile.ZipInfo(i.filename, i.date_time); zi.external_attr = i.external_attr; zi.compress_type = zipfile.ZIP_DEFLATED
        zout.writestr(zi, data); rec.append((i.filename, b64(data), len(data)))
    assert sorted(changed) == sorted(EDITS), changed
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    for r in rec: w.writerow(r)
    w.writerow((recname, "", "")); zout.writestr(recname, buf.getvalue())
open(patch_out, "w").write("".join(diffs))
# verify: everything except the edited members and RECORD is byte-identical (native .so included)
a, b = zipfile.ZipFile(src), zipfile.ZipFile(dst)
diff = [n for n in a.namelist() if n != recname and a.read(n) != b.read(n)]
assert sorted(diff) == sorted(changed) and set(a.namelist()) == set(b.namelist()), diff
so = [n for n in a.namelist() if n.endswith(".so") or ".so." in n]
print(json.dumps({"changed": changed, "native_files_identical": len(so), "members": len(a.namelist())}, indent=1))
