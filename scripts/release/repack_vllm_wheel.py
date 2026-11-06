#!/usr/bin/env python3.13
"""Derive the Stokastisk vLLM wheel: remove FlashAttention native binaries, regenerate RECORD.
usage: repack_vllm_wheel.py <src.whl> <dst.whl>. Nothing else is modified."""
import base64, csv, hashlib, io, json, sys, zipfile
src, dst = sys.argv[1:3]
STRIP = ("vllm/vllm_flash_attn/_vllm_fa", "vllm/vllm_flash_attn/_vllm_fa4")
def b64(d): return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(d).digest()).rstrip(b"=").decode()
zin = zipfile.ZipFile(src); removed, rec = [], []
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zout:
    recname = [n for n in zin.namelist() if n.endswith(".dist-info/RECORD")][0]
    for i in zin.infolist():
        if i.filename == recname: continue
        if i.filename.startswith(STRIP) and i.filename.endswith(".so"):
            removed.append(i.filename); continue
        data = zin.read(i.filename)
        zi = zipfile.ZipInfo(i.filename, i.date_time); zi.external_attr = i.external_attr; zi.compress_type = zipfile.ZIP_DEFLATED
        zout.writestr(zi, data); rec.append((i.filename, b64(data), len(data)))
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    for r in rec: w.writerow(r)
    w.writerow((recname, "", ""))
    zout.writestr(recname, buf.getvalue())
print(json.dumps({"removed": removed, "kept_entries": len(rec)}, indent=1))
