"""Stage 05b: prove from vLLM's OWN log which attention backend each TP run used. usage: 05_backend_evidence.py [TP ...]
Format verified against vLLM 0.25.1 vllm/platforms/cuda.py:
  explicit backend  : logger.info("Using %s backend.", selected_backend)   -> "Using AttentionBackendEnum.TRITON_ATTN backend."
  auto-selected     : "Using TRITON_ATTN attention backend out of potential backends: [...]."
A missing line is reported as INSUFFICIENT evidence (FAIL), never as a pass. Other backends are matched only by name."""
import json, re, sys
from _common import OUT, check, finish, insufficient, warn

TRITON = re.compile(r"Using (?:AttentionBackendEnum\.)?TRITON_ATTN (?:backend\.|attention backend out of potential backends)")
OTHERS = ("FLASH_ATTN", "FLASHINFER", "FLASHMLA", "FLEX_ATTENTION", "XFORMERS", "CUTLASS_MLA", "TRITON_MLA")
OTHER = re.compile(r"Using (?:AttentionBackendEnum\.)?(?:%s)\w* (?:backend\.|attention backend out of potential backends)" % "|".join(OTHERS))
REJECT = re.compile(r"Selected backend .* is not valid")


def analyse(log):
    return {"triton_lines": TRITON.findall(log), "other_backend_lines": OTHER.findall(log),
            "backend_rejected": bool(REJECT.search(log)), "log_bytes": len(log)}


def main():
    tps = [int(a) for a in sys.argv[1:]] or [1, 2]
    R, ev = [], {}
    for tp in tps:
        f = OUT / f"vllm_tp{tp}.log"
        if not f.is_file():
            insufficient(R, f"TP{tp}: vllm log present", str(f)); continue
        ev[tp] = a = analyse(f.read_text(errors="replace"))
        if a["triton_lines"]:
            check(R, f"TP{tp}: vLLM logged TRITON_ATTN selection", True, f"{len(a['triton_lines'])} line(s) (count is not asserted: it depends on vLLM's process topology)")
        else:
            insufficient(R, f"TP{tp}: vLLM logged TRITON_ATTN selection", "no selection line in the log; did the engine start? see vllm_tp%d.log" % tp)
        check(R, f"TP{tp}: no other attention backend selected", not a["other_backend_lines"], a["other_backend_lines"][:3])
        check(R, f"TP{tp}: backend not rejected by vLLM", not a["backend_rejected"])
        fk = re.findall(r"force killing remaining processes count=(\d+)", f.read_text(errors="replace"))
        if fk: warn(R, f"TP{tp}: vLLM had to force-kill {fk} process(es) at shutdown", "vLLM's own log; GPU memory release is recorded in 05_inference_tp%d.json" % tp)
    (OUT / "05_backend_log_evidence.json").write_text(json.dumps(ev, indent=1))
    finish("05_backend_evidence", R, {"evidence": ev})


if __name__ == "__main__":
    main()
