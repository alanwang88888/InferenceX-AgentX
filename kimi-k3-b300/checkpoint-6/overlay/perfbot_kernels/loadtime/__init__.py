"""PERFBOT-LOADTIME: trace of the fastsafetensors weight stream per rank (W1').

Switch: PERFBOT_EXP_LOADTIME=1. Only the value "1" is on. The overlay model.py (KimiK3ForConditionalGeneration.__init__,
tag "# PERFBOT BUILD: LOADTIME") reads the switch and calls install() while the model is built, before the default
loader asks for its weights iterator. With the switch off, this module does not load and nothing prints.

What install() wraps. default_loader.py imports fastsafetensors_weights_iterator by name and calls that module global
when load_format is "fastsafetensors" (vLLM 49cb2bbf). install() replaces the global with a wrapper that calls the
original and passes every (name, tensor) pair through unchanged, in order. Trace only: the served bytes do not change.

Times, host wall clock, per call of the iterator:
- wait_s: time blocked in next() of the original iterator (ParallelLoader construction, file read, H2D copy, NCCL
  scatter: everything the stream does before it hands over a tensor).
- apply_s: time between a yield and the next request (the model's load_weights work on that tensor).
- per-file wait: wait_s split by the safetensors file of each tensor, from model.safetensors.index.json beside the
  files; a name that the index does not have counts under "?".

Line (stderr, flushed), one per iterator call, when the stream ends or is closed:
  PERFBOT-LOADTIME call=<n> rank=<r> total_s=<s> wait_s=<s> apply_s=<s> tensors=<n> bytes=<n> files=<n> wait_by_file=<f>:<s>,...
wait_by_file lists every file by basename, sorted by name. Rank: torch.distributed global rank, else "-".
Any error in the trace prints "PERFBOT-LOADTIME disabled: <error>" once; the stream then passes through untraced.
"""

import importlib
import json
import os
import sys
import time

ENV = "PERFBOT_EXP_LOADTIME"
TAG = "PERFBOT-LOADTIME"
LOADER = "vllm.model_executor.model_loader.default_loader"
NAME = "fastsafetensors_weights_iterator"
INDEX = "model.safetensors.index.json"
_calls = [0]


def say(line: str) -> None:
    print(f"{TAG} {line}", file=sys.stderr, flush=True)


def rank() -> str:
    try:
        import torch.distributed as dist
        return str(dist.get_rank()) if dist.is_initialized() else "-"
    except Exception:
        return "-"


def file_of(files) -> dict:
    """Tensor name -> file basename, from the index beside the first file; {} when there is no index."""
    if not files:
        return {}
    path = os.path.join(os.path.dirname(files[0]), INDEX)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return dict(json.load(f)["weight_map"])


def timed(stream, files, clock=time.perf_counter):
    """Yield every pair of `stream` unchanged; print one PERFBOT-LOADTIME line when it ends or is closed."""
    _calls[0] += 1
    call = _calls[0]
    try:
        where = file_of(files)
    except Exception as e:
        say(f"disabled: {type(e).__name__}: {e}")
        yield from stream
        return
    names = sorted({os.path.basename(f) for f in files})
    by_file = dict.fromkeys(names, 0.0)
    wait = apply = 0.0
    tensors = nbytes = 0
    it = iter(stream)
    start = clock()
    try:
        while True:
            a = clock()
            try:
                pair = next(it)
            except StopIteration:
                wait += clock() - a
                break
            b = clock()
            wait += b - a
            f = where.get(pair[0], "?")
            by_file[f] = by_file.get(f, 0.0) + (b - a)
            tensors += 1
            nbytes += int(getattr(pair[1], "nbytes", 0))
            yield pair
            apply += clock() - b
    finally:
        close = getattr(it, "close", None)
        if close is not None:
            close()
        per = ",".join(f"{k}:{v:.3f}" for k, v in by_file.items())
        say(f"call={call} rank={rank()} total_s={clock() - start:.3f} wait_s={wait:.3f} apply_s={apply:.3f} "
            f"tensors={tensors} bytes={nbytes} files={len(files)} wait_by_file={per}")


def install() -> bool:
    """Wrap default_loader's iterator global once per interpreter; True when the wrapper is in place."""
    try:
        m = importlib.import_module(LOADER)
        original = getattr(m, NAME)
        if getattr(original, "perfbot_loadtime", False):
            return True

        def traced(hf_weights_files, *args, **kwargs):
            return timed(original(hf_weights_files, *args, **kwargs), list(hf_weights_files))

        traced.perfbot_loadtime = True
        traced.__wrapped__ = original
        setattr(m, NAME, traced)
        say(f"on: {LOADER}.{NAME}")
        return True
    except Exception as e:
        say(f"disabled: {type(e).__name__}: {e}")
        return False
