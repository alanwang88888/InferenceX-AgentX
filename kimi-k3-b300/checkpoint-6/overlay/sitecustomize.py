"""Overlay import hook: make `overlay/vllm/<module path>.py` replace the image's module (docs/overlay-design.md).

serve.sh puts `$W/overlay` first on PYTHONPATH. That alone does not shadow a submodule of `vllm`: the image's
`vllm` is a regular package (it has `__init__.py`), so `vllm.__path__` is the image directory only and
`overlay/vllm/...` is never searched. Python imports `sitecustomize` from sys.path at startup in every
interpreter the server starts (API server, engine core, spawned workers), so this file installs one
meta-path finder that maps `vllm.<a>.<b>` to `overlay/vllm/<a>/<b>.py` when that file exists: whole-module
replacement, file by file, no runtime patching. Then it runs the image's own sitecustomize (the Ubuntu
apport hook), which this file shadows.

Image key (ROUND-UP G1): every overlay/vllm module is a copy of the image's at BASE (contract.json image.vllm_commit),
so the finder installs only when the running vLLM's version names that commit (`+g<BASE>`; the 49cb2bbf image ships
vllm 0.32.0rc1.dev281+g49cb2bbf7). overlay/vllm_base.json pins each copy's sha256 at BASE; an edit is listed there
(driver/tests/test_overlay_base.py). On any other vLLM the copies would shadow other modules: the overlay stays off
and says so once on stderr (`PERFBOT-OVERLAY off`). No version found: off, silent.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_OVERLAY = os.path.dirname(os.path.abspath(__file__))
_PREFIX = "vllm."
BASE = "49cb2bbf"
OFF_ENV = "PERFBOT_OVERLAY_OFF"


def vllm_version():
    try:
        from importlib.metadata import version
        return version("vllm")
    except Exception:
        return None


def base_matches(v) -> bool:
    return v is not None and f"+g{BASE}" in v


class PerfbotOverlayFinder(importlib.abc.MetaPathFinder):
    """`vllm.x.y` -> `<overlay>/vllm/x/y.py` when that file exists; every other import is untouched."""

    overlay_root = _OVERLAY

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_PREFIX):
            return None
        candidate = os.path.join(self.overlay_root, *fullname.split(".")) + ".py"
        if not os.path.isfile(candidate):
            return None
        return importlib.util.spec_from_file_location(fullname, candidate)


def install(version, environ) -> str:
    """Install the finder when `version` is the base and the arm did not switch the overlay off; return the state
    line (also printed once on stderr, so server.log shows which overlay every interpreter ran), or "" when silent.
    PERFBOT_OVERLAY_OFF=1 is the control arm of a same-tree pair (R-UP-OVL ovl-ctl): the image's own modules."""
    if environ.get(OFF_ENV) == "1":
        line = f"PERFBOT-OVERLAY off: {OFF_ENV}=1"
    elif not base_matches(version):
        line = "" if version is None else f"PERFBOT-OVERLAY off: vllm {version} is not the overlay base +g{BASE}"
    else:
        if not any(getattr(f, "overlay_root", None) == _OVERLAY for f in sys.meta_path):
            sys.meta_path.insert(0, PerfbotOverlayFinder())
        line = f"PERFBOT-OVERLAY on: vllm {version} base +g{BASE}"
    if line:
        print(line, file=sys.stderr, flush=True)
    return line


install(vllm_version(), os.environ)


SWITCH_PREFIX = "PERFBOT_EXP_"
_STDLIB = {os.__file__, os.path.join(os.path.dirname(os.__file__), "_collections_abc.py"), "<frozen os>",
           "<frozen _collections_abc>"}
_said = set()


def mark(name: str, value: str, frame=None) -> None:
    """Switch liveness, one format for every switch: `PERFBOT-SWITCH <name>=<item> <file>:<line>` on stderr, once per
    item and call site per interpreter, for each comma item of an on value (not "" or "0"). <file>:<line> is the
    reading module (relative to the overlay), so server.log proves which module read which item (the ingest's
    LIVENESS MISSING check). A module that expands an item into others (an alias) calls mark() for each one it turns on.
    Only a module under the overlay (every bundle module is) marks: a whole-environment copy (os.environ.copy, dict(),
    subprocess, platform) reads every key and proves nothing (t2v3 2026-10-20: CUTE_CACHE marked at 12 such sites)."""
    if value in ("", "0"):
        return
    f = frame or sys._getframe(1)
    while f.f_back is not None and f.f_code.co_filename in _STDLIB and f.f_code.co_name in ("get", "getenv"):
        f = f.f_back  # .get / os.getenv: the reading module; a copy() frame stays and is not the overlay's
    if not f.f_code.co_filename.startswith(_OVERLAY + os.sep):
        return
    where = os.path.relpath(f.f_code.co_filename, _OVERLAY)
    for item in (x.strip() for x in value.split(",")):
        if item and (name, item, where, f.f_lineno) not in _said:
            _said.add((name, item, where, f.f_lineno))
            print(f"PERFBOT-SWITCH {name}={item} {where}:{f.f_lineno}", file=sys.stderr, flush=True)


_SUBSCR = __import__("dis").opmap["BINARY_SUBSCR"]


class _SwitchMarkedEnviron(type(os.environ)):
    """os.environ whose reads of a PERFBOT_EXP_* switch call mark(); values unchanged. A read straight from a module is
    marked only at a subscript (`os.environ[k]`): dict(os.environ) also calls this from the module's frame, at a CALL."""

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(key, str) and key.startswith(SWITCH_PREFIX):
            f = sys._getframe(1)
            if f.f_code.co_filename in _STDLIB or f.f_code.co_code[f.f_lasti] == _SUBSCR:
                mark(key, value, f)
        return value


os.environ.__class__ = _SwitchMarkedEnviron


def _run_image_sitecustomize() -> None:
    others = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _OVERLAY]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", others)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


try:
    _run_image_sitecustomize()
except Exception:  # the image's hook is best-effort there too (apport); never block interpreter start
    pass
