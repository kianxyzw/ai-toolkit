"""The partition knob must be plumbed all the way through, for all four values.

`model_kwargs.partition` selects one of four checkpoints, and it has to control
two independent things:

  1. WHICH FILE loads          -> _dit_component() / COMFY_FILES
  2. WHETHER REFERENCES APPLY  -> is_ref2va

Only (1) was ever exercised. `is_ref2va` was written as an exact match against
"dit_ref2va" back when the BF16 `ref2va` shards were the only reference
partition we ran, so `ref2va_pruned` — a reference checkpoint — reported False:

    training refs   -> ValueError("reference conditioning needs ... ref2va")
    sample refs     -> silently dropped, preview renders UNCONDITIONED

The silent half is the dangerous one: a pruned-ref2va preview would have looked
like a real ref2va sample while ignoring both reference videos, and the number
it produced would have gone into the Phase-3 table as a ref2va measurement.

Pruning removes the timestep MLP. It does not remove the reference blocks.

Usage:  python testing/test_h3_partition.py
"""

import ast
import os
import re
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "extensions_built_in", "diffusion_models", "minimax_h3",
                   "minimax_h3.py")


def _from_source():
    """The knob is pure string logic over model_kwargs, so it must be testable
    without torch — a pod run is far too expensive a place to learn that a
    partition name does not resolve. Pull the two methods out of the shipped
    source and exec them standalone; they close over no module globals."""
    tree = ast.parse(open(SRC, encoding="utf-8").read())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "MinimaxH3Model")
    wanted = {"_dit_component", "is_ref2va"}
    fns = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {f.name for f in fns} == wanted, f"methods missing from {cls.name}: {fns}"
    ns = {}
    for fn in fns:
        fn.decorator_list = []  # drop @property -> plain callable
        mod = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
        exec(compile(mod, SRC, "exec"), ns)
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].__dict__.get("id") == "COMFY_FILES":
            consts = ast.literal_eval(node.value)
    return ns["_dit_component"], ns["is_ref2va"], consts


_dit_component_fn, _is_ref2va_fn, COMFY_FILES = _from_source()

# partition -> (expected component, carries reference blocks)
PARTITIONS = {
    "fl2va": ("dit_fl2va", False),
    "fl2va_pruned": ("dit_fl2va_pruned", False),
    "ref2va": ("dit_ref2va", True),
    "ref2va_pruned": ("dit_ref2va_pruned", True),
}
DEFAULT_PARTITION = "fl2va_pruned"


class _Probe:
    """The two methods under test, over the only state they read."""

    def __init__(self, partition=None):
        kwargs = {} if partition is None else {"partition": partition}
        self.model_config = SimpleNamespace(model_kwargs=kwargs)

    def _dit_component(self):
        return _dit_component_fn(self)

    @property
    def is_ref2va(self):
        return _is_ref2va_fn(self)


def _cross_check_real_class():
    """On the pod, where torch is present, assert the exec'd copies agree with
    the real class — so the cheap local test cannot drift from what runs."""
    try:
        from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import (
            COMFY_FILES as real_files, MinimaxH3Model,
        )
    except ImportError as e:
        # ONLY a missing dependency may skip. A broad `except Exception` here
        # swallowed an AttributeError from the cross-check's own stub and
        # printed SKIPPED, so this block reported "fine" in the torch-less env
        # (where it never ran) and was never run in the env that has torch —
        # it had never executed successfully anywhere. Same shape as the
        # "benign by default" stage-policy bug: the default must be loud.
        print(f"  [--] real-class cross-check SKIPPED (ImportError: {e}) — "
              "source-level logic above still fully exercised")
        return False
    assert real_files == COMFY_FILES, "COMFY_FILES drifted from the parsed copy"
    for partition, (component, want_refs) in PARTITIONS.items():
        m = SimpleNamespace(model_config=SimpleNamespace(
            model_kwargs={"partition": partition}))
        # is_ref2va calls self._dit_component(); the stub must answer it as a
        # bound method, not merely have the unbound function available.
        m._dit_component = lambda _m=m: MinimaxH3Model._dit_component(_m)
        assert MinimaxH3Model._dit_component(m) == component
        assert MinimaxH3Model.is_ref2va.fget(m) is want_refs
    print("  [ok] real MinimaxH3Model agrees on all 4 partitions")
    return True


def main():
    # (1) every partition selects its own distinct file
    for partition, (component, _) in PARTITIONS.items():
        got = _Probe(partition)._dit_component()
        assert got == component, f"{partition}: component {got}, want {component}"
        assert component in COMFY_FILES, f"{component} missing from COMFY_FILES"
    files = {COMFY_FILES[c] for c, _ in PARTITIONS.values()}
    assert len(files) == len(PARTITIONS), f"partitions share a file: {files}"
    print(f"  [ok] 4 partitions -> 4 distinct checkpoints")

    # (2) reference conditioning follows ref2va-ness, NOT pruned-ness
    for partition, (_, want_refs) in PARTITIONS.items():
        got = _Probe(partition).is_ref2va
        assert got is want_refs, f"{partition}: is_ref2va={got}, want {want_refs}"
        print(f"  [ok] {partition:14} -> is_ref2va {got}")

    # (3) the default stays fl2va_pruned, and stays reference-free
    d = _Probe()
    assert d._dit_component() == f"dit_{DEFAULT_PARTITION}"
    assert d.is_ref2va is False
    print(f"  [ok] default partition {DEFAULT_PARTITION}, is_ref2va False")

    # (4) case-insensitive, and unknown values are rejected loudly rather than
    #     falling through to the default
    assert _Probe("REF2VA_Pruned").is_ref2va is True
    for bad in ("ref2va_prunned", "ref2v", "", "dit_ref2va"):
        try:
            _Probe(bad)._dit_component()
        except ValueError:
            pass
        else:
            raise AssertionError(f"partition {bad!r} was accepted")
    print("  [ok] case-insensitive; 4 malformed values rejected")

    # (5) the historical rule fails exactly the case that cost us the bug —
    #     proof this suite is not a restatement of the code it tests
    old_rule = {p: (c == "dit_ref2va") for p, (c, _) in PARTITIONS.items()}
    wrong = [p for p, (_, want) in PARTITIONS.items() if old_rule[p] is not want]
    assert wrong == ["ref2va_pruned"], f"expected ref2va_pruned to regress, got {wrong}"
    print("  [ok] exact-match rule still fails ref2va_pruned (the bug)")

    # (6) no OTHER site may re-derive reference-ness by string equality; every
    #     gate must route through is_ref2va, or this fix is only half applied
    src = open(SRC, encoding="utf-8").read()
    body = src.split("def is_ref2va", 1)[1].split("\n    def ", 1)[0]
    stray = [
        m for m in re.findall(r'[=!]=\s*"dit_ref2va[^"]*"|[=!]=\s*"ref2va[^"]*"', src)
        if m not in re.findall(r'[=!]=\s*"[^"]*"', body)
    ]
    assert not stray, f"reference-ness re-derived by equality outside is_ref2va: {stray}"
    gates = src.count("self.is_ref2va")
    assert gates >= 4, f"expected >=4 is_ref2va gates, found {gates}"
    print(f"  [ok] all {gates} reference gates route through is_ref2va")

    real = _cross_check_real_class()
    tail = "" if real else " (source-level; run on the pod for the class check)"
    print(f"TEST PASS - partition knob plumbed for all four values{tail}")


if __name__ == "__main__":
    main()
