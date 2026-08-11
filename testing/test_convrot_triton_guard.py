"""ConvRot's triton guard must test COMPILABILITY, not importability.

The guard wrapped `import triton` in try/except and, on success, committed to
the fused kernels. Our failure was a compile-time `NameError: libdevice is not
defined` INSIDE a jit kernel: triton imports perfectly, the guard passes, and
the documented "fall back to plain torch ops" path is unreachable in exactly
the situation it was written for. The caller dies instead of degrading.

That single unreachable fallback is what pushed h3-crossview off ConvRot onto
optimum-quanto, which then cost four pod runs of P3 investigation before R3b
showed the detour was not even the cause of the mush.

Root cause of the NameError itself: several kernel builders do
`from triton.language.extra import libdevice` inside the function, so the name
is a local. On triton versions that resolve a jit kernel's free variables
through fn.__globals__ only, it has to exist at module scope - which is why
_import_triton already publishes `triton` and `tl` there. `libdevice` was
missed; it is published now too.

Usage:  python testing/test_convrot_triton_guard.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.util import convrot_quant as cq


def reset():
    cq._triton_ok = None
    os.environ.pop("AITK_CONVROT_FALLBACK", None)


def fake_triton_installed():
    """Make `import triton` succeed regardless of the local environment.

    Without this the import-failure branch short-circuits the guard and every
    assertion below passes for the WRONG reason - the guard would return False
    because triton is missing, not because the probe said so. A test that
    cannot distinguish those is not testing the fix.
    """
    import types

    if "triton" in sys.modules:
        return
    triton = types.ModuleType("triton")
    triton.jit = lambda fn: fn
    lang = types.ModuleType("triton.language")
    lang.constexpr = int
    triton.language = lang
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = lang


def main():
    fake_triton_installed()
    # --- 1. a compile failure must disable the fused path ------------------
    reset()
    real_probe = cq._probe_compile
    cq._probe_compile = lambda: False
    try:
        assert cq._triton_available() is False, (
            "guard still reports available when kernels do not compile - the "
            "fallback is unreachable again")
    finally:
        cq._probe_compile = real_probe
    print("  [ok] compile failure -> guard False (fallback reachable)")

    # --- 2. the result is cached, not re-probed per call -------------------
    calls = []

    def counting_probe():
        calls.append(1)
        return False

    reset()
    cq._probe_compile = counting_probe
    try:
        for _ in range(5):
            cq._triton_available()
        assert len(calls) == 1, f"probed {len(calls)} times; must cache"
    finally:
        cq._probe_compile = real_probe
    print("  [ok] probe runs once and is cached")

    # --- 3. env override forces the fallback without probing at all --------
    reset()
    os.environ["AITK_CONVROT_FALLBACK"] = "1"
    cq._probe_compile = lambda: (_ for _ in ()).throw(
        AssertionError("probe must not run when the override is set"))
    try:
        assert cq._triton_available() is False
    finally:
        cq._probe_compile = real_probe
        reset()
    print("  [ok] AITK_CONVROT_FALLBACK=1 forces the fallback, no probe")

    # --- 4. a passing probe still enables the fused path -------------------
    reset()
    cq._probe_compile = lambda: True
    try:
        assert cq._triton_available() is True, (
            "the fix must not disable ConvRot where it genuinely works")
    finally:
        cq._probe_compile = real_probe
        reset()
    print("  [ok] working kernels still enable the fused path")

    # --- 5. libdevice is published to module globals -----------------------
    # the actual NameError fix; skipped cleanly where triton is absent
    if not hasattr(sys.modules.get("triton"), "__file__"):
        print("  [skip] real triton not installed here; libdevice publication "
              "unverified locally (the pod exercises it)")
    else:
        cq._import_triton()
        assert "libdevice" in vars(cq), (
            "libdevice not published to module globals - the kernels that "
            "reference it as a free variable will NameError at compile")
        print("  [ok] libdevice published to module globals")

    print("TEST PASS - guard tests compilability; fallback is reachable")


if __name__ == "__main__":
    main()
