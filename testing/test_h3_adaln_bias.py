"""adaln_bias must match what load_state_dict actually sees, across ALL
checkpoint families. Regression suite for two pod-costing failures.

Three families disagree, and every simpler rule gets one wrong:

  pruned int8       t-table, adaln linears stay fp16    bias PRESENT
  non-pruned int8   no t-table, adaln linears ConvRot   bias ABSENT
  BF16 shards       no t-table, nothing quantized       bias PRESENT

History, both paid for on pods:
  attempt 3  upstream rule (bias = pruned) built the BF16 model WITHOUT bias
             -> "unexpected [blocks.N.adaln_proj.linear.bias, ...]"
  item 6     raw-key rule built the non-pruned int8 model WITH bias, because
             the key IS in the raw dict but import_comfy_quantized_layers
             consumes it -> "missing [blocks.N.adaln_proj.linear.bias, ...]"

The rule under test asks the question load_state_dict will ask: the bias
survives only if the key exists AND its module is not ConvRot-quantized.

Usage:  python testing/test_h3_adaln_bias.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import (
    _adaln_block_bias,
)
from extensions_built_in.diffusion_models.minimax_h3.src.transformer import (
    MiniMaxH3AdalnProj,
    MiniMaxH3TransformerParams,
)

# verbatim from the attempt-3 pod traceback (formatter truncates at 8)
ATTEMPT3_UNEXPECTED = [f"blocks.{i}.adaln_proj.linear.bias" for i in range(8)]
# verbatim from the item-6 pod traceback - same names, opposite direction
ITEM6_MISSING = list(ATTEMPT3_UNEXPECTED)


def family_pruned_int8():
    """t-table present; adaln linears are fp16, so NOT ConvRot-marked."""
    sd = {"adaln_t_table": None}
    for i in range(4):
        sd[f"blocks.{i}.adaln_proj.linear.weight"] = None
        sd[f"blocks.{i}.adaln_proj.linear.bias"] = None
        # a genuinely quantized sibling, to prove the marker is matched per
        # MODULE and not merely "does any comfy_quant key exist"
        sd[f"blocks.{i}.attn.qkv.comfy_quant"] = None
        sd[f"blocks.{i}.attn.qkv.weight"] = None
    sd["final_layer.adaln_proj.linear.bias"] = None
    return sd


def family_nonpruned_int8():
    """No t-table; the adaln linears themselves ARE ConvRot-quantized, so the
    import consumes their bias before load_state_dict ever sees it."""
    sd = {}
    for i in range(4):
        sd[f"blocks.{i}.adaln_proj.linear.weight"] = None
        sd[f"blocks.{i}.adaln_proj.linear.bias"] = None
        sd[f"blocks.{i}.adaln_proj.linear.comfy_quant"] = None
    sd["final_layer.adaln_proj.linear.bias"] = None
    return sd


def family_bf16_shards():
    """No t-table, nothing quantized anywhere."""
    sd = {}
    for i in range(4):
        sd[f"blocks.{i}.adaln_proj.linear.weight"] = None
        sd[f"blocks.{i}.adaln_proj.linear.bias"] = None
    sd["final_layer.adaln_proj.linear.bias"] = None
    return sd


def main():
    cases = [
        ("(a) pruned int8      ", family_pruned_int8(), True),
        ("(b) non-pruned int8  ", family_nonpruned_int8(), False),
        ("(c) BF16 shards      ", family_bf16_shards(), True),
    ]
    for name, sd, want in cases:
        got = _adaln_block_bias(sd)
        assert got is want, f"{name}: adaln_bias={got}, expected {want}"
        print(f"  [ok] {name} -> adaln_bias {got}")

    # (d) the two historical regressions, as their verbatim key lists
    hist = {k: None for k in ATTEMPT3_UNEXPECTED}
    hist.update({f"blocks.{i}.adaln_proj.linear.weight": None for i in range(8)})
    assert _adaln_block_bias(hist) is True, (
        "attempt-3 regression: unquantized bias keys must resolve to True")
    print("  [ok] (d) attempt-3 unexpected-keys list -> True (was False, the bug)")

    quantized_hist = dict(hist)
    quantized_hist.update(
        {f"blocks.{i}.adaln_proj.linear.comfy_quant": None for i in range(8)})
    assert _adaln_block_bias(quantized_hist) is False, (
        "item-6 regression: ConvRot-consumed bias keys must resolve to False")
    print("  [ok] (d) item-6 missing-keys list -> False (was True, the bug)")

    # neither historical rule passes all four - proof the new one is not just
    # a restatement of either
    for label, rule in (
        ("upstream (bias = pruned)", lambda sd: "adaln_t_table" in sd),
        ("raw-key (bias = key present)",
         lambda sd: any(k.startswith("blocks.") and k.endswith("adaln_proj.linear.bias")
                        for k in sd)),
    ):
        wrong = [n for n, sd, want in cases if rule(sd) is not want]
        assert wrong, f"{label} unexpectedly passes everything"
        print(f"  [ok] {label} still fails {len(wrong)} case(s): {wrong[0].strip()}")

    # the parameter actually reaches the module both ways
    for want in (True, False):
        p = MiniMaxH3TransformerParams()
        p.adaln_bias_from_checkpoint = want
        m = MiniMaxH3AdalnProj(8, 4, expand=6, modalities=3, bias=p.adaln_bias)
        assert (m.linear.bias is not None) is want
    print("  [ok] the resolved flag reaches nn.Linear(bias=...)")

    print("TEST PASS - adaln_bias correct across all four checkpoint families")


if __name__ == "__main__":
    main()
