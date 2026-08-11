"""adaln_bias must be read from the checkpoint, not inferred from pruned-ness.

Regression for the R3 blocker (h3-crossview, 2026-08-11). The fork could not
load MiniMaxAI/MiniMax-H3's non-pruned BF16 Ref2VA shards at all:

  ValueError: MiniMax-H3 transformer load mismatch: missing [],
    unexpected ['blocks.0.adaln_proj.linear.bias', ...]

`adaln_bias` was `adaln_t_table_size is not None` — bias only for PRUNED
checkpoints — on the stated assumption that "the original weights lack" the
block bias. The real non-pruned checkpoint carries it, so the model was built
without the parameter and every real bias tensor was rejected. Three pod
attempts died at 8 seconds each before this was visible.

Pruned-ness only ever CORRELATED with the bias. The key itself is the ground
truth, so that is what the loader now reads.

Usage:  python testing/test_h3_adaln_bias.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from extensions_built_in.diffusion_models.minimax_h3.src.transformer import (
    MiniMaxH3TransformerParams,
)

# verbatim from the R3 attempt-3 pod traceback (truncated at 8 by the error
# formatter); these are exactly the keys that had nowhere to go
ATTEMPT3_UNEXPECTED = [
    "blocks.0.adaln_proj.linear.bias",
    "blocks.1.adaln_proj.linear.bias",
    "blocks.2.adaln_proj.linear.bias",
    "blocks.3.adaln_proj.linear.bias",
    "blocks.4.adaln_proj.linear.bias",
    "blocks.5.adaln_proj.linear.bias",
    "blocks.6.adaln_proj.linear.bias",
    "blocks.7.adaln_proj.linear.bias",
]


def infer(state_dict_keys):
    """Mirror of the loader's inference (minimax_h3.py::_load_transformer)."""
    return any(
        k.startswith("blocks.") and k.endswith("adaln_proj.linear.bias")
        for k in state_dict_keys
    )


def main():
    # --- 1. the regression case: non-pruned checkpoint WITH block bias -----
    non_pruned = ["blocks.0.adaln_proj.linear.weight", "final_layer.adaln_proj.linear.weight",
                  "final_layer.adaln_proj.linear.bias"] + ATTEMPT3_UNEXPECTED
    p = MiniMaxH3TransformerParams()
    p.adaln_bias_from_checkpoint = infer(non_pruned)
    assert p.adaln_t_table_size is None, "this fixture is the NON-pruned shape"
    assert p.adaln_bias is True, (
        "the exact configuration that blocked R3 still resolves to bias=False")
    print("  [ok] non-pruned + block bias -> adaln_bias True "
          "(the R3 blocker; the old rule said False)")

    # --- 2. non-pruned WITHOUT block bias ----------------------------------
    # the shape upstream assumed was universal. The final layer's bias is
    # present here and must NOT be mistaken for a block bias.
    no_block_bias = ["blocks.0.adaln_proj.linear.weight",
                     "final_layer.adaln_proj.linear.weight",
                     "final_layer.adaln_proj.linear.bias"]
    p = MiniMaxH3TransformerParams()
    p.adaln_bias_from_checkpoint = infer(no_block_bias)
    assert p.adaln_bias is False, "the final layer's bias leaked into the answer"
    print("  [ok] non-pruned, no block bias -> False (final-layer bias ignored)")

    # --- 3. pruned checkpoint, bias present --------------------------------
    pruned = ["adaln_t_table"] + ATTEMPT3_UNEXPECTED
    p = MiniMaxH3TransformerParams()
    p.adaln_t_table_size = 8
    p.adaln_bias_from_checkpoint = infer(pruned)
    assert p.adaln_bias is True
    assert p.adaln_apply_silu is False, "pruned checkpoints skip the SiLU"
    print("  [ok] pruned + block bias -> True (unchanged from before)")

    # --- 4. fallback when nothing was inferred -----------------------------
    # None must keep the OLD behaviour, so an unrelated caller that never sets
    # it is not silently changed by this fix
    p = MiniMaxH3TransformerParams()
    assert p.adaln_bias is False
    p.adaln_t_table_size = 8
    assert p.adaln_bias is True
    print("  [ok] uninferred (None) falls back to the pruned-ness heuristic")

    # --- 5. the module actually builds the parameter both ways -------------
    from extensions_built_in.diffusion_models.minimax_h3.src.transformer import (
        MiniMaxH3AdalnProj,
    )
    with_bias = MiniMaxH3AdalnProj(8, 4, expand=6, modalities=3, bias=True)
    without = MiniMaxH3AdalnProj(8, 4, expand=6, modalities=3, bias=False)
    assert with_bias.linear.bias is not None and without.linear.bias is None
    # and a state dict carrying a bias loads into the bias-ful module cleanly
    sd = {"linear.weight": torch.zeros_like(with_bias.linear.weight),
          "linear.bias": torch.zeros_like(with_bias.linear.bias)}
    res = with_bias.load_state_dict(sd, strict=False)
    assert not res.unexpected_keys, f"unexpected: {res.unexpected_keys}"
    res = without.load_state_dict(sd, strict=False)
    assert "linear.bias" in res.unexpected_keys, (
        "the bias-less module should reject a bias - that is the failure being fixed")
    print("  [ok] bias-ful module accepts the bias; bias-less one rejects it")

    print("TEST PASS - adaln_bias is read from the checkpoint, not from pruned-ness")


if __name__ == "__main__":
    main()
