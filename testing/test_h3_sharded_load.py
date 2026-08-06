"""Test sharded-checkpoint loading for MiniMax H3.

Saves a tiny randomly-initialized MiniMaxH3Transformer as a sharded
safetensors checkpoint (index json + 3 shards), merges it back through
MinimaxH3Model._load_sharded_state_dict, and loads it into a fresh
meta-device transformer with assign=True — the same path the real loader
takes for the original BF16 weights. Checks key completeness, tensor
equality, and the directory/index resolution branches.

Usage:  python testing/test_h3_sharded_load.py
"""

import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import MinimaxH3Model
from extensions_built_in.diffusion_models.minimax_h3.src.transformer import (
    MiniMaxH3Transformer,
    MiniMaxH3TransformerParams,
)

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def main():
    torch.manual_seed(3)
    params = MiniMaxH3TransformerParams(
        hidden_size=64,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=1,
        attention_head_dim=128,
        ffn_hidden_size=96,
        text_dim=48,
        time_embed_hidden_size=32,
        time_embed_dim=16,
    )
    src = MiniMaxH3Transformer(params)
    full_sd = {k: v.contiguous() for k, v in src.state_dict().items()}

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "output", "h3_sharded_load"
    )
    out_dir = os.path.abspath(out_dir)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # split into 3 shards, deliberately interleaved so merge order matters
    keys = sorted(full_sd.keys())
    shards = [keys[0::3], keys[1::3], keys[2::3]]
    weight_map = {}
    for i, shard_keys in enumerate(shards):
        name = f"model-{i + 1:05d}-of-00003.safetensors"
        save_file({k: full_sd[k] for k in shard_keys}, os.path.join(out_dir, name))
        for k in shard_keys:
            weight_map[k] = name
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)

    # merge via the loader helper (index path form)
    merged = MinimaxH3Model._load_sharded_state_dict(
        os.path.join(out_dir, "model.safetensors.index.json")
    )
    check("all keys merged", set(merged.keys()) == set(full_sd.keys()),
          f"{len(merged)} vs {len(full_sd)}")
    check("tensors identical", all(torch.equal(merged[k], full_sd[k]) for k in full_sd))

    # load into a meta-device transformer with assign=True (the real load path)
    with torch.device("meta"):
        dst = MiniMaxH3Transformer(params)
    result = dst.load_state_dict(merged, assign=True, strict=False)
    check("no missing keys", len(result.missing_keys) == 0, str(result.missing_keys[:5]))
    check("no unexpected keys", len(result.unexpected_keys) == 0,
          str(result.unexpected_keys[:5]))
    check("weights materialized", not any(p.is_meta for p in dst.parameters()))
    check("weights equal after assign",
          torch.equal(dst.state_dict()["condition_proj.weight"],
                      full_sd["condition_proj.weight"]))

    print()
    if FAILURES:
        print(f"TEST FAIL - {len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("TEST PASS - sharded checkpoint merge + meta-device assign load")


if __name__ == "__main__":
    main()
