"""The R6a reference-presence assertion: does the pack actually carry refs?

Pre-launch blocker for the G3 twin probe. The 2026-08-12 hardware check passed
``ITEM_R6A`` on *inference* — no partition error, a plausible loss, an earlier
verification of the same path — because the on-disk check it relied on looked
under a cache naming that does not exist. R6a's entire arm is the reference
path, so presence has to be measured, not inferred.

Two halves, both tested here:

  - the pure geometry check (``src.packing.assert_reference_rows``): presence,
    non-emptiness, alignment, and the R5c dropout relaxation
  - the call site inside ``get_noise_prediction``: a real CPU forward through a
    tiny transformer, with references and without, so the abort is proven by
    behaviour rather than by reading the source

⚠ The decisive case is the DELIBERATELY REF-LESS batch: a dataset that declares
two reference streams while the batch carries none. Before this assertion that
run trained happily and reported a normal loss.

Usage:  python testing/test_h3_reference_assert.py
"""

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import MinimaxH3Model
from extensions_built_in.diffusion_models.minimax_h3.src.packing import (
    ReferenceRowsMissing,
    assert_reference_rows,
    audio_latent_num_frames,
)
from extensions_built_in.diffusion_models.minimax_h3.src.transformer import (
    MiniMaxH3Transformer,
    MiniMaxH3TransformerParams,
)
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import ModelConfig

FAILURES = []

VIDEO_BLOCK = {"kind": "video", "latent_t": 2, "latent_h": 8, "latent_w": 10}


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def raises(name, fn, fragment=""):
    try:
        fn()
    except ReferenceRowsMissing as e:
        check(name, fragment.lower() in str(e).lower(),
              f"message did not mention {fragment!r}: {e}")
        return
    except Exception as e:  # noqa: BLE001
        check(name, False, f"wrong exception type {type(e).__name__}: {e}")
        return
    check(name, False, "no ReferenceRowsMissing raised")


# ---------------------------------------------------------------------------
# 1. the pure geometry check
# ---------------------------------------------------------------------------


def test_pure():
    print("\npure check (src.packing.assert_reference_rows)")

    summary = assert_reference_rows(
        declared_streams=2,
        ref_blocks=(VIDEO_BLOCK, VIDEO_BLOCK),
        ref_row_counts=[40, 40],
        num_condition_video_rows=80,
    )
    check("aligned 2-stream batch passes", "REF_ASSERT_OK" in summary, summary)
    check("summary carries the measured row count",
          "packed_reference_rows=80" in summary, summary)

    raises("declared 2, zero blocks -> abort",
           lambda: assert_reference_rows(
               declared_streams=2, ref_blocks=(), ref_row_counts=[],
               num_condition_video_rows=0),
           "ZERO reference blocks")

    raises("declared 2, one block, no dropout -> abort",
           lambda: assert_reference_rows(
               declared_streams=2, ref_blocks=(VIDEO_BLOCK,),
               ref_row_counts=[40], num_condition_video_rows=40),
           "count mismatch")

    summary = assert_reference_rows(
        declared_streams=2, ref_blocks=(VIDEO_BLOCK,), ref_row_counts=[40],
        num_condition_video_rows=40, dropout_configured=True)
    check("declared 2, one block, dropout configured -> allowed",
          "blocks=1" in summary, summary)

    raises("dropout configured still forbids zero references",
           lambda: assert_reference_rows(
               declared_streams=2, ref_blocks=(), ref_row_counts=[],
               num_condition_video_rows=0, dropout_configured=True),
           "ZERO reference blocks")

    raises("blocks present but contributing no rows -> abort",
           lambda: assert_reference_rows(
               declared_streams=1, ref_blocks=(VIDEO_BLOCK,),
               ref_row_counts=[0], num_condition_video_rows=0),
           "0 rows")

    raises("layout/caller row mismatch -> abort",
           lambda: assert_reference_rows(
               declared_streams=2, ref_blocks=(VIDEO_BLOCK, VIDEO_BLOCK),
               ref_row_counts=[40, 40], num_condition_video_rows=41),
           "misaligned")

    # keyframe (i2v) rows count into the same reserve; refs never combine with
    # them, but the arithmetic has to admit them or a valid i2v run would abort
    summary = assert_reference_rows(
        declared_streams=0, ref_blocks=(), ref_row_counts=[],
        num_condition_video_rows=20, extra_condition_rows=20)
    check("keyframe-only condition rows accepted", "blocks=0" in summary, summary)

    # a t2va run declares nothing and packs nothing: the check must stay out of
    # its way, or every non-reference config in the repo starts failing
    summary = assert_reference_rows(
        declared_streams=0, ref_blocks=(), ref_row_counts=[],
        num_condition_video_rows=0)
    check("undeclared, unreferenced run passes untouched",
          "packed_reference_rows=0" in summary, summary)


# ---------------------------------------------------------------------------
# 2. the call site, through a real forward
# ---------------------------------------------------------------------------


def make_model(partition="ref2va"):
    torch.manual_seed(7)
    params = MiniMaxH3TransformerParams(
        hidden_size=128,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=128,
        ffn_hidden_size=256,
        text_dim=64,
        time_embed_hidden_size=64,
        time_embed_dim=32,
    )
    model = MinimaxH3Model(
        device="cpu",
        model_config=ModelConfig(
            name_or_path="dummy",
            arch="minimax_h3",
            model_kwargs={"partition": partition},
            dtype="float32",
        ),
        dtype="float32",
    )
    model.model = MiniMaxH3Transformer(params).float()
    return model


def make_batch(num_frames, refs=None, kinds=None, reference_path=None,
               reference_dropout=0.0):
    return SimpleNamespace(
        dataset_config=SimpleNamespace(
            do_i2v=False,
            do_audio=False,
            reference_path=reference_path,
            reference_dropout=reference_dropout,
            reference_dropout_end=None,
        ),
        num_frames=num_frames,
        first_frame_latents=None,
        tensor=None,
        audio_latents=None,
        audio_data=None,
        audio_noise=None,
        audio_target=None,
        audio_pred_slot=None,
        reference_latents=refs,
        reference_kinds=kinds,
        reference_tensors=None,
    )


def make_embeds(length=11, dim=64):
    tags = torch.ones(length, dtype=torch.long)
    tags[3:7] = 0
    pe = AdvancedPromptEmbeds(
        text_embeds=[torch.randn(length, dim)], text_token_tags=[tags]
    )
    pe.frozen_dtype_keys = ["text_token_tags"]
    return pe


def test_forward():
    print("\ncall site (get_noise_prediction, real CPU forward)")
    torch.manual_seed(0)
    num_frames = 5
    latent = torch.randn(1, 24, 2, 8, 10)
    timestep = torch.tensor([500.0])
    audio_latent_num_frames(num_frames)  # keeps the import honest

    model = make_model("ref2va")
    embeds = make_embeds()
    two_refs = [torch.randn(1, 24, 2, 8, 10), torch.randn(1, 24, 2, 6, 8)]
    paths = ["/data/source_ref", "/data/raymap_ref"]

    # the R6a config, as configured: two declared streams, two present
    batch = make_batch(num_frames, refs=two_refs, kinds=["video", "video"],
                       reference_path=paths)
    pred = model.get_noise_prediction(latent, timestep, embeds, batch=batch)
    check("normal 2-reference batch trains", bool(torch.isfinite(pred).all()))

    # ⚠ THE CASE THIS EXISTS FOR. The config still declares both reference
    # streams; the batch arrives with none — the dataloader branch went silent,
    # the cache was empty, the partition was wrong. Before this assertion the
    # step ran and the loss looked entirely normal.
    batch = make_batch(num_frames, refs=None, kinds=None, reference_path=paths)
    raises("deliberately ref-less batch ABORTS",
           lambda: model.get_noise_prediction(
               latent, timestep, embeds, batch=batch),
           "ZERO reference blocks")

    # ... and it is the DECLARATION that makes it an abort, not the absence of
    # references. Without this, the test would pass against an assertion that
    # simply rejects every reference-free forward — including plain t2va.
    batch = make_batch(num_frames, refs=None, kinds=None, reference_path=None)
    pred = model.get_noise_prediction(latent, timestep, embeds, batch=batch)
    check("undeclared ref-less batch still runs (t2va path untouched)",
          bool(torch.isfinite(pred).all()))

    # half the references lost, no dropout configured to explain it
    batch = make_batch(num_frames, refs=two_refs[:1], kinds=["video"],
                       reference_path=paths)
    raises("one of two references lost ABORTS",
           lambda: model.get_noise_prediction(
               latent, timestep, embeds, batch=batch),
           "count mismatch")

    # ... unless R5c dropout is configured, where one stream dropping is the
    # curriculum working as designed
    batch = make_batch(num_frames, refs=two_refs[:1], kinds=["video"],
                       reference_path=paths, reference_dropout=[0.0, 0.3])
    pred = model.get_noise_prediction(latent, timestep, embeds, batch=batch)
    check("dropout-configured partial reference set trains",
          bool(torch.isfinite(pred).all()))

    # sampling has no dataset config to state an expectation
    pred = model.get_noise_prediction(latent, timestep, embeds, batch=None)
    check("batch=None (sampling) skips the check",
          bool(torch.isfinite(pred).all()))


def main():
    test_pure()
    test_forward()
    print()
    if FAILURES:
        print(f"TEST FAIL - {len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("TEST PASS - reference-presence assertion")


if __name__ == "__main__":
    main()
