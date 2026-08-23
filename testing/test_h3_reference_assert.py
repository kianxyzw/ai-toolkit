"""The R6a reference-presence assertion: does the pack actually carry refs?

Pre-launch blocker for the G3 twin probe. The 2026-08-12 hardware check passed
``ITEM_R6A`` on *inference* — no partition error, a plausible loss, an earlier
verification of the same path — because the on-disk check it relied on looked
under a cache naming that does not exist. R6a's entire arm is the reference
path, so presence has to be measured, not inferred.

Two halves, both tested here:

  - the pure geometry check (``src.packing.assert_reference_rows``): presence,
    non-emptiness, alignment, the R5c dropout relaxation, the R6c-E forbid
    inversion, and the R6c-EA EXACT-COUNT branch (AA-2)
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
    ReferenceRowsForbidden,
    ReferenceRowsMissing,
    ReferenceRowsWrongCount,
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
IMAGE_BLOCK = {"kind": "image", "latent_h": 48, "latent_w": 48}


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def raises(name, fn, fragment="", cls=ReferenceRowsMissing):
    try:
        fn()
    except cls as e:
        # exact class for the inverted polarity: a ReferenceRowsMissing where
        # ReferenceRowsForbidden was expected is the WRONG abort, not an abort
        if cls is not ReferenceRowsMissing and type(e) is not cls:
            check(name, False, f"raised {type(e).__name__}, wanted {cls.__name__}")
            return
        check(name, fragment.lower() in str(e).lower(),
              f"message did not mention {fragment!r}: {e}")
        return
    except Exception as e:  # noqa: BLE001
        check(name, False, f"wrong exception type {type(e).__name__}: {e}")
        return
    check(name, False, f"no {cls.__name__} raised")


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


# ---------------------------------------------------------------------------
# 1b. the INVERTED polarity (R6c-E, amendment_r6ce A-2, 2026-08-22)
# ---------------------------------------------------------------------------


def test_pure_inverted():
    print("\npure check, forbid_references=True (R6c-E A-2)")

    # the R6c-E shape: nothing declared, nothing packed -> OK, and the summary
    # says so in the two tokens the orchestrator greps for at pricing
    summary = assert_reference_rows(
        declared_streams=0, ref_blocks=(), ref_row_counts=[],
        num_condition_video_rows=0, forbid_references=True)
    check("zero references under forbid -> OK", "REF_ASSERT_OK" in summary, summary)
    check("summary reports packed_reference_rows=0 forbid_references=True",
          "packed_reference_rows=0" in summary and "forbid_references=True" in summary,
          summary)

    # ⚠ THE CASE THIS EXISTS FOR: the R6c batch (one raymap stream, packed)
    # handed to the R6c-E assertion must ABORT with the forbidden class
    raises("WRONG-MODE COPY: the R6c batch under the R6c-E assertion aborts",
           lambda: assert_reference_rows(
               declared_streams=1, ref_blocks=(VIDEO_BLOCK,), ref_row_counts=[40],
               num_condition_video_rows=40, forbid_references=True),
           "FORBIDDEN", cls=ReferenceRowsForbidden)
    raises("a stray packed block with nothing declared still aborts",
           lambda: assert_reference_rows(
               declared_streams=0, ref_blocks=(VIDEO_BLOCK,), ref_row_counts=[40],
               num_condition_video_rows=40, forbid_references=True),
           "stray reference", cls=ReferenceRowsForbidden)
    raises("a declared stream with nothing packed still aborts (config, not batch)",
           lambda: assert_reference_rows(
               declared_streams=1, ref_blocks=(), ref_row_counts=[],
               num_condition_video_rows=0, forbid_references=True),
           "declares 1", cls=ReferenceRowsForbidden)
    # the mirror: the R6c-E batch (no references) handed to the R6c assertion
    # (one declared stream) aborts the OTHER way - both polarities fail on the
    # wrong-mode copy, neither is "reject every reference-free forward"
    raises("WRONG-MODE COPY: the R6c-E batch under the R6c assertion aborts",
           lambda: assert_reference_rows(
               declared_streams=1, ref_blocks=(), ref_row_counts=[],
               num_condition_video_rows=0, forbid_references=False),
           "ZERO reference blocks")
    # keyframe (i2v) rows are not references: a forbid-mode i2v run may keep them
    summary = assert_reference_rows(
        declared_streams=0, ref_blocks=(), ref_row_counts=[],
        num_condition_video_rows=20, extra_condition_rows=20,
        forbid_references=True)
    check("keyframe rows are not references under forbid", "blocks=0" in summary,
          summary)
    # and the default polarity is unchanged: forbid is opt-in
    summary = assert_reference_rows(
        declared_streams=2, ref_blocks=(VIDEO_BLOCK, VIDEO_BLOCK),
        ref_row_counts=[40, 40], num_condition_video_rows=80)
    check("default polarity unchanged (forbid_references=False in the summary)",
          "forbid_references=False" in summary, summary)


# ---------------------------------------------------------------------------
# 1c. the EXACT-COUNT polarity (R6c-EA, amendment_r6ce_anchored AA-2, 2026-08-23)
# ---------------------------------------------------------------------------


def test_pure_exact_count():
    """A-2 re-inverted a second time: exactly ONE image-kind reference block.

    Zero is the A-1 defect returning (no anchor - the R6c-E circuit E8 judged
    NO-SCENE-ANCHOR); two or more is the R6c/R6-T defect returning (a second
    stream is copyable or command-bearing).
    """
    print("\npure check, require_reference_streams=1 (R6c-EA AA-2)")

    summary = assert_reference_rows(
        declared_streams=1, ref_blocks=(IMAGE_BLOCK,), ref_row_counts=[576],
        num_condition_video_rows=576, require_reference_streams=1,
        require_reference_kind="image")
    check("the anchored shape (one image block) passes",
          "REF_ASSERT_OK" in summary, summary)
    check("summary carries the armed count and the kinds",
          "require_reference_streams=1" in summary
          and "reference_kinds=['image']" in summary, summary)
    check("the anchor's 576 rows are reported",
          "packed_reference_rows=576" in summary, summary)

    # ⚠ WRONG-MODE COPY: the R6c-E batch (nothing declared, nothing packed)
    raises("WRONG-MODE COPY: the R6c-E batch under the R6c-EA assertion aborts",
           lambda: assert_reference_rows(
               declared_streams=0, ref_blocks=(), ref_row_counts=[],
               num_condition_video_rows=0, require_reference_streams=1),
           "A-1 defect returning", cls=ReferenceRowsWrongCount)

    # ⚠ WRONG-MODE COPY: a second stream - the R6c/R6-T shape
    raises("WRONG-MODE COPY: two packed references under R6c-EA abort",
           lambda: assert_reference_rows(
               declared_streams=2, ref_blocks=(IMAGE_BLOCK, VIDEO_BLOCK),
               ref_row_counts=[576, 40], num_condition_video_rows=616,
               require_reference_streams=1),
           "R6c/R6-T defect returning", cls=ReferenceRowsWrongCount)

    # a VIDEO reference where the anchor was required: motion for the
    # preservation prior to copy, which AA-1 (iv) exists to remove
    raises("a video reference where an image was required aborts",
           lambda: assert_reference_rows(
               declared_streams=1, ref_blocks=(VIDEO_BLOCK,), ref_row_counts=[40],
               num_condition_video_rows=40, require_reference_streams=1,
               require_reference_kind="image"),
           "copy\nframe for frame".replace("\n", " "), cls=ReferenceRowsWrongCount)

    # the declaration and the pack must AGREE, both directions
    raises("declared 2 but exactly 1 required aborts on the declaration",
           lambda: assert_reference_rows(
               declared_streams=2, ref_blocks=(IMAGE_BLOCK,), ref_row_counts=[576],
               num_condition_video_rows=576, dropout_configured=True,
               require_reference_streams=1),
           "the dataset declares 2", cls=ReferenceRowsWrongCount)

    # a mode cannot both forbid and require the same channel
    try:
        assert_reference_rows(
            declared_streams=1, ref_blocks=(IMAGE_BLOCK,), ref_row_counts=[576],
            num_condition_video_rows=576, forbid_references=True,
            require_reference_streams=1)
        check("forbid + require is a config error", False, "no ValueError")
    except ValueError as e:
        check("forbid + require is a config error", "both" in str(e), str(e))
    except Exception as e:  # noqa: BLE001
        check("forbid + require is a config error", False, repr(e))

    # and the default polarity is untouched: the branch is opt-in
    summary = assert_reference_rows(
        declared_streams=2, ref_blocks=(VIDEO_BLOCK, VIDEO_BLOCK),
        ref_row_counts=[40, 40], num_condition_video_rows=80)
    check("default polarity unchanged (require_reference_streams=None)",
          "require_reference_streams=None" in summary, summary)


def test_forward_exact_count():
    """The call site under model_kwargs.require_reference_streams (R6c-EA)."""
    print("\ncall site, require_reference_streams=1 (real CPU forward)")
    torch.manual_seed(0)
    num_frames = 5
    latent = torch.randn(1, 24, 2, 8, 10)
    timestep = torch.tensor([500.0])
    MinimaxH3Model._ref_assert_reported = False

    model = make_model("ref2va", {"require_reference_streams": 1,
                                  "require_reference_kind": "image"})
    embeds = make_embeds()
    # an IMAGE reference is one latent frame on its own canvas
    anchor = [torch.randn(1, 24, 1, 8, 10)]
    two = [torch.randn(1, 24, 1, 8, 10), torch.randn(1, 24, 2, 6, 8)]

    batch = make_batch(num_frames, refs=anchor, kinds=["image"],
                       reference_path=["/data/reference"])
    pred = model.get_noise_prediction(latent, timestep, embeds, batch=batch)
    check("R6c-EA batch (one image anchor) trains under the exact-count assertion",
          bool(torch.isfinite(pred).all()))

    # ⚠ WRONG-MODE COPY: the R6c-E batch (no anchor) under the R6c-EA call site
    batch = make_batch(num_frames, refs=None, kinds=None, reference_path=None)
    raises("R6c-E batch under the R6c-EA call site ABORTS",
           lambda: model.get_noise_prediction(
               latent, timestep, embeds, batch=batch),
           "A-1 defect returning", cls=ReferenceRowsWrongCount)

    # ⚠ WRONG-MODE COPY: a second stream under the R6c-EA call site
    batch = make_batch(num_frames, refs=two, kinds=["image", "video"],
                       reference_path=["/data/reference", "/data/raymap_ref"])
    raises("two references under the R6c-EA call site ABORT",
           lambda: model.get_noise_prediction(
               latent, timestep, embeds, batch=batch),
           "R6c/R6-T defect returning", cls=ReferenceRowsWrongCount)

    # ⚠ a video reference where the anchor was required
    batch = make_batch(num_frames, refs=[torch.randn(1, 24, 2, 8, 10)],
                       kinds=["video"], reference_path=["/data/reference"])
    raises("a video reference under the R6c-EA call site ABORTS",
           lambda: model.get_noise_prediction(
               latent, timestep, embeds, batch=batch),
           "video reference", cls=ReferenceRowsWrongCount)

    # ⚠ the mirror wrong-mode copy: the R6c-EA batch under the R6c-E model
    r6ce_model = make_model("ref2va", {"require_zero_references": True})
    batch = make_batch(num_frames, refs=anchor, kinds=["image"],
                       reference_path=["/data/reference"])
    raises("R6c-EA batch under the R6c-E call site ABORTS",
           lambda: r6ce_model.get_noise_prediction(
               latent, timestep, embeds, batch=batch),
           "FORBIDDEN", cls=ReferenceRowsForbidden)

    # sampling still skips: the pipeline supplies its own references
    pred = model.get_noise_prediction(latent, timestep, embeds, batch=None)
    check("batch=None (sampling) skips the exact-count check too",
          bool(torch.isfinite(pred).all()))


def make_model(partition="ref2va", model_kwargs=None):
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
            model_kwargs={"partition": partition, **(model_kwargs or {})},
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


def test_forward_inverted():
    """The call site under model_kwargs.require_zero_references (R6c-E)."""
    print("\ncall site, require_zero_references=True (real CPU forward)")
    torch.manual_seed(0)
    num_frames = 5
    latent = torch.randn(1, 24, 2, 8, 10)
    timestep = torch.tensor([500.0])
    MinimaxH3Model._ref_assert_reported = False

    model = make_model("ref2va", {"require_zero_references": True})
    embeds = make_embeds()
    one_ref = [torch.randn(1, 24, 2, 8, 10)]

    # the R6c-E batch: nothing declared, nothing packed -> trains
    batch = make_batch(num_frames, refs=None, kinds=None, reference_path=None)
    pred = model.get_noise_prediction(latent, timestep, embeds, batch=batch)
    check("R6c-E batch (no references) trains under the inverted assertion",
          bool(torch.isfinite(pred).all()))

    # ⚠ WRONG-MODE COPY: the R6c batch (declared + packed raymap) under the
    # R6c-E model must ABORT with the forbidden class
    batch = make_batch(num_frames, refs=one_ref, kinds=["video"],
                       reference_path=["/data/raymap_ref"])
    raises("R6c batch under the R6c-E call site ABORTS",
           lambda: model.get_noise_prediction(
               latent, timestep, embeds, batch=batch),
           "FORBIDDEN", cls=ReferenceRowsForbidden)

    # a reference that arrived with NO declaration (a dataloader surprise)
    # aborts too: the batch, not the config, is what the pack reads
    batch = make_batch(num_frames, refs=one_ref, kinds=["video"], reference_path=None)
    raises("undeclared stray reference under the R6c-E call site ABORTS",
           lambda: model.get_noise_prediction(
               latent, timestep, embeds, batch=batch),
           "stray reference", cls=ReferenceRowsForbidden)

    # and the mirror wrong-mode copy: the R6c-E batch under the R6c model
    # (one declared stream) aborts the original way
    r6c_model = make_model("ref2va")
    batch = make_batch(num_frames, refs=None, kinds=None,
                       reference_path=["/data/raymap_ref"])
    raises("R6c-E batch under the R6c call site ABORTS (declared, none packed)",
           lambda: r6c_model.get_noise_prediction(
               latent, timestep, embeds, batch=batch),
           "ZERO reference blocks")

    # sampling still skips: the pipeline supplies its own inputs
    pred = model.get_noise_prediction(latent, timestep, embeds, batch=None)
    check("batch=None (sampling) skips the inverted check too",
          bool(torch.isfinite(pred).all()))


def main():
    test_pure()
    test_pure_inverted()
    test_pure_exact_count()
    test_forward()
    test_forward_inverted()
    test_forward_exact_count()
    print()
    if FAILURES:
        print(f"TEST FAIL - {len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("TEST PASS - reference-presence assertion")


if __name__ == "__main__":
    main()
