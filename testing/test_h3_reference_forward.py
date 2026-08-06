"""CPU smoke test for MiniMax H3 ref2va reference conditioning.

Runs get_noise_prediction on a tiny randomly-initialized transformer with
reference latents on the batch and checks:

  - the prediction covers exactly the target rows (reference rows excluded)
  - reference rows enter the packed sequence (sequence length grows by the
    reference row count) and their row timesteps pin at the 0.999 noise-aug
    level while target rows carry the drawn timestep
  - references of different canvases than the target work
  - fl2va partition + references raises, do_i2v + references raises

Usage:  python testing/test_h3_reference_forward.py
"""

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import MinimaxH3Model
from extensions_built_in.diffusion_models.minimax_h3.src.packing import (
    KEYFRAME_NOISE_AUG_T,
    audio_latent_num_frames,
)
from extensions_built_in.diffusion_models.minimax_h3.src.transformer import (
    MiniMaxH3Transformer,
    MiniMaxH3TransformerParams,
)
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import ModelConfig

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


class RecordingTransformer(torch.nn.Module):
    """Wraps the tiny transformer and records the forward kwargs."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.last = None

    @property
    def device(self):
        return self.inner.device

    def forward(self, **kwargs):
        self.last = kwargs
        return self.inner(**kwargs)


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
    model.model = RecordingTransformer(MiniMaxH3Transformer(params).float())
    return model


def make_batch(num_frames, refs=None, kinds=None, do_i2v=False):
    return SimpleNamespace(
        dataset_config=SimpleNamespace(do_i2v=do_i2v, do_audio=False),
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
    tags[3:7] = 0  # a vision block
    pe = AdvancedPromptEmbeds(
        text_embeds=[torch.randn(length, dim)], text_token_tags=[tags]
    )
    pe.frozen_dtype_keys = ["text_token_tags"]
    return pe


def main():
    torch.manual_seed(0)
    num_frames = 5  # -> 2 latent frames
    t_lat, h_lat, w_lat = 2, 8, 10
    a_lat = audio_latent_num_frames(num_frames)
    latent = torch.randn(1, 24, t_lat, h_lat, w_lat)
    timestep = torch.tensor([500.0])

    model = make_model("ref2va")
    embeds = make_embeds()

    # --- baseline: no references ------------------------------------------
    batch = make_batch(num_frames)
    pred = model.get_noise_prediction(latent, timestep, embeds, batch=batch)
    check("no-refs prediction shape", tuple(pred.shape) == tuple(latent.shape))
    base_seq = model.model.last["token_tags"].shape[1]

    # --- two video references, one off-canvas -----------------------------
    refs = [torch.randn(1, 24, 2, 8, 10), torch.randn(1, 24, 3, 6, 8)]
    ref_rows = 2 * (8 // 2) * (10 // 2) + 3 * (6 // 2) * (8 // 2)
    batch = make_batch(num_frames, refs=refs, kinds=["video", "video"])
    pred = model.get_noise_prediction(latent, timestep, embeds, batch=batch)
    check("ref prediction shape (target rows only)", tuple(pred.shape) == tuple(latent.shape))
    check("prediction finite", bool(torch.isfinite(pred).all()))
    rec = model.model.last
    check("sequence grew by reference rows",
          rec["token_tags"].shape[1] == base_seq + ref_rows,
          f"{rec['token_tags'].shape[1]} vs {base_seq} + {ref_rows}")
    check("hidden video rows = refs + target",
          rec["hidden_states"].shape[1] == ref_rows + t_lat * (h_lat // 2) * (w_lat // 2))
    row_t = rec["row_timesteps"]
    vidx = rec["video_indices"]
    ref_t = row_t[0, vidx[:ref_rows]]
    tgt_t = row_t[0, vidx[ref_rows:]]
    check("reference rows pinned at 0.999",
          bool((ref_t == KEYFRAME_NOISE_AUG_T).all()), str(ref_t.unique()))
    check("target rows at drawn timestep",
          bool(torch.allclose(tgt_t, torch.full_like(tgt_t, 0.5))), str(tgt_t.unique()))
    check("reference rows tagged video",
          bool((rec["token_tags"][0, vidx[:ref_rows]] == 0).all()))

    # --- image reference ---------------------------------------------------
    batch = make_batch(num_frames, refs=[torch.randn(1, 24, 1, 8, 10)], kinds=["image"])
    pred = model.get_noise_prediction(latent, timestep, embeds, batch=batch)
    check("image ref prediction shape", tuple(pred.shape) == tuple(latent.shape))

    # --- determinism of the layout (same seed, same refs) ------------------
    torch.manual_seed(11)
    batch = make_batch(num_frames, refs=[torch.randn(1, 24, 2, 8, 10)], kinds=["video"])
    p1 = model.get_noise_prediction(latent, timestep, embeds, batch=batch)
    torch.manual_seed(11)
    batch = make_batch(num_frames, refs=[torch.randn(1, 24, 2, 8, 10)], kinds=["video"])
    p2 = model.get_noise_prediction(latent, timestep, embeds, batch=batch)
    check("deterministic under fixed seed", bool(torch.equal(p1, p2)))

    # --- guards ------------------------------------------------------------
    fl_model = make_model("fl2va")
    batch = make_batch(num_frames, refs=[torch.randn(1, 24, 2, 8, 10)], kinds=["video"])
    try:
        fl_model.get_noise_prediction(latent, timestep, embeds, batch=batch)
        check("fl2va + refs rejected", False, "no ValueError")
    except ValueError:
        check("fl2va + refs rejected", True)

    batch = make_batch(num_frames, refs=[torch.randn(1, 24, 2, 8, 10)],
                       kinds=["video"], do_i2v=True)
    try:
        model.get_noise_prediction(latent, timestep, embeds, batch=batch)
        check("do_i2v + refs rejected", False, "no ValueError")
    except ValueError:
        check("do_i2v + refs rejected", True)

    print()
    if FAILURES:
        print(f"TEST FAIL - {len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("TEST PASS - reference conditioning forward")


if __name__ == "__main__":
    main()
