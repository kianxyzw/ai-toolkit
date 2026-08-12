"""Prove the R6b camera encoder is wired to BEHAVIOUR, not just to config.

Exit 0 only if a forward with a nonzero ``cam_pose`` actually perturbs the
target video rows once zero-init is removed — and perturbs nothing else.

Why this script exists rather than a config assertion: this project has now
shipped the same bug three times. ``dit_path`` was silently ignored while we
believed we were training on BF16 shards. ``is_ref2va`` returned False for
``ref2va_pruned``, so previews rendered unconditioned and still printed a
confident ``spatial_std``. And ``model_kwargs.camera_encoder`` was read by
nothing at all, which — because the encoder arm also drops the raymap — would
have trained with no camera conditioning whatsoever and returned a FAIL that
reads as an architecture verdict. The standing rule that came out of it:
**"plumbed to the config" and "plumbed to the behaviour" are separate claims,
so test both.**

Run:  python testing/prove_camera_encoder_wired.py
Also invoked by projects/h3-crossview/scripts/stage_r6_probe.py, which refuses
to declare the encoder arm launch-ready until this exits 0.
"""

import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extensions_built_in.diffusion_models.minimax_h3.src.camera_encoder import (  # noqa: E402
    POSE_DIM, latent_frame_indices, poses_to_latent_vectors)
from extensions_built_in.diffusion_models.minimax_h3.src.transformer import (  # noqa: E402
    MiniMaxH3Transformer, MiniMaxH3TransformerParams)

FAILURES = []


def check(name, fn):
    try:
        fn()
        print(f"  ok   {name}")
    except Exception as e:  # noqa: BLE001
        FAILURES.append(name)
        print(f"  FAIL {name}: {e}")


def pruned_params():
    """The PRUNED structure — D1's base. attention_head_dim >= rope's 96."""
    return MiniMaxH3TransformerParams(
        hidden_size=64, num_layers=2, num_attention_heads=2,
        attention_head_dim=128, ffn_hidden_size=128, text_dim=48,
        time_embed_dim=32, time_embed_hidden_size=64,
        token_refiner_num_layers=1, adaln_t_table_size=32,
        adaln_bias_from_checkpoint=True)


def make_batch(p, n_cond_frames=2, n_target_frames=3, rows_per_frame=4,
               text_len=5, audio_rows=2):
    torch.manual_seed(0)
    n_video = (n_cond_frames + n_target_frames) * rows_per_frame
    seq = text_len + n_video + audio_rows
    text_indices = torch.arange(text_len)
    video_indices = torch.arange(text_len, text_len + n_video)
    audio_indices = torch.arange(text_len + n_video, seq)
    vpd = p.latents_dim * p.patch_size[0] * p.patch_size[1] * p.patch_size[2]
    tags = torch.zeros(1, seq, dtype=torch.long)
    tags[:, text_indices] = 1
    tags[:, audio_indices] = 2
    kw = dict(
        hidden_states=torch.randn(1, n_video, vpd),
        audio_hidden_states=torch.randn(1, audio_rows, p.audio_latents_dim),
        encoder_hidden_states=torch.randn(1, text_len, p.text_dim),
        row_timesteps=torch.full((1, seq), 0.5),
        token_tags=tags, position_ids=torch.zeros(1, seq, 3),
        video_indices=video_indices, audio_indices=audio_indices,
        text_indices=text_indices)
    num_cond = n_cond_frames * rows_per_frame
    return kw, video_indices[num_cond:], n_target_frames, num_cond


# ---------------------------------------------------------------------------

def test_zero_init_is_an_exact_no_op():
    p = pruned_params()
    torch.manual_seed(1)
    model = MiniMaxH3Transformer(p).eval()
    kw, tgt, nf, _ = make_batch(p)
    pose = torch.randn(1, nf, POSE_DIM)
    with torch.no_grad():
        v0, a0 = model(**kw)
        model.attach_camera_encoder(bottleneck=16)
        v1, a1 = model(**kw, cam_pose=pose, target_video_indices=tgt)
    assert torch.equal(v0, v1) and torch.equal(a0, a1), \
        "attaching the encoder changed the output at step 0"


def test_nonzero_pose_perturbs_target_rows_once_zero_init_is_removed():
    """THE claim. Without this, 'wired' means nothing."""
    p = pruned_params()
    torch.manual_seed(2)
    model = MiniMaxH3Transformer(p).eval()
    enc = model.attach_camera_encoder(bottleneck=16)
    kw, tgt, nf, _ = make_batch(p)
    for h in enc.heads:
        torch.nn.init.normal_(h.weight, std=0.5)
    zero_pose = torch.zeros(1, nf, POSE_DIM)
    live_pose = torch.randn(1, nf, POSE_DIM) * 3.0
    with torch.no_grad():
        v_zero, _ = model(**kw, cam_pose=zero_pose, target_video_indices=tgt)
        v_live, _ = model(**kw, cam_pose=live_pose, target_video_indices=tgt)
    assert not torch.equal(v_zero, v_live), (
        "a nonzero cam_pose did not change the output — the pose reaches the "
        "encoder but the encoder does not reach the model")


def test_the_write_touches_target_rows_and_only_target_rows():
    """Measured on the residual stream entering block 0, NOT on the output.

    Attention is global: once target rows move, every other row's OUTPUT moves
    too, and that is conditioning working. Only the write site can separate
    "wrote into audio/reference rows" from "attention propagated".
    """
    p = pruned_params()
    torch.manual_seed(3)
    model = MiniMaxH3Transformer(p).eval()
    enc = model.attach_camera_encoder(bottleneck=16)
    for h in enc.heads:
        torch.nn.init.normal_(h.weight, std=0.5)
    kw, tgt, nf, num_cond = make_batch(p)
    seen = []
    handle = model.blocks[0].register_forward_pre_hook(
        lambda _m, a: seen.append(a[0].detach().clone()))
    try:
        with torch.no_grad():
            model(**kw)
            model(**kw, cam_pose=torch.randn(1, nf, POSE_DIM) * 3.0,
                  target_video_indices=tgt)
    finally:
        handle.remove()
    delta = (seen[1] - seen[0])[0]
    touched = set((delta.abs().sum(-1) > 0).nonzero().flatten().tolist())
    assert touched == set(tgt.tolist()), (
        f"extra rows written: {sorted(touched - set(tgt.tolist()))}; "
        f"missed: {sorted(set(tgt.tolist()) - touched)}")
    for name, idx in (("audio", kw["audio_indices"]),
                      ("text", kw["text_indices"]),
                      ("reference video", kw["video_indices"][:num_cond])):
        assert not (touched & set(idx.tolist())), f"wrote into {name} rows"


def test_adaln_is_untouched():
    p = pruned_params()
    model = MiniMaxH3Transformer(p)
    before = {k: v.clone() for k, v in model.state_dict().items() if "adaln" in k}
    model.attach_camera_encoder(bottleneck=16)
    after = {k: v for k, v in model.state_dict().items() if "adaln" in k}
    assert set(before) == set(after), "adaln key set changed"
    for k in before:
        assert torch.equal(before[k], after[k]), f"adaln tensor {k} modified"


def test_latent_temporal_reduction_matches_the_causal_vae_grid():
    assert latent_frame_indices(73, 19) == [0] + [4 * k for k in range(1, 19)]
    assert latent_frame_indices(73, 19)[-1] == 72, "last latent must see frame 72"
    c2w = torch.zeros(73, 4, 4)
    c2w[:, :3, :3] = torch.eye(3)
    c2w[:, 0, 3] = torch.linspace(0, 5, 73)  # 5 m of travel, metres
    v = poses_to_latent_vectors(c2w, 19)
    assert v.shape == (19, POSE_DIM)
    assert abs(float(v[-1, 3]) - 5.0) < 1e-4


def test_double_centimetre_pose_is_rejected_at_the_reduction():
    c2w = torch.zeros(73, 4, 4)
    c2w[:, :3, :3] = torch.eye(3)
    c2w[:, 0, 3] = torch.linspace(0, 5, 73) / 100.0
    try:
        poses_to_latent_vectors(c2w, 19)
    except ValueError:
        return
    raise AssertionError("accepted a double-converted (centimetre) trajectory")


def test_the_encoder_is_actually_in_the_optimizer():
    """The layer that fails most quietly of all.

    BaseSDTrainProcess calls ``unet.requires_grad_(False)`` and optimizes only
    the LoRA's parameters. The encoder is deliberately excluded from LoRA
    targeting, so without a dedicated param group it stays frozen at zero-init
    — the loss still falls (on the LoRA), the run still succeeds, and the arm
    reports FAIL for an architecture that was never trained.
    """
    src = (ROOT / "extensions_built_in" / "diffusion_models" / "minimax_h3"
           / "minimax_h3.py").read_text(encoding="utf-8", errors="replace")
    assert "def get_additional_training_params" in src, \
        "the model contributes no optimizer group for the camera encoder"
    assert "requires_grad_(True)" in src, \
        "the encoder is never un-frozen after unet.requires_grad_(False)"
    trainer = (ROOT / "extensions_built_in" / "sd_trainer"
               / "SDTrainer.py").read_text(encoding="utf-8", errors="replace")
    assert "get_additional_training_params" in trainer, \
        "SDTrainer never asks the model for extra trainable modules"
    assert "def load_additional_training_modules" in trainer, \
        "SDTrainer does not override the hook the base class provides"

    # behaviour: an optimizer step over that group must move the heads off zero
    p = pruned_params()
    torch.manual_seed(7)
    model = MiniMaxH3Transformer(p)
    enc = model.attach_camera_encoder(bottleneck=16)
    enc.requires_grad_(True)
    assert enc.is_zero_initialized()
    kw, tgt, nf, _ = make_batch(p)
    opt = torch.optim.SGD([q for q in enc.parameters() if q.requires_grad], lr=0.1)
    v, _ = model(**kw, cam_pose=torch.randn(1, nf, POSE_DIM),
                 target_video_indices=tgt)
    v.sum().backward()
    grads = [q.grad for q in enc.heads[0].parameters() if q.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads), \
        "no gradient reached the encoder heads"
    opt.step()
    assert not enc.is_zero_initialized(), \
        "an optimizer step did not move the encoder off zero-init"


def test_the_trained_encoder_is_actually_saved():
    """A trained encoder that is not persisted is a discarded experiment.

    It is excluded from LoRA targeting, so network.save_weights never sees it;
    without a dedicated save path every checkpoint would drop it and the LoRA
    would depend on weights nobody kept.
    """
    src = (ROOT / "extensions_built_in" / "diffusion_models" / "minimax_h3"
           / "minimax_h3.py").read_text(encoding="utf-8", errors="replace")
    assert "def get_additional_save_state_dict" in src,         "the model exposes no save path for the camera encoder"
    base = (ROOT / "jobs" / "process"
            / "BaseSDTrainProcess.py").read_text(encoding="utf-8", errors="replace")
    assert "get_additional_save_state_dict" in base,         "the save loop never asks the model for side-module weights"
    # keys are namespaced and stay out of the LoRA's key space
    p = pruned_params()
    model = MiniMaxH3Transformer(p)
    model.attach_camera_encoder(bottleneck=16)
    import types
    fake = types.SimpleNamespace(model=model)
    from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import (
        MinimaxH3Model)
    sd = MinimaxH3Model.get_additional_save_state_dict(fake)
    assert "camera_encoder" in sd and sd["camera_encoder"], "nothing to save"
    assert all(k.startswith("camera_encoder.") for k in sd["camera_encoder"])
    assert not any(k.startswith("diffusion_model.") for k in sd["camera_encoder"])


def test_the_loader_consumes_the_flag_and_feeds_the_pose():
    """Source-level companion to the behaviour tests above: the forward wiring
    can be perfect while nothing ever turns it on for a real run."""
    src = (ROOT / "extensions_built_in" / "diffusion_models" / "minimax_h3"
           / "minimax_h3.py").read_text(encoding="utf-8", errors="replace")
    assert re.search(r"model_kwargs\.get\(\s*[\"']camera_encoder[\"']", src), \
        "_load_transformer never reads model_kwargs.camera_encoder"
    assert "attach_camera_encoder(" in src, "loader never attaches the encoder"
    assert re.search(r"cam_pose\s*=\s*cam_pose", src), \
        "get_noise_prediction never passes cam_pose to the transformer"
    assert re.search(r"target_video_indices\s*=\s*target_video_indices", src), \
        "get_noise_prediction never passes target_video_indices"
    assert "video_indices[num_cond:]" in src, (
        "target rows are not derived as video_indices[num_cond:] — reference "
        "rows come FIRST, so any other slice writes into the conditioning")
    assert "FileNotFoundError" in src, \
        "a missing pose sidecar must be fatal, not silently skipped"


if __name__ == "__main__":
    print("R6b camera-encoder wiring proof")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            check(name, fn)
    if FAILURES:
        print(f"\nCAMERA ENCODER NOT WIRED — {len(FAILURES)} failed: "
              + ", ".join(FAILURES))
        sys.exit(1)
    print("\nCAMERA ENCODER WIRED — nonzero cam_pose perturbs target rows "
          "(and only those); zero-init is an exact no-op at step 0")
