"""R6b camera-encoder module — local tests, no weights, no GPU.

The three claims the R6b brief requires before the arm may run, plus the
guards for the two ways this module can fail silently.

Run:  python testing/test_h3_camera_encoder.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extensions_built_in.diffusion_models.minimax_h3.src.camera_encoder import (  # noqa: E402
    MiniMaxH3CameraEncoder, POSE_DIM, add_camera_embedding,
    encoder_state_dict_prefix, pose_vectors_from_c2w)
from extensions_built_in.diffusion_models.minimax_h3.src.transformer import (  # noqa: E402
    MiniMaxH3Transformer, MiniMaxH3TransformerParams)

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  ok   {name}")
    except Exception as e:  # noqa: BLE001
        FAIL.append((name, e))
        print(f"  FAIL {name}: {e}")


def tiny_params(**kw):
    """A structurally faithful but small H3: same shapes, 2 blocks."""
    # attention_head_dim must be >= rope's 96 output channels (2 * 3 *
    # rope_inv_freq_len); 128 is the shipped value, so keep it real.
    base = dict(hidden_size=64, num_layers=2, num_attention_heads=2,
                attention_head_dim=128, ffn_hidden_size=128, text_dim=48,
                time_embed_dim=32, time_embed_hidden_size=64,
                token_refiner_num_layers=1)
    base.update(kw)
    return MiniMaxH3TransformerParams(**base)


def pruned_params(**kw):
    """The PRUNED structure — adaln driven by a t-table, block bias present.

    R6b must resolve against this, not against the BF16 shards: D1 reverted to
    the pruned repack, and the 2026-08-12 verification found three separate
    bugs from code that assumed the non-pruned structure.
    """
    # adaln_apply_silu is a derived @property (False when a t-table is
    # present), not a constructor argument — passing it is a TypeError.
    return tiny_params(adaln_t_table_size=32, adaln_bias_from_checkpoint=True,
                       **kw)


def make_batch(p, num_cond_frames=1, num_target_frames=3, rows_per_frame=4,
               text_len=5, audio_rows=2, batch=1, seed=0):
    """A packed sequence with reference video rows FIRST, then target rows."""
    torch.manual_seed(seed)
    n_video = (num_cond_frames + num_target_frames) * rows_per_frame
    seq = text_len + n_video + audio_rows
    text_indices = torch.arange(text_len)
    video_indices = torch.arange(text_len, text_len + n_video)
    audio_indices = torch.arange(text_len + n_video, seq)
    video_patch_dim = p.latents_dim * p.patch_size[0] * p.patch_size[1] * p.patch_size[2]
    tags = torch.zeros(batch, seq, dtype=torch.long)
    tags[:, text_indices] = 1
    tags[:, audio_indices] = 2
    return dict(
        hidden_states=torch.randn(batch, n_video, video_patch_dim),
        audio_hidden_states=torch.randn(batch, audio_rows, p.audio_latents_dim),
        encoder_hidden_states=torch.randn(batch, text_len, p.text_dim),
        row_timesteps=torch.full((batch, seq), 0.5),
        token_tags=tags,
        position_ids=torch.zeros(batch, seq, 3),
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
    ), video_indices[num_cond_frames * rows_per_frame:], num_target_frames


# ---------------------------------------------------------------------------
# 1. zero-init proves a no-op at step 0 — BIT-identical, not "close"
# ---------------------------------------------------------------------------

def test_zero_init_is_bit_identical_to_no_encoder():
    p = pruned_params()
    torch.manual_seed(1)
    model = MiniMaxH3Transformer(p).eval()
    batch, target_idx, n_frames = make_batch(p)
    pose = torch.randn(1, n_frames, POSE_DIM)

    with torch.no_grad():
        before_v, before_a = model(**batch)
        model.attach_camera_encoder(bottleneck=16)
        after_v, after_a = model(**batch, cam_pose=pose,
                                 target_video_indices=target_idx)

    assert torch.equal(before_v, after_v), "video output changed at step 0"
    assert torch.equal(before_a, after_a), "audio output changed at step 0"


def test_encoder_reports_zero_initialized_and_stops_being_a_no_op_once_trained():
    """The positive control. A no-op test passes trivially if the module is
    never wired in — this proves the wiring is live by breaking it."""
    p = pruned_params()
    torch.manual_seed(1)
    model = MiniMaxH3Transformer(p).eval()
    enc = model.attach_camera_encoder(bottleneck=16)
    assert enc.is_zero_initialized()
    batch, target_idx, n_frames = make_batch(p)
    pose = torch.randn(1, n_frames, POSE_DIM)

    with torch.no_grad():
        base, _ = model(**batch, cam_pose=pose, target_video_indices=target_idx)
        for head in enc.heads:  # simulate a training step having happened
            torch.nn.init.normal_(head.weight, std=0.05)
        assert not enc.is_zero_initialized()
        moved, _ = model(**batch, cam_pose=pose, target_video_indices=target_idx)

    assert not torch.equal(base, moved), \
        "output did not change after training the heads — the encoder is not wired in"


def test_gradients_reach_the_encoder():
    p = pruned_params()
    torch.manual_seed(1)
    model = MiniMaxH3Transformer(p)
    enc = model.attach_camera_encoder(bottleneck=16)
    batch, target_idx, n_frames = make_batch(p)
    pose = torch.randn(1, n_frames, POSE_DIM)
    v, _ = model(**batch, cam_pose=pose, target_video_indices=target_idx)
    v.sum().backward()
    assert enc.heads[0].weight.grad is not None
    assert enc.trunk[0].weight.grad is not None
    assert enc.trunk[0].weight.grad.abs().sum() == 0, (
        "trunk received a non-zero gradient through zero-init heads — "
        "the chain rule says it must be exactly zero on the first step")


# ---------------------------------------------------------------------------
# 2. shapes resolve on the PRUNED structure
# ---------------------------------------------------------------------------

def test_shapes_resolve_on_the_pruned_structure():
    p = pruned_params()
    model = MiniMaxH3Transformer(p)
    enc = model.attach_camera_encoder(bottleneck=16)
    assert model.time_embedder is None, "control: this is the pruned structure"
    assert hasattr(model, "adaln_t_table")
    assert len(enc.heads) == p.num_layers
    assert enc.heads[0].out_features == p.hidden_size
    batch, target_idx, n_frames = make_batch(p)
    pose = torch.randn(1, n_frames, POSE_DIM)
    v, a = model(**batch, cam_pose=pose, target_video_indices=target_idx)
    assert v.shape[1] == batch["hidden_states"].shape[1]
    assert a.shape[1] == batch["audio_hidden_states"].shape[1]


def test_shapes_resolve_on_the_non_pruned_structure_too():
    p = tiny_params()
    model = MiniMaxH3Transformer(p)
    model.attach_camera_encoder(bottleneck=16)
    assert model.time_embedder is not None, "control: non-pruned"
    batch, target_idx, n_frames = make_batch(p)
    model(**batch, cam_pose=torch.randn(1, n_frames, POSE_DIM),
          target_video_indices=target_idx)


def test_full_size_head_count_and_parameter_budget():
    """The bottleneck is the reason this arm is affordable on H3.

    ReCamMaster's per-block ``dim x dim`` projector at H3's 5376/50 would be
    1.45B parameters. Assert the deviation actually bought what it claims.
    """
    p = MiniMaxH3TransformerParams()
    enc = MiniMaxH3CameraEncoder(p.hidden_size, p.num_layers, bottleneck=256)
    n = sum(x.numel() for x in enc.parameters())
    recam_style = p.num_layers * (p.hidden_size * p.hidden_size + p.hidden_size)
    assert n < recam_style / 10, f"{n/1e6:.1f}M is not << {recam_style/1e6:.0f}M"
    assert n < 100e6, f"{n/1e6:.1f}M parameters is too large for a side-module"


# ---------------------------------------------------------------------------
# 3. it writes to TARGET video rows only — never refs, adaln, or audio
# ---------------------------------------------------------------------------

def test_embedding_lands_only_on_target_rows():
    x = torch.zeros(1, 10, 4)
    delta = torch.ones(1, 3, 4)
    idx = torch.tensor([5, 6, 7])
    out = add_camera_embedding(x, delta, idx)
    assert torch.equal(out[0, 5:8], torch.ones(3, 4))
    untouched = torch.cat([out[0, :5], out[0, 8:]])
    assert torch.equal(untouched, torch.zeros(7, 4)), "wrote outside the target rows"


def test_the_write_lands_only_on_target_rows_in_the_real_residual_stream():
    """The specific corruption this arm could cause: reference rows come FIRST
    in video_indices, so an off-by-num_cond slice writes the camera embedding
    into the conditioning the model is supposed to read.

    ⚠ Asserted at the WRITE SITE, on the residual stream entering block 0 —
    not on the model output. Attention is global, so once the target rows
    change, every other row's *output* changes too; that is conditioning
    working as designed. An end-to-end check therefore cannot distinguish
    "wrote into the reference rows" from "attention propagated", and would
    fail on correct code (it did, when this test was first written that way).
    """
    p = pruned_params()
    torch.manual_seed(3)
    model = MiniMaxH3Transformer(p).eval()
    enc = model.attach_camera_encoder(bottleneck=16)
    for head in enc.heads:
        torch.nn.init.normal_(head.weight, std=0.5)
    n_cond_frames, rows_per_frame = 2, 4
    batch, target_idx, n_frames = make_batch(
        p, num_cond_frames=n_cond_frames, rows_per_frame=rows_per_frame)
    pose = torch.randn(1, n_frames, POSE_DIM)

    seen = []
    handle = model.blocks[0].register_forward_pre_hook(
        lambda _m, args: seen.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            model(**batch)
            model(**batch, cam_pose=pose, target_video_indices=target_idx)
    finally:
        handle.remove()

    base, moved = seen
    delta = (moved - base)[0]
    touched = set((delta.abs().sum(-1) > 0).nonzero().flatten().tolist())
    assert touched == set(target_idx.tolist()), (
        f"wrote to rows {sorted(touched - set(target_idx.tolist()))} outside the "
        f"target set; missed {sorted(set(target_idx.tolist()) - touched)}")
    # and the rows it must never reach, named explicitly
    for name, idx in (("text", batch["text_indices"]),
                      ("audio", batch["audio_indices"]),
                      ("reference video", batch["video_indices"][:n_cond_frames
                                                                 * rows_per_frame])):
        assert not (touched & set(idx.tolist())), f"wrote into {name} rows"


def test_adaln_parameters_are_untouched():
    p = pruned_params()
    model = MiniMaxH3Transformer(p)
    before = {k: v.clone() for k, v in model.state_dict().items() if "adaln" in k}
    model.attach_camera_encoder(bottleneck=16)
    after = {k: v for k, v in model.state_dict().items() if "adaln" in k}
    assert set(before) == set(after), "attaching the encoder changed adaln keys"
    for k in before:
        assert torch.equal(before[k], after[k]), f"adaln tensor {k} was modified"


def test_missing_target_indices_is_an_error_not_a_guess():
    p = pruned_params()
    model = MiniMaxH3Transformer(p)
    model.attach_camera_encoder(bottleneck=16)
    batch, _, n_frames = make_batch(p)
    try:
        model(**batch, cam_pose=torch.randn(1, n_frames, POSE_DIM))
    except ValueError:
        return
    raise AssertionError("guessed the target rows instead of refusing")


def test_row_frame_mismatch_is_rejected():
    p = pruned_params()
    model = MiniMaxH3Transformer(p)
    model.attach_camera_encoder(bottleneck=16)
    batch, target_idx, n_frames = make_batch(p)
    try:
        model(**batch, cam_pose=torch.randn(1, n_frames + 2, POSE_DIM),
              target_video_indices=target_idx)
    except ValueError:
        return
    raise AssertionError("accepted a pose sequence that does not divide the rows")


def test_frames_expand_frame_major_not_tiled():
    """repeat_interleave vs repeat. Both give the right SHAPE; only one pairs
    each frame with its own rows."""
    enc = MiniMaxH3CameraEncoder(hidden_size=4, num_layers=1, bottleneck=3)
    torch.nn.init.eye_(enc.heads[0].weight[:3, :3])
    feats = torch.tensor([[[1.0, 0, 0], [0, 1.0, 0]]])
    out = enc.block_delta(0, feats, rows_per_frame=2)
    assert out.shape == (1, 4, 4)
    assert torch.equal(out[0, 0], out[0, 1]), "rows of frame 0 must share an embedding"
    assert torch.equal(out[0, 2], out[0, 3]), "rows of frame 1 must share an embedding"
    assert not torch.equal(out[0, 0], out[0, 2]), "frames must differ (tiled, not interleaved)"


# ---------------------------------------------------------------------------
# 4. LoRA separability + the unit-scale guard
# ---------------------------------------------------------------------------

def test_encoder_keys_are_explicitly_named_and_not_diffusion_model_prefixed():
    """Item 5b pinned the LoRA artifact at 416 tensors, ALL ``diffusion_model.``
    prefixed with zero adaln keys. The encoder must be separable from it."""
    p = pruned_params()
    model = MiniMaxH3Transformer(p)
    enc = model.attach_camera_encoder(bottleneck=16)
    keys = list(enc.state_dict())
    assert keys, "encoder has no parameters"
    assert not any(k.startswith("diffusion_model.") for k in keys)
    assert encoder_state_dict_prefix() == "camera_encoder."
    in_model = [k for k in model.state_dict() if k.startswith("camera_encoder.")]
    assert len(in_model) == len(keys), "encoder keys are not under one prefix"
    # every encoder key is findable by the exclusion substring a config uses
    assert all("camera_encoder" in k for k in in_model)


def test_lora_targeting_sweeps_up_the_encoder_unless_it_is_excluded():
    """The claim the R6b brief actually makes, measured with upstream's own
    selection rule rather than by inspecting key names.

    ai-toolkit's LoRA targeting selects by module CLASS under
    ``target_lora_modules = ["MiniMaxH3Transformer"]``, so every ``nn.Linear``
    attached under the transformer is a candidate — including the encoder's.
    Three counts, and the middle one is the positive control: without it,
    "excluded" could mean "was never a candidate", and the config line would
    be cargo-cult.
    """
    from toolkit.lora_special import LINEAR_MODULES

    def count(root, ignore):
        n = 0
        for _, module in root.named_modules():
            if module.__class__.__name__ != "MiniMaxH3Transformer":
                continue
            for child_name, child in module.named_modules():
                if child.__class__.__name__ not in LINEAR_MODULES:
                    continue
                if not any(w in child_name for w in ignore):
                    n += 1
        return n

    p = MiniMaxH3TransformerParams()
    p.adaln_t_table_size = 1000            # the PRUNED structure
    p.adaln_bias_from_checkpoint = True
    with torch.device("meta"):
        model = MiniMaxH3Transformer(p)
        baseline = count(model, ("adaln_proj",))
        model.attach_camera_encoder(bottleneck=256)
        swept = count(model, ("adaln_proj",))
        excluded = count(model, ("adaln_proj", "camera_encoder"))

    assert swept > baseline, (
        "control failed: the encoder was never a LoRA candidate, so excluding "
        "it proves nothing — re-check the targeting rule before trusting this")
    assert swept - baseline == p.num_layers + 2, (
        f"expected {p.num_layers} heads + 2 trunk linears to be swept up, "
        f"got {swept - baseline}")
    assert excluded == baseline, (
        f"ignore_if_contains=['adaln_proj','camera_encoder'] left {excluded} "
        f"targets, want the pre-encoder {baseline} — the LoRA artifact must be "
        "unchanged by attaching this module")


def test_encoder_saves_and_loads_separately_from_the_lora():
    """Item 5b pinned the LoRA at 416 tensors, all `diffusion_model.` prefixed.
    The encoder is a SECOND artifact: its own keys, its own file, and a
    round-trip that does not touch the LoRA's key space."""
    import io

    p = pruned_params()
    model = MiniMaxH3Transformer(p)
    enc = model.attach_camera_encoder(bottleneck=16)
    for h in enc.heads:  # give it something to lose
        torch.nn.init.normal_(h.weight, std=0.1)
    sd = enc.state_dict()
    assert not any(k.startswith("diffusion_model.") for k in sd)

    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    enc.zero_init()
    assert enc.is_zero_initialized()
    enc.load_state_dict(torch.load(buf, weights_only=True))
    assert not enc.is_zero_initialized(), "round-trip lost the trained heads"
    for k, v in sd.items():
        assert torch.equal(enc.state_dict()[k], v), f"{k} did not round-trip"

    # and the two key spaces do not overlap
    lora_space = {k for k in model.state_dict()
                  if "adaln_proj" not in k and "camera_encoder" not in k}
    enc_space = {f"camera_encoder.{k}" for k in sd}
    assert not (lora_space & enc_space)


def test_encoder_works_with_the_raymap_reference_dropped():
    """The encoder arm's actual configuration: ONE reference stream (source),
    not two. Its whole premise is that the trajectory arrives as numbers, so
    the packed sequence it trains on has fewer condition rows than R6a's —
    and the target-row slice must still be right."""
    p = pruned_params()
    torch.manual_seed(5)
    model = MiniMaxH3Transformer(p).eval()
    enc = model.attach_camera_encoder(bottleneck=16)
    for h in enc.heads:
        torch.nn.init.normal_(h.weight, std=0.5)
    # one reference stream => one block of condition rows
    batch, target_idx, n_frames = make_batch(p, num_cond_frames=1, rows_per_frame=4)
    pose = torch.randn(1, n_frames, POSE_DIM)
    seen = []
    handle = model.blocks[0].register_forward_pre_hook(
        lambda _m, a: seen.append(a[0].detach().clone()))
    try:
        with torch.no_grad():
            model(**batch)
            model(**batch, cam_pose=pose, target_video_indices=target_idx)
    finally:
        handle.remove()
    delta = (seen[1] - seen[0])[0]
    touched = set((delta.abs().sum(-1) > 0).nonzero().flatten().tolist())
    assert touched == set(target_idx.tolist())
    # the single reference block is still untouched
    assert not (touched & set(batch["video_indices"][:4].tolist()))


def test_latent_reduction_and_pose_helpers():
    from extensions_built_in.diffusion_models.minimax_h3.src.camera_encoder import (
        latent_frame_indices, poses_to_latent_vectors)
    assert latent_frame_indices(73, 19) == [0] + [4 * k for k in range(1, 19)]
    assert latent_frame_indices(1, 1) == [0]
    # non-4x counts must degrade, not raise, inside a training step
    idx = latent_frame_indices(50, 13)
    assert len(idx) == 13 and idx[0] == 0 and idx[-1] == 49
    c2w = torch.zeros(73, 4, 4)
    c2w[:, :3, :3] = torch.eye(3)
    c2w[:, 0, 3] = torch.linspace(0, 5, 73)
    assert poses_to_latent_vectors(c2w, 19).shape == (19, POSE_DIM)


def test_pose_vectors_layout_matches_recammaster():
    c2w = torch.zeros(2, 4, 4)
    c2w[:, :3, :3] = torch.eye(3)
    c2w[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    c2w[1, :3, 3] = torch.tensor([4.0, 5.0, 6.0])
    v = pose_vectors_from_c2w(c2w)
    assert v.shape == (2, POSE_DIM)
    # row-major 3x4: [R00 R01 R02 t0 | R10 R11 R12 t1 | R20 R21 R22 t2]
    assert torch.equal(v[0], torch.tensor([1., 0, 0, 1., 0, 1., 0, 2., 0, 0, 1., 3.]))


def test_a_static_trajectory_with_float_noise_is_accepted():
    """E3 (2026-08-22): a static camera anchored to its own frame 0 carries a
    ~1e-15 m translation span - float noise. The scale check must not read
    it as a double centimetre conversion (which would be ~0.006 m, four
    orders of magnitude larger); a static command is a legitimate command
    (13.6% of the manifest, 3 of R6c-E's 36)."""
    from extensions_built_in.diffusion_models.minimax_h3.src.camera_encoder import (
        poses_to_latent_vectors)
    c2w = torch.eye(4).repeat(73, 1, 1)
    c2w[:, :3, 3] = torch.tensor([1.5e-15, -2.0e-16, 0.0])
    v = poses_to_latent_vectors(c2w, 22)
    assert v.shape == (22, POSE_DIM)
    c2w[:, :3, 3] = 0.0
    poses_to_latent_vectors(c2w, 22)


def test_double_centimetre_conversion_is_rejected():
    """ReCamMaster divides translations by 100; extrinsics.py already did.
    Re-applying leaves ~1 cm of motion — trains fine, conditions on nothing,
    and reads as an architecture result. Same shape as the R1 bug."""
    c2w = torch.zeros(3, 4, 4)
    c2w[:, :3, :3] = torch.eye(3)
    c2w[:, 0, 3] = torch.tensor([0.0, 2.5, 5.0])  # metres — fine
    pose_vectors_from_c2w(c2w)
    try:
        pose_vectors_from_c2w(c2w / 100.0)  # the double conversion
    except ValueError as e:
        assert "twice" in str(e)
        return
    raise AssertionError("accepted double-converted centimetres")


# ---------------------------------------------------------------------------
# 5. E12b — the output GAIN knob (one multiplicative scale, config-plumbed)
#
# The knob exists because R6c-EA's trained channel moves the output in 2 of 4
# sources and sits at or under the generator's noise floor in the other two
# (E12, 2026-08-24). Scaling the SAME weights separates "weak but right" from
# "command-uncorrelated noise". Everything below pins the three properties the
# sweep's reading depends on; break any one and the sweep measures something
# else.
# ---------------------------------------------------------------------------

def test_gain_one_is_an_exact_no_op_against_the_ungained_module():
    """Default 1.0 must be BIT-identical, on TRAINED heads.

    Not "close": the multiply is skipped rather than performed, so the op
    sequence is the un-gained one. If this ever becomes an approximation, the
    sweep's gain=1.0 arm stops being a re-run of the banked run.
    """
    p = pruned_params()
    torch.manual_seed(7)
    model = MiniMaxH3Transformer(p).eval()
    enc = model.attach_camera_encoder(bottleneck=16)
    for head in enc.heads:                       # a trained encoder, not zeros
        torch.nn.init.normal_(head.weight, std=0.05)
        torch.nn.init.normal_(head.bias, std=0.05)
    batch, target_idx, n_frames = make_batch(p)
    pose = torch.randn(1, n_frames, POSE_DIM)

    assert enc.gain == 1.0, "the default gain is not 1.0"
    with torch.no_grad():
        a_v, a_a = model(**batch, cam_pose=pose, target_video_indices=target_idx)
        # the same module with the knob explicitly set to its default
        enc.set_gain(1.0)
        b_v, b_a = model(**batch, cam_pose=pose, target_video_indices=target_idx)
    assert torch.equal(a_v, b_v) and torch.equal(a_a, b_a)

    feats = enc.trunk_features(pose)
    d1 = enc.block_delta(0, feats, 4)
    plain = enc.heads[0](feats).repeat_interleave(4, dim=1)
    assert torch.equal(d1, plain), "gain=1.0 changed the injected tensor"


def test_gain_zero_kills_the_channel_exactly_like_zero_init():
    """0.0 must reproduce the untrained base bit-for-bit, whatever the heads
    hold — the control arm, and the property that makes the knob safe to hand
    a distilled base."""
    p = pruned_params()
    torch.manual_seed(8)
    model = MiniMaxH3Transformer(p).eval()
    batch, target_idx, n_frames = make_batch(p)
    pose = torch.randn(1, n_frames, POSE_DIM)
    with torch.no_grad():
        no_encoder_v, no_encoder_a = model(**batch)

    enc = model.attach_camera_encoder(bottleneck=16)
    for head in enc.heads:
        torch.nn.init.normal_(head.weight, std=0.5)
        torch.nn.init.normal_(head.bias, std=0.5)
    with torch.no_grad():
        live_v, _ = model(**batch, cam_pose=pose, target_video_indices=target_idx)
        enc.set_gain(0.0)
        dead_v, dead_a = model(**batch, cam_pose=pose,
                               target_video_indices=target_idx)
    assert not torch.equal(live_v, no_encoder_v), \
        "control: trained heads do move the output"
    assert torch.equal(dead_v, no_encoder_v), "gain=0.0 did not kill the channel"
    assert torch.equal(dead_a, no_encoder_a)
    assert not enc.is_zero_initialized(), \
        "control: the WEIGHTS are still trained — only the gain is zero"


def test_gain_scales_the_injected_tensor_exactly_at_powers_of_two():
    """The sweep's {1, 2, 4, 8} are powers of two, so the scaling is exact in
    floating point and this is an equality, not a tolerance. A gain that only
    approximately scaled would blur the very trend the read is registered on.
    """
    p = pruned_params()
    torch.manual_seed(9)
    model = MiniMaxH3Transformer(p).eval()
    enc = model.attach_camera_encoder(bottleneck=16)
    for head in enc.heads:
        torch.nn.init.normal_(head.weight, std=0.05)
        torch.nn.init.normal_(head.bias, std=0.05)
    _, _, n_frames = make_batch(p)
    feats = enc.trunk_features(torch.randn(1, n_frames, POSE_DIM))
    layer = p.num_layers - 1          # the LAST block, not a guessed index
    base = enc.block_delta(layer, feats, 4)
    for g in (2.0, 4.0, 8.0):
        enc.set_gain(g)
        assert torch.equal(enc.block_delta(layer, feats, 4), base * g), f"gain {g}"


def test_gain_scales_the_TARGET_ROWS_ONLY_at_the_write_site():
    """Scaling must not reach a row the un-gained module does not write.

    Asserted at the write site (the residual stream entering block 0), for the
    same reason the R6b write test is: attention is global, so an end-to-end
    check cannot tell "wrote elsewhere" from "propagated", and would fail on
    correct code.
    """
    p = pruned_params()
    torch.manual_seed(10)
    model = MiniMaxH3Transformer(p).eval()
    enc = model.attach_camera_encoder(bottleneck=16)
    for head in enc.heads:
        torch.nn.init.normal_(head.weight, std=0.5)
    n_cond_frames, rows_per_frame = 2, 4
    batch, target_idx, n_frames = make_batch(
        p, num_cond_frames=n_cond_frames, rows_per_frame=rows_per_frame)
    pose = torch.randn(1, n_frames, POSE_DIM)

    seen = []
    handle = model.blocks[0].register_forward_pre_hook(
        lambda _m, args: seen.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            enc.set_gain(1.0)
            model(**batch, cam_pose=pose, target_video_indices=target_idx)
            enc.set_gain(4.0)
            model(**batch, cam_pose=pose, target_video_indices=target_idx)
    finally:
        handle.remove()

    target = set(target_idx.tolist())
    moved = set(((seen[1] - seen[0])[0].abs().sum(-1) > 0)
                .nonzero().flatten().tolist())
    assert moved == target, \
        f"gain moved {sorted(moved - target)} outside the target rows"


def test_gain_is_not_in_the_state_dict():
    """A registered buffer would join ``state_dict`` — and the R6c-EA generator
    loads the banked encoder with an exact key check that raises on any missing
    key, so a gain buffer would make every banked checkpoint unloadable. The
    knob is a run-time scale over frozen weights and stays out of the artifact.
    """
    p = pruned_params()
    model = MiniMaxH3Transformer(p)
    enc = model.attach_camera_encoder(bottleneck=16, gain=4.0)
    assert enc.gain == 4.0
    keys = list(enc.state_dict().keys())
    assert not any("gain" in k for k in keys), keys
    fresh = MiniMaxH3CameraEncoder(p.hidden_size, p.num_layers, bottleneck=16)
    missing, unexpected = fresh.load_state_dict(enc.state_dict(), strict=False)
    assert not missing and not unexpected, (missing, unexpected)


def test_a_nonsense_gain_is_refused_rather_than_silently_scaled():
    """A negative gain INVERTS the command instead of scaling it — a different
    experiment with a different reading. Refused here so it cannot arrive as a
    result that looks like "amplification made it worse"."""
    p = pruned_params()
    model = MiniMaxH3Transformer(p)
    enc = model.attach_camera_encoder(bottleneck=16)
    for bad in (-1.0, float("nan"), float("inf")):
        try:
            enc.set_gain(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted gain {bad}")
    assert enc.gain == 1.0, "a refused gain must leave the knob where it was"


def test_attach_and_the_module_default_agree_on_one_point_zero():
    p = pruned_params()
    model = MiniMaxH3Transformer(p)
    assert model.attach_camera_encoder(bottleneck=16).gain == 1.0
    assert MiniMaxH3CameraEncoder(64, 2, bottleneck=16).gain == 1.0


if __name__ == "__main__":
    print("R6b camera encoder")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            check(name, fn)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)
