"""Camera-trajectory encoder for MiniMax-H3 — R6b, the Plan-B conditioning arm.

R6a conditions on the camera by *rendering* the trajectory into a reference
video (the Plücker raymap) and letting H3's native reference path carry it.
This module is the other arm: feed the trajectory in as numbers, through a
small trainable side-module added to the target video's token rows.

Ported from ReCamMaster (KwaiVGI, arXiv 2503.11647 §4.3; `train_recammaster.py`)
with three deliberate deviations, each recorded because copying them blindly
would be wrong here:

**1. A bottleneck, because H3 is not Wan-1.3B.** ReCamMaster puts a full
``nn.Linear(12, dim)`` plus a ``dim x dim`` projector in *every* block. At
Wan's 1536-dim / 30 blocks that is ~70M parameters; at H3's **5376-dim / 50
blocks** the projector alone would be **1.45B** — larger than most of the
things it is meant to condition, and far outside the "small side-module"
this arm is supposed to test. Instead: one shared trunk (12 -> bottleneck,
SiLU, bottleneck -> bottleneck) and a per-block zero-init head
(bottleneck -> hidden). At the default bottleneck of 256 that is ~69M
parameters — the per-block specialization ReCamMaster's ablation cares about,
at 5% of the cost.

**2. Injection at the block input, because H3 has no separate spatial
attention.** ReCamMaster adds the embedding to the *spatial*-attention output,
then feeds the 3D-attention layers. H3 runs one fused attention per block, so
there is no equivalent seam; the closest faithful analogue is the block input,
which is positionally identical to "after the previous block".

**3. NO ``/100`` on the translation.** ReCamMaster's dataloader does
``c2w[:3, 3] /= 100`` to convert UE's centimetres to metres. **Our pipeline
already did that** — ``harness/data/extrinsics.py`` converts at parse time and
documents that everything it returns is in metres. Applying it again would
divide every translation by 100 a second time, leaving ~1 cm of camera motion
and a conditioning channel that carries nothing. That failure is silent and
looks exactly like "the architecture doesn't bind", which is the R1 bug
wearing a different hat — so :func:`pose_vectors_from_c2w` asserts the scale
instead of trusting it.

What this module must never touch, per the R6b brief and the distilled-base
constraint: **adaln** (the modulation path the pruned checkpoint drives from a
lookup table) and **audio rows**. It writes to target video rows only.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

POSE_DIM = 12  # 3x4 camera-to-world, row-major — ReCamMaster's layout

# Calibrated against the real distribution, not guessed — this project's own
# lesson is that an uncalibrated pass/fail constant is worse than no check
# (the degeneracy threshold that sat inside the failure band).
#
# Measured over manifest_v1's arc pool: path_length_m spans 0.62 .. 9.01 m.
# Divide that range by 100 and it becomes 0.0062 .. 0.090 m. The two ranges do
# not overlap, and 0.1 m sits in the gap: every real trajectory passes, every
# double-converted one fails. A trajectory genuinely under 10 cm carries no
# usable camera signal anyway, so rejecting it is correct on its own terms.
MIN_PLAUSIBLE_TRANSLATION_M = 0.1
MAX_PLAUSIBLE_TRANSLATION_M = 1000.0


def pose_vectors_from_c2w(c2w: torch.Tensor, check_scale: bool = True) -> torch.Tensor:
    """(F, 4, 4) camera-to-world **in metres** -> (F, 12) pose vectors.

    Takes the top 3x4 block (rotation + translation) row-major, matching
    ReCamMaster's ``rearrange(pose, 'b c d -> b (c d)')``.

    ``check_scale`` asserts the translation looks like metres. It is on by
    default and it exists because the alternative — a channel of ~1 cm
    translations — trains without error and reads as an architecture result.
    Pass False only for synthetic tensors in tests.
    """
    if c2w.ndim != 3 or c2w.shape[-2:] != (4, 4):
        raise ValueError(f"expected (F, 4, 4) camera-to-world, got {tuple(c2w.shape)}")
    if check_scale:
        span = float(c2w[:, :3, 3].max() - c2w[:, :3, 3].min())
        if span > 0 and not (MIN_PLAUSIBLE_TRANSLATION_M <= span
                             <= MAX_PLAUSIBLE_TRANSLATION_M):
            raise ValueError(
                f"translation span {span:g} is not metres — a span this small "
                "usually means centimetres were converted twice (ReCamMaster's "
                "`c2w[:3,3] /= 100` re-applied on top of extrinsics.py, which "
                "already converts). Fix the caller; do not relax this check."
            )
    return c2w[:, :3, :].reshape(c2w.shape[0], POSE_DIM)


def latent_frame_indices(num_video_frames: int, num_latent_frames: int) -> list[int]:
    """Which VIDEO frame represents each LATENT frame.

    The visual VAE is causal and 4x temporal: latent 0 corresponds to video
    frame 0 alone, and latent k>0 to the group of four frames ending at
    ``4k``. 73 video frames therefore give 19 latents ((73-1)/4 + 1).

    A representative frame is taken rather than an average because averaging
    rotation matrices is not a rotation — the mean of two 90-degree-apart
    orientations is not a valid pose, and it would silently shrink the
    conditioning signal toward the identity. The group's LAST frame is the
    temporally aligned choice under a causal VAE.

    Falls back to a uniform spread when the counts do not fit the 4x causal
    grid, so a different frame count degrades to something sane rather than
    raising deep inside a training step.
    """
    if num_latent_frames <= 0:
        raise ValueError(f"num_latent_frames must be positive, got {num_latent_frames}")
    if num_video_frames <= 0:
        raise ValueError(f"num_video_frames must be positive, got {num_video_frames}")
    if num_video_frames == (num_latent_frames - 1) * 4 + 1:
        return [0] + [4 * k for k in range(1, num_latent_frames)]
    if num_latent_frames == 1:
        return [0]
    step = (num_video_frames - 1) / (num_latent_frames - 1)
    return [min(num_video_frames - 1, int(round(k * step)))
            for k in range(num_latent_frames)]


def poses_to_latent_vectors(c2w: torch.Tensor, num_latent_frames: int,
                            check_scale: bool = True) -> torch.Tensor:
    """(F, 4, 4) c2w in metres -> (num_latent_frames, 12) pose vectors.

    The sidecar stores per-VIDEO-frame poses so it is independent of whatever
    temporal compression the VAE applies; the reduction to latent resolution
    happens here, once, where the latent frame count is known.
    """
    idx = latent_frame_indices(int(c2w.shape[0]), num_latent_frames)
    return pose_vectors_from_c2w(c2w[idx], check_scale=check_scale)


class MiniMaxH3CameraEncoder(nn.Module):
    """Trajectory -> per-block additive embedding for the target video rows.

    Zero-init on every per-block head makes the whole module an **exact**
    no-op at step 0: the added term is ``0 * anything``, so a model with the
    encoder attached produces bit-identical output to one without it. That is
    non-negotiable on a CFG-distilled base — a randomly-initialized addition
    into a distilled model's residual stream destroys it before training can
    recover, which is the same reason ControlNet zero-inits its connections.
    ReCamMaster zero-inits the encoder and identity-inits its projector; the
    composition is a no-op either way, and zero-init on the *output* is the
    formulation that stays a no-op if the trunk is ever pre-trained.
    """

    def __init__(self, hidden_size: int, num_layers: int, bottleneck: int = 256,
                 pose_dim: int = POSE_DIM):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bottleneck = bottleneck
        self.pose_dim = pose_dim
        self.trunk = nn.Sequential(
            nn.Linear(pose_dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, bottleneck),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(bottleneck, hidden_size) for _ in range(num_layers)]
        )
        self.zero_init()

    def zero_init(self):
        """Zero every per-block head. Idempotent; called from __init__."""
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def is_zero_initialized(self) -> bool:
        return all(bool((h.weight == 0).all() and (h.bias == 0).all())
                   for h in self.heads)

    def trunk_features(self, pose: torch.Tensor) -> torch.Tensor:
        """(B, T, pose_dim) -> (B, T, bottleneck). Computed once per forward."""
        if pose.shape[-1] != self.pose_dim:
            raise ValueError(
                f"pose vectors must be {self.pose_dim}-d, got {pose.shape[-1]}")
        return self.trunk(pose.to(self.trunk[0].weight.dtype))

    def block_delta(self, layer: int, features: torch.Tensor,
                    rows_per_frame: int) -> torch.Tensor:
        """(B, T, bottleneck) -> (B, T * rows_per_frame, hidden) for one block.

        Each latent frame's embedding repeats across that frame's spatial token
        rows, which is ReCamMaster's "expanded via repetition, then added
        element-wise". ``repeat_interleave`` — not ``repeat`` — because the
        target rows are ordered frame-major (all rows of frame 0, then frame
        1); tiling instead would pair every frame with the wrong rows and
        still produce a plausible-looking tensor of the right shape.
        """
        return self.heads[layer](features).repeat_interleave(rows_per_frame, dim=1)

    def forward(self, pose: torch.Tensor, layer: int,
                rows_per_frame: int) -> torch.Tensor:
        return self.block_delta(layer, self.trunk_features(pose), rows_per_frame)


def add_camera_embedding(
    x: torch.Tensor,
    delta: torch.Tensor,
    target_video_indices: torch.Tensor,
) -> torch.Tensor:
    """Add ``delta`` to ``x`` at the target video rows, and nowhere else.

    ``target_video_indices`` is ``layout.video_indices[num_condition_video_rows:]``
    — the packing module's own convention, where reference rows come first.
    Slicing it wrong is the failure this function is factored out to make
    testable: writing into the reference rows would corrupt the conditioning
    the model is supposed to read, and writing into audio or text rows would
    be invisible in the loss until much later.
    """
    if delta.shape[1] != target_video_indices.shape[0]:
        raise ValueError(
            f"camera embedding covers {delta.shape[1]} rows but there are "
            f"{target_video_indices.shape[0]} target video rows — the pose "
            "sequence and the latent frame count disagree"
        )
    return x.index_add(1, target_video_indices, delta.to(x.dtype))


def encoder_state_dict_prefix() -> str:
    """Where these weights live in a checkpoint.

    Deliberately NOT under ``diffusion_model.`` — that prefix is what ComfyUI's
    LoRA loader scans, and the R6b artifact must stay separable from the LoRA
    (item 5b measured 416 tensors, all ``diffusion_model.``-prefixed, and that
    count is a regression check).
    """
    return "camera_encoder."
