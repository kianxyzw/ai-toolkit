"""Packed-sequence geometry for MiniMax-H3.

One transformer forward runs over a single packed 1-D sequence:

    [ text (L) | keyframe conditions (C) | target audio (A) | target video (V) ]

for t2va/fl2va, and for ref2va:

    [ text (L) | reference blocks (R) | target audio (A) | target video (V) ]

where each reference block contributes rows on its OWN spatial grid (references
keep their native canvases) and advances the shared rotary clock — an image by
1.0 units, audio by one unit per latent, a video by its full temporal span —
so the target's clock starts after the last reference.

This module owns everything needed to place a row in that sequence and give it
its (t, h, w) rotary coordinate, plus the sigma-shift math that couples the
video (shift 12) and audio (shift 3) flow schedules.

Rotary coordinates are built in float64 and with numpy's ``linspace`` because
video and audio share one 40-units-per-second rotary clock (video advances
5/3 units per pixel frame at 24 fps, audio one unit per latent at 40/s) and
that shared clock is the released checkpoint's audio/video alignment.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Per-row modality tags — these index the transformer's AdaLN table, so the
# values are a checkpoint contract.
VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2
PAD_TAG = -1

FPS = 24
SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
CANVAS_MULTIPLE = 32

# video VAE: 17 pixel frames per chunk -> 5 latent frames, 3 trailing latents
# dropped overall, so 17n+5 pixel frames <-> 5n+2 latent frames
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5

AUDIO_LATENTS_PER_SECOND = 40
AUDIO_CHANNELS = 2
AUDIO_SAMPLE_RATE = 32000

# released flow shifts (exponential): video 12, audio 3
VIDEO_SIGMA_SHIFT = 12.0
AUDIO_SIGMA_SHIFT = 3.0

# keyframe conditioning rows are noised to t = 0.999 and pinned there; the
# posterior sample of the keyframe VAE encode uses a fixed seed of 42
KEYFRAME_NOISE_AUG_T = 0.999
KEYFRAME_ENCODE_SEED = 42
# reference video/image rows pin at the same 0.999; reference audio rows stay
# clean (t = 1.0)
AUDIO_COND_NOISE_AUG_T = 1.0

# rotary-time constants: one latent frame spans 5/3 * frames_per_latent units,
# the (1, 4, 4, 4, 4) pattern mirroring the VAE's 17 -> 5 frame grouping
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


# ---------------------------------------------------------------------------
# Frame / canvas arithmetic
# ---------------------------------------------------------------------------


def align_num_frames(num_frames: int) -> int:
    """Snap a frame count UP to the next 17n+5 the video VAE can encode."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    while num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


def align_num_frames_down(num_frames: int) -> int:
    """Snap a frame count DOWN to the previous 17n+5 (minimum 5)."""
    num_frames = max(num_frames, LATENTS_PER_CHUNK)
    while num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        num_frames -= 1
    return num_frames


def video_latent_num_frames(num_frames: int) -> int:
    """17n+5 pixel frames -> 5n+2 latent frames."""
    if num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        raise ValueError(f"num_frames must be of the form 17n+5, got {num_frames}")
    return (num_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int) -> int:
    """Audio latents covering `num_frames` video frames at 24 fps / 40 Hz."""
    return int(round(num_frames / FPS * AUDIO_LATENTS_PER_SECOND))


def resolve_canvas_size(aspect_width: float, aspect_height: float) -> Tuple[int, int]:
    """Aspect ratio -> (height, width): short edge 768, area capped at 768*1344,
    both axes rounded to the nearest multiple of 32."""
    ratio = aspect_width / aspect_height
    if ratio >= 1.0:
        width, height = SHORT_EDGE * ratio, float(SHORT_EDGE)
    else:
        width, height = float(SHORT_EDGE), SHORT_EDGE / ratio
    area = width * height
    if area > MAX_PIXELS:
        scale = (MAX_PIXELS / area) ** 0.5
        width, height = width * scale, height * scale
    m = CANVAS_MULTIPLE
    return max(m, round(height / m) * m), max(m, round(width / m) * m)


def prepare_keyframe_image(
    image: Image.Image, height: int, width: int, stretch: bool = True
):
    """Put a keyframe onto the target canvas: the geometry anchor is stretched,
    a follower keyframe is cover-cropped."""
    if image.size == (width, height):
        return image
    if stretch:
        return image.resize((width, height), Image.Resampling.LANCZOS)
    scale = max(width / image.size[0], height / image.size[1])
    resized_size = (
        max(width, round(image.size[0] * scale)),
        max(height, round(image.size[1] * scale)),
    )
    left = max(0, (resized_size[0] - width) // 2)
    top = max(0, (resized_size[1] - height) // 2)
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    return resized.crop((left, top, left + width, top + height))


# ---------------------------------------------------------------------------
# Row packing
# ---------------------------------------------------------------------------


def patchify_video_latents(latents: torch.Tensor, patch_size=(1, 2, 2)) -> torch.Tensor:
    """(B, C, T, H, W) -> (B, N, C * prod(patch)) rows, frame-major then
    row-major, feature order [c, pt, ph, pw]."""
    pt, ph, pw = patch_size
    b, c, t, h, w = latents.shape
    latents = latents.reshape(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(b, -1, c * pt * ph * pw).contiguous()


def unpatchify_video_tokens(
    rows: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    channels: int = 24,
    patch_size=(1, 2, 2),
) -> torch.Tensor:
    """(B, N, C * prod(patch)) -> (B, C, T, H, W). Inverse of patchify."""
    pt, ph, pw = patch_size
    b = rows.shape[0]
    rows = rows.reshape(
        b,
        num_latent_frames // pt,
        latent_height // ph,
        latent_width // pw,
        channels,
        pt,
        ph,
        pw,
    )
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(
        b, channels, num_latent_frames, latent_height, latent_width
    ).contiguous()


def pack_audio_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, 2, C, T) stereo audio latents -> (B, 2*T, C) channel-major rows
    (all T frames of channel 0, then channel 1)."""
    return (
        latents.permute(0, 1, 3, 2)
        .reshape(latents.shape[0], -1, latents.shape[2])
        .contiguous()
    )


def unpack_audio_tokens(rows: torch.Tensor, num_audio_latents: int) -> torch.Tensor:
    """(B, 2*T, C) channel-major rows -> (B, 2, C, T)."""
    b, _, c = rows.shape
    rows = rows.reshape(b, AUDIO_CHANNELS, num_audio_latents, c)
    return rows.permute(0, 1, 3, 2).contiguous()


# ---------------------------------------------------------------------------
# Rotary grids (float64, numpy linspace — the released grid must reproduce)
# ---------------------------------------------------------------------------


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    # numpy linspace(endpoint=False) is start + arange(n) * (stop-start)/n,
    # which is not bit-identical to torch.linspace
    grid = (
        np.linspace(left, left + ratio, dim // patch, endpoint=False)
        * _ROPE_SPATIAL_SCALE
    )
    return torch.from_numpy(grid).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [
            _ROPE_FRAME_RESCALE
            * _ROPE_FRAMES_PER_LATENT[i % len(_ROPE_FRAMES_PER_LATENT)]
            for i in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat(
        [torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)]
    )


def _temporal_position_span(num_latent_frames: int) -> float:
    # numpy pairwise sum on purpose: the reference computes the "last" keyframe
    # anchor this way and the summation orders differ in the last ulp
    spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
    for i in range(len(_ROPE_FRAMES_PER_LATENT)):
        spans[i :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[i]
    return float(spans.sum())


def _video_t_span_sum(num_latent_frames: int) -> float:
    # plain left-to-right sum: the reference advances the ref2va rotary cursor
    # past a reference video this way (NOT the pairwise sum above)
    return float(
        sum(
            _ROPE_FRAME_RESCALE
            * _ROPE_FRAMES_PER_LATENT[i % len(_ROPE_FRAMES_PER_LATENT)]
            for i in range(num_latent_frames)
        )
    )


def _frame_position_grid(latent_height: int, latent_width: int, ph: int, pw: int):
    """(rows_per_frame, 2) area-normalized (h, w) coordinates of one latent
    frame's patch rows, plus the width axis grid (for audio-row pinning)."""
    sqrt_area = math.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, ph, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, pw, sqrt_area)
    frame_grid = torch.stack(
        [g.reshape(-1) for g in torch.meshgrid(height_grid, width_grid, indexing="ij")],
        dim=-1,
    )
    return frame_grid, width_grid


def _audio_position_grid(
    origin: float, num_latents: int, w_low: float, w_high: float
) -> torch.Tensor:
    """Channel-major stereo audio rows: one rotary unit per latent, no height
    coordinate, width pinned per channel to the given grid extremes."""
    g = torch.zeros(num_latents * AUDIO_CHANNELS, 3, dtype=torch.float64)
    g[:, 0] = (origin + torch.arange(num_latents, dtype=torch.float64)).repeat(
        AUDIO_CHANNELS
    )
    g[:num_latents, 2] = w_low
    g[num_latents:, 2] = w_high
    return g


# ---------------------------------------------------------------------------
# Sequence layout
# ---------------------------------------------------------------------------


@dataclass
class PackedLayout:
    """Structural description of one packed sequence (one batch item)."""

    sequence_length: int
    position_ids: torch.Tensor  # (S, 3) float64
    token_tags: torch.Tensor  # (S,) long
    video_indices: torch.Tensor  # condition rows first, then target rows
    audio_indices: torch.Tensor  # condition rows first, then target rows
    text_indices: torch.Tensor
    num_condition_video_rows: int
    num_condition_audio_rows: int = 0


def build_packed_sequence(
    text_token_tags: torch.Tensor,  # (L,) long: 1 text, 0 for vision-block rows
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size=(1, 2, 2),
    keyframe_anchors: Tuple[str, ...] = (),
    reference_blocks: Tuple[dict, ...] = (),
) -> PackedLayout:
    """Build the packed layout used by t2va/fl2va (``keyframe_anchors``) and
    ref2va (``reference_blocks``):

        [text | keyframe conditions | reference blocks | target audio | target video]

    ``reference_blocks`` mirrors the released ref2va conditioning, one dict per
    reference in presentation order:

      - ``{"kind": "image", "latent_h": h, "latent_w": w}``
      - ``{"kind": "video" | "video_audio", "latent_t": t, "latent_h": h,
         "latent_w": w, "ref_audio_t": a}`` — a ``video_audio`` block packs its
        soundtrack's audio rows immediately BEFORE its video rows, both
        starting at the block's cursor origin
      - ``{"kind": "audio", "ref_audio_t": a}``

    Latent dims are VAE-latent (pixels / 16); every reference keeps its own
    canvas. Video/image reference rows count into ``num_condition_video_rows``
    (they sit first in ``video_indices``, matching the row order the caller
    must use for ``hidden_states``); reference audio rows likewise into
    ``num_condition_audio_rows``.
    """
    _, ph, pw = patch_size
    if keyframe_anchors and reference_blocks:
        raise ValueError(
            "keyframe_anchors (fl2va) and reference_blocks (ref2va) cannot be "
            "combined; the released partitions never mix them"
        )
    num_text = int(text_token_tags.shape[0])
    frame_grid, width_grid = _frame_position_grid(latent_height, latent_width, ph, pw)
    rows_per_frame = frame_grid.shape[0]
    target_audio_w = (float(width_grid[0]), float(width_grid[-1]))

    # media segments after the text span, in sequence order:
    # (positions (N, 3) f64, "video" | "audio", is_condition)
    segments = []

    for anchor in keyframe_anchors:
        if anchor == "first":
            anchor_time = float(num_text)
        elif anchor == "last":
            anchor_time = (
                float(num_text)
                + _temporal_position_span(num_latent_frames)
                - _ROPE_FRAME_RESCALE
            )
        else:
            raise ValueError(
                f"keyframe anchor must be 'first' or 'last', got {anchor!r}"
            )
        g = torch.empty(rows_per_frame, 3, dtype=torch.float64)
        g[:, 0] = anchor_time
        g[:, 1:] = frame_grid
        segments.append((g, "video", True))

    # references advance a running rotary cursor; the target clock starts
    # after the last reference (with no references the cursor stays at
    # num_text and the layout reduces to the t2va/fl2va one)
    cursor = float(num_text)
    for blk in reference_blocks:
        kind = blk["kind"]
        if kind == "image":
            r_frame, _ = _frame_position_grid(
                int(blk["latent_h"]), int(blk["latent_w"]), ph, pw
            )
            g = torch.empty(r_frame.shape[0], 3, dtype=torch.float64)
            g[:, 0] = cursor
            g[:, 1:] = r_frame
            segments.append((g, "video", True))
            cursor += 1.0
        elif kind == "audio":
            rt = int(blk["ref_audio_t"])
            if rt > 0:
                # standalone audio rides on the TARGET's width extremes
                segments.append(
                    (_audio_position_grid(cursor, rt, *target_audio_w), "audio", True)
                )
            cursor += float(rt)
        elif kind in ("video", "video_audio"):
            rt = int(blk.get("ref_audio_t", 0) or 0)
            vt = int(blk["latent_t"])
            r_frame, r_width_grid = _frame_position_grid(
                int(blk["latent_h"]), int(blk["latent_w"]), ph, pw
            )
            if rt > 0:
                # a soundtrack pins to its OWN video's width extremes
                segments.append(
                    (
                        _audio_position_grid(
                            cursor, rt, float(r_width_grid[0]), float(r_width_grid[-1])
                        ),
                        "audio",
                        True,
                    )
                )
            g = torch.empty(vt, r_frame.shape[0], 3, dtype=torch.float64)
            g[:, :, 0] = _temporal_position_grid(vt, cursor)[:, None]
            g[:, :, 1:] = r_frame[None]
            segments.append((g.reshape(-1, 3), "video", True))
            cursor += max(float(rt), _video_t_span_sum(vt))
        else:
            raise ValueError(f"unknown reference block kind {kind!r}")

    # target audio then target video, always the last two segments
    segments.append(
        (
            _audio_position_grid(cursor, num_audio_latents, *target_audio_w),
            "audio",
            False,
        )
    )
    video_pos = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_pos[:, :, 0] = _temporal_position_grid(num_latent_frames, cursor)[:, None]
    video_pos[:, :, 1:] = frame_grid[None]
    segments.append((video_pos.reshape(-1, 3), "video", False))

    # text rows sit on the time axis at their row index; the media clock
    # continues from there, so prompt length shifts the whole media clock
    text_pos = torch.zeros(num_text, 3, dtype=torch.float64)
    text_pos[:, 0] = torch.arange(num_text, dtype=torch.float64)

    position_parts = [text_pos]
    cond_video_idx, target_video_idx = [], []
    cond_audio_idx, target_audio_idx = [], []
    row = num_text
    for g, kind, is_condition in segments:
        n = g.shape[0]
        idx = torch.arange(row, row + n)
        if kind == "video":
            (cond_video_idx if is_condition else target_video_idx).append(idx)
        else:
            (cond_audio_idx if is_condition else target_audio_idx).append(idx)
        position_parts.append(g)
        row += n
    seq_len = row

    position_ids = torch.cat(position_parts)
    video_indices = torch.cat(cond_video_idx + target_video_idx)
    audio_indices = torch.cat(cond_audio_idx + target_audio_idx)
    text_indices = torch.arange(num_text)
    num_cond_video = sum(int(i.shape[0]) for i in cond_video_idx)
    num_cond_audio = sum(int(i.shape[0]) for i in cond_audio_idx)

    token_tags = torch.empty(seq_len, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = AUDIO_TAG
    token_tags[video_indices] = VIDEO_TAG

    return PackedLayout(
        sequence_length=seq_len,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_cond_video,
        num_condition_audio_rows=num_cond_audio,
    )


def build_row_timesteps(
    layout: PackedLayout,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: Optional[float] = None,
    condition_audio_timestep: Optional[float] = None,
) -> torch.Tensor:
    """Per-row timestep values (S,) float32. Text rows inherit the video
    timestep; condition video/audio rows stay pinned at their noise-aug
    levels."""
    if condition_video_timestep is None:
        condition_video_timestep = max(video_timestep, KEYFRAME_NOISE_AUG_T)
    if condition_audio_timestep is None:
        condition_audio_timestep = max(audio_timestep, AUDIO_COND_NOISE_AUG_T)
    row_t = torch.full(
        (layout.sequence_length,), float(video_timestep), dtype=torch.float32
    )
    row_t[layout.video_indices[: layout.num_condition_video_rows]] = float(
        condition_video_timestep
    )
    row_t[layout.audio_indices] = float(audio_timestep)
    row_t[layout.audio_indices[: layout.num_condition_audio_rows]] = float(
        condition_audio_timestep
    )
    return row_t


def pad_layouts_to_batch(layouts: List[PackedLayout]):
    """Stack per-item layouts that share the same media geometry but may have
    different text lengths into batched transformer inputs.

    Items are right-padded in the TEXT segment to the batch's max text length
    (pad rows tagged -1, masked out of attention as keys, positions zero).
    Returns (position_ids (B, S, 3) f64, token_tags (B, S) long,
    video_indices, audio_indices, text_indices, pad_counts) where the index
    tensors describe the shared padded layout: [text_max | cond | audio | video].
    """
    max_text = max(int(l.text_indices.shape[0]) for l in layouts)
    ref = layouts[0]
    ref_text = int(ref.text_indices.shape[0])
    media_len = ref.sequence_length - ref_text
    for l in layouts:
        lt = int(l.text_indices.shape[0])
        if (
            l.sequence_length - lt != media_len
            or l.num_condition_video_rows != ref.num_condition_video_rows
            or l.num_condition_audio_rows != ref.num_condition_audio_rows
            or not torch.equal(l.video_indices - lt, ref.video_indices - ref_text)
            or not torch.equal(l.audio_indices - lt, ref.audio_indices - ref_text)
        ):
            raise ValueError(
                "all layouts in a batch must share the same media geometry "
                "(including reference block shapes and order)"
            )
    seq_len = max_text + media_len

    b = len(layouts)
    position_ids = torch.zeros(b, seq_len, 3, dtype=torch.float64)
    token_tags = torch.full((b, seq_len), PAD_TAG, dtype=torch.long)
    pad_counts = []
    for i, l in enumerate(layouts):
        lt = int(l.text_indices.shape[0])
        position_ids[i, :lt] = l.position_ids[:lt]
        position_ids[i, max_text:] = l.position_ids[lt:]
        token_tags[i, :lt] = l.token_tags[:lt]
        token_tags[i, max_text:] = l.token_tags[lt:]
        pad_counts.append(max_text - lt)

    offset = max_text - int(ref.text_indices.shape[0])
    video_indices = ref.video_indices + offset
    audio_indices = ref.audio_indices + offset
    text_indices = torch.arange(max_text)
    return (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        pad_counts,
    )


# ---------------------------------------------------------------------------
# Sigma-shift math (video shift 12, audio shift 3, exponential)
# ---------------------------------------------------------------------------


def shift_sigma(sigma, shift: float):
    """Exponential timeshift: shift * sigma / (1 + (shift - 1) * sigma)."""
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def remap_sigma(
    sigma, from_shift: float = VIDEO_SIGMA_SHIFT, to_shift: float = AUDIO_SIGMA_SHIFT
):
    """Map a sigma on the `from_shift` schedule onto the `to_shift` schedule
    at the same underlying schedule position (the video/audio coupling)."""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return shift_sigma(base, to_shift)


def build_sigma_schedule(
    num_inference_steps: int, shift: float = VIDEO_SIGMA_SHIFT
) -> torch.Tensor:
    """The released sampling grid: linspace(1, 0, steps + 1) through the
    exponential shift, consecutive duplicates collapsed — `steps` yields
    `steps` model evaluations (the released repo counts the terminal 0 in
    `steps`; we don't, so sample_steps means model evals)."""
    base = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float32)
    sigmas = shift_sigma(base, shift)
    return torch.unique_consecutive(sigmas)
