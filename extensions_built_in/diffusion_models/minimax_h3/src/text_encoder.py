"""Qwen3-VL conditioning for MiniMax-H3.

MiniMax-H3 conditions on the **unnormalized** ``hidden_states[50]`` of its
Qwen3-VL-32B conditioner (``hidden_states[0]`` is the embedding output, so
this is the output of decoder layer 49, before the final norm). The LM head
and layers 50..63 are never used, which lets the loader truncate the stack.

The presentation is raw tokens — no chat template, no special tokens:

  - t2va: the verbatim prompt.
  - fl2va: per keyframe, a ``"<Picture i>: "`` label plus a vision block
    (``<|vision_start|>`` + one ``<|image_pad|>`` per merged vision patch +
    ``<|vision_end|>``), then the verbatim prompt. Vision-block rows are
    tagged as *video* (0) rather than text (1) — the transformer's AdaLN
    modality selection keys off these tags.
  - ref2va: per reference, in presentation order with 1-based ordinals per
    type, then the verbatim prompt:

      image -> ``"<Picture i>: "`` + vision block
      audio -> ``"<Audio j>: "``            (audio never enters Qwen)
      video -> ``"<Video k>: "`` then, per 2-frame temporal block of the
               2 fps subsample, ``"<T.T seconds>"`` + a vision block whose
               temporal patch holds the TWO frames (a still image repeats
               one frame instead)

    A vision block's rows — including the flanking ``<|vision_start|>`` /
    ``<|vision_end|>`` tokens — are tagged video (0); labels and timestamps
    are text (1).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .packing import TEXT_TAG, VIDEO_TAG

TEXT_ENCODER_LAYER = 50

# resize bounds of the released video-block preprocessing (patch grid area in
# pixels); the processor's own attributes override when present
_VIDEO_BLOCK_MIN_PIXELS = 3136
_VIDEO_BLOCK_MAX_PIXELS = 12845056


def sample_ref_video_2fps(frames: torch.Tensor, presentation_fps: float = 24.0):
    """(T, C, H, W) frames declared at ``presentation_fps`` -> (frames_2fps,
    timestamps) the way the released pipeline shows reference videos to Qwen:
    every ``fps // 2``-th frame, timestamped at 0.5 s steps."""
    step = max(1, int(presentation_fps // 2))
    indices = list(range(0, int(frames.shape[0]), step))
    timestamps = [i / 2.0 for i in range(len(indices))]
    return frames[indices], timestamps


def process_ref_video_block(frames: torch.Tensor, processor):
    """One 2-frame temporal block -> (flatten_patches, grid_thw (1, 3)).

    ``frames`` is (2, C, H, W) float in [0, 1]. Mirrors the released
    preprocessing: resize to the patch-grid multiple (area-clamped), normalize,
    then pack with the two frames filling the temporal patch and ``grid_t = 1``
    — the same patch layout the HF image processor emits for stills, except a
    still repeats one frame where this uses two distinct ones.
    """
    ip = processor.image_processor
    patch_size = int(getattr(ip, "patch_size", 16))
    merge_size = int(getattr(ip, "merge_size", 2))
    temporal = int(getattr(ip, "temporal_patch_size", 2))
    min_pixels = int(getattr(ip, "min_pixels", None) or _VIDEO_BLOCK_MIN_PIXELS)
    max_pixels = int(getattr(ip, "max_pixels", None) or _VIDEO_BLOCK_MAX_PIXELS)
    if frames.shape[0] != temporal:
        raise ValueError(
            f"video block needs {temporal} frames, got {int(frames.shape[0])}"
        )

    imgs = frames.to(torch.float32)
    height, width = int(imgs.shape[2]), int(imgs.shape[3])
    factor = patch_size * merge_size
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        import math

        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        import math

        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    imgs = F.interpolate(imgs, size=(h_bar, w_bar), mode="bilinear", align_corners=False)
    mean = torch.tensor(ip.image_mean, dtype=imgs.dtype).view(1, 3, 1, 1)
    std = torch.tensor(ip.image_std, dtype=imgs.dtype).view(1, 3, 1, 1)
    imgs = (imgs - mean) / std

    grid_h = h_bar // patch_size
    grid_w = w_bar // patch_size
    patches = imgs.reshape(
        1,
        temporal,
        3,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten = patches.reshape(
        grid_h * grid_w, 3 * temporal * patch_size * patch_size
    )
    grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
    return flatten, grid_thw


def build_minimax_h3_presentation(
    tokenizer,
    processor,
    prompt: str,
    keyframes: Optional[List] = None,  # PIL images already on the target canvas
    ref_items: Optional[List[dict]] = None,  # see encode_minimax_h3_prompt
    max_length: Optional[int] = None,  # cap on PROMPT tokens (vision blocks are never cut)
) -> Tuple[List[int], List[int], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Tokenize one prompt (+ keyframes or references) WITHOUT running the
    encoder. Returns (token_ids, token_tags, pixel_values, image_grid_thw) —
    split out from the encode so caching layers and tests can build the
    presentation cheaply."""
    if keyframes and ref_items:
        raise ValueError(
            "keyframes (fl2va) and ref_items (ref2va) cannot be combined"
        )

    pixel_parts: List[torch.Tensor] = []
    grid_parts: List[torch.Tensor] = []
    token_ids: List[int] = []
    token_tags: List[int] = []

    vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    merge = processor.image_processor.merge_size**2

    def add_text(s: str):
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        token_ids.extend(ids)
        token_tags.extend([TEXT_TAG] * len(ids))

    def add_vision_block(flatten: torch.Tensor, grid_row: torch.Tensor):
        pixel_parts.append(flatten)
        grid_parts.append(grid_row.reshape(1, 3))
        num_tokens = int(grid_row.prod()) // merge
        ids = [vision_start] + [image_pad] * num_tokens + [vision_end]
        token_ids.extend(ids)
        token_tags.extend([VIDEO_TAG] * len(ids))

    if keyframes:
        vision = processor.image_processor(images=keyframes, return_tensors="pt")
        offsets = [0]
        for i in range(len(keyframes)):
            offsets.append(offsets[-1] + int(vision["image_grid_thw"][i].prod()))
        for i in range(len(keyframes)):
            add_text(f"<Picture {i + 1}>: ")
            add_vision_block(
                vision["pixel_values"][offsets[i] : offsets[i + 1]],
                vision["image_grid_thw"][i],
            )
    elif ref_items:
        counters = {"image": 0, "audio": 0, "video": 0}
        for item in ref_items:
            kind = item["type"]
            counters[kind] += 1
            if kind == "image":
                vision = processor.image_processor(
                    images=[item["data"]], return_tensors="pt"
                )
                add_text(f"<Picture {counters['image']}>: ")
                add_vision_block(vision["pixel_values"], vision["image_grid_thw"][0])
            elif kind == "audio":
                # the label alone: audio never enters Qwen
                add_text(f"<Audio {counters['audio']}>: ")
            elif kind == "video":
                frames = item["data"]  # (T, C, H, W) in [0, 1], 2 fps subsample
                timestamps = item.get("timestamps")
                if timestamps is None:
                    timestamps = [i / 2.0 for i in range(int(frames.shape[0]))]
                if frames.shape[0] % 2 == 1:  # repeat-pad to the temporal patch
                    frames = torch.cat([frames, frames[-1:]], dim=0)
                    timestamps = list(timestamps) + [timestamps[-1]]
                add_text(f"<Video {counters['video']}>: ")
                for i in range(0, int(frames.shape[0]), 2):
                    block_ts = (timestamps[i] + timestamps[i + 1]) / 2.0
                    add_text("<%.1f seconds>" % block_ts)
                    flatten, grid = process_ref_video_block(
                        frames[i : i + 2], processor
                    )
                    add_vision_block(flatten, grid[0])
            else:
                raise ValueError(f"unknown ref item type {kind!r}")

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if max_length is not None and max_length > 0:
        # the cap applies to the caption only; a vision block is structural
        # conditioning and cannot be truncated without corrupting it
        prompt_ids = prompt_ids[:max_length]
    token_ids += prompt_ids
    token_tags += [TEXT_TAG] * len(prompt_ids)
    if len(token_ids) == 0:
        # empty (unconditional) prompt: a single pad token keeps the sequence
        # non-degenerate; the model was not trained with CFG so this is only
        # ever a fallback
        token_ids = [tokenizer.pad_token_id or 0]
        token_tags = [TEXT_TAG]

    pixel_values = torch.cat(pixel_parts, dim=0) if pixel_parts else None
    image_grid_thw = torch.cat(grid_parts, dim=0) if grid_parts else None
    return token_ids, token_tags, pixel_values, image_grid_thw


@torch.no_grad()
def encode_minimax_h3_prompt(
    text_encoder,  # transformers Qwen3VLForConditionalGeneration
    tokenizer,  # Qwen2TokenizerFast
    processor,  # Qwen3VLProcessor (needed only when keyframes/refs are present)
    prompt: str,
    keyframes: Optional[List] = None,  # PIL images already on the target canvas
    ref_items: Optional[List[dict]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_length: Optional[int] = None,  # cap on PROMPT tokens (vision blocks are never cut)
):
    """Encode ONE prompt (with optional keyframes or ref2va references) into
    MiniMax-H3 conditioning.

    ``ref_items`` is a list of dicts in presentation order:

      - ``{"type": "image", "data": <PIL image or array>}``
      - ``{"type": "audio"}`` (label only)
      - ``{"type": "video", "data": (T, C, H, W) float [0, 1] at 2 fps,
         "timestamps": [seconds, ...]}`` — use ``sample_ref_video_2fps`` to
        produce the subsample from a 24 fps clip

    Returns (embeds (L, 5120), token_tags (L,) long). The embeds come from
    ``hidden_states[50]`` unnormalized. A stack truncated to exactly 50 layers
    also works ONLY if the final ``model.norm`` has been replaced with an
    Identity (transformers applies the final norm to the last entry of
    ``hidden_states``); the loader in minimax_h3.py does exactly that.
    """
    num_layers = text_encoder.config.text_config.num_hidden_layers
    if num_layers < TEXT_ENCODER_LAYER:
        raise ValueError(
            f"MiniMax-H3 needs at least {TEXT_ENCODER_LAYER} Qwen3-VL decoder "
            f"layers to read hidden_states[{TEXT_ENCODER_LAYER}], got {num_layers}"
        )
    if device is None:
        device = text_encoder.device

    token_ids, token_tags, pixel_values, image_grid_thw = (
        build_minimax_h3_presentation(
            tokenizer,
            processor,
            prompt,
            keyframes=keyframes,
            ref_items=ref_items,
            max_length=max_length,
        )
    )

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor(
        processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device
    )

    # call the inner .model directly: the LM head's vocab projection is dead
    # weight here and hidden_states[50] is all that is consumed
    outputs = text_encoder.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_token_type_ids,
        pixel_values=None
        if pixel_values is None
        else pixel_values.to(device, text_encoder.dtype),
        image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(device),
        use_cache=False,
        output_hidden_states=True,
    )
    layer = min(TEXT_ENCODER_LAYER, len(outputs.hidden_states) - 1)
    embeds = outputs.hidden_states[layer][0]
    if dtype is not None:
        embeds = embeds.to(dtype)
    return embeds, torch.tensor(token_tags, dtype=torch.long)
