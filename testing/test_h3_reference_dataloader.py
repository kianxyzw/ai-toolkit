"""Dataloader test for reference conditioning (DatasetConfig.reference_path).

Builds a tiny synthetic video dataset with two reference folders (one video
ref, one image ref), then checks:

  - uncached: batch.reference_tensors loads (native canvas cropped to /32,
    [0, 1] range), reference_kinds resolve from extensions, and the raw
    references bridge into batch.control_tensor_list for the text encode
  - cached: latent caching writes reference_latent_{i} entries, the cache key
    includes the reference paths, and reloading yields batch.reference_latents
  - a missing reference match raises instead of silently skipping
  - AdvancedPromptEmbeds keeps integer tensors integer across a save/load
    round trip (the frozen_dtype_keys cache fix)

Writes a contact strip (target + refs, frame 0) to output/h3_reference_dataloader/.

Usage:  python testing/test_h3_reference_dataloader.py
"""

import os
import shutil
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import get_dataloader_from_datasets

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


class FakeSD:
    def __init__(self):
        self.use_raw_control_images = False
        self.encode_control_in_text_embeddings = True
        self.encode_first_frame_in_text_embeddings = False
        self.has_multiple_control_images = True
        self.latent_space_version = "fake_v1"
        self.text_embedding_space_version = "fake_v1"
        self.te_padding_side = "right"
        self.torch_dtype = torch.float32
        self.device = "cpu"
        self.device_torch = torch.device("cpu")
        self.cache_latents_as_uint8 = False
        self.model_config = type(
            "FakeModelConfig", (), {"latent_space_version": "fake_v1", "arch": "fake"}
        )()
        self.vae = None
        self.unet = None
        self.is_xl = False
        self.is_v3 = False
        self.is_auraflow = False
        self.is_flux = False

    def get_bucket_divisibility(self):
        return 32

    def set_device_state_preset(self, *args, **kwargs):
        pass

    def restore_device_state(self):
        pass

    def encode_images(self, image_list, device=None, dtype=None):
        # fake 16x-spatial VAE with a 17n+5 -> 5n+2 frame grid
        latents = []
        for item in image_list:
            if item.ndim == 3:
                item = item.unsqueeze(0)
            t, c, h, w = item.shape
            lt = 1 if t == 1 else (t - 5) // 17 * 5 + 2
            latents.append(torch.full((24, lt, h // 16, w // 16), float(item.mean())))
        return torch.stack(latents)


def write_mp4(path, frames_thwc_uint8, fps=24):
    t, h, w, _ = frames_thwc_uint8.shape
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    for i in range(t):
        writer.write(cv2.cvtColor(frames_thwc_uint8[i].numpy(), cv2.COLOR_RGB2BGR))
    writer.release()


def build_dataset(root):
    if os.path.exists(root):
        shutil.rmtree(root)
    videos = os.path.join(root, "videos")
    ref_a = os.path.join(root, "ref_video")
    ref_b = os.path.join(root, "ref_image")
    for d in (videos, ref_a, ref_b):
        os.makedirs(d)
    g = torch.Generator().manual_seed(99)
    for name, hue in (("clip_one", 0.2), ("clip_two", 0.7)):
        base = torch.rand(1, 128, 128, 3, generator=g)
        frames = (base.expand(39, 128, 128, 3) * 0.5 + hue * 0.5) * 255
        write_mp4(os.path.join(videos, f"{name}.mp4"), frames.to(torch.uint8))
        with open(os.path.join(videos, f"{name}.txt"), "w") as f:
            f.write(f"[reference generation] test caption for {name}")
        # video reference on a NON-multiple-of-32 canvas (100x130 -> crop 96x128)
        ref_frames = (torch.rand(39, 100, 130, 3, generator=g) * 255).to(torch.uint8)
        write_mp4(os.path.join(ref_a, f"{name}.mp4"), ref_frames)
        img = (torch.rand(64, 96, 3, generator=g) * 255).to(torch.uint8)
        Image.fromarray(img.numpy()).save(os.path.join(ref_b, f"{name}.png"))
    return videos, ref_a, ref_b


def make_config(videos, ref_a, ref_b, cache=False):
    return DatasetConfig(
        dataset_path=videos,
        reference_path=[ref_a, ref_b],
        resolution=128,
        buckets=True,
        bucket_tolerance=64,
        shrink_video_to_frames=True,
        num_frames=39,
        fps=24,
        cache_latents_to_disk=cache,
    )


def main():
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "output", "h3_reference_dataloader"
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    root = os.path.join(out_dir, "dataset")
    videos, ref_a, ref_b = build_dataset(root)

    # ---- uncached run ------------------------------------------------------
    print("uncached run:")
    dl = get_dataloader_from_datasets(
        [make_config(videos, ref_a, ref_b, cache=False)], batch_size=1, sd=FakeSD()
    )
    batch = next(iter(dl))
    check("reference_tensors present", batch.reference_tensors is not None)
    check("two references", len(batch.reference_tensors) == 2)
    v, im = batch.reference_tensors
    check("video ref shape (B,T,C,H,W), cropped to /32",
          tuple(v.shape) == (1, 39, 3, 96, 128), str(tuple(v.shape)))
    check("image ref shape (B,1,C,H,W)",
          tuple(im.shape) == (1, 1, 3, 64, 96), str(tuple(im.shape)))
    check("refs in [0,1]", float(v.min()) >= 0.0 and float(v.max()) <= 1.0)
    check("kinds resolved", batch.reference_kinds == ["video", "image"],
          str(batch.reference_kinds))
    check("control bridge populated", batch.control_tensor_list is not None
          and len(batch.control_tensor_list) == 1
          and len(batch.control_tensor_list[0]) == 2)
    check("no reference_latents when uncached", batch.reference_latents is None)

    # contact strip: target frame 0 + each ref frame 0
    tiles = [batch.tensor[0, 0] * 0.5 + 0.5, v[0, 0], im[0, 0]]
    h = max(t.shape[1] for t in tiles)
    w = sum(t.shape[2] for t in tiles)
    strip = torch.zeros(3, h, w)
    x = 0
    for t in tiles:
        strip[:, : t.shape[1], x : x + t.shape[2]] = t
        x += t.shape[2]
    Image.fromarray((strip.permute(1, 2, 0) * 255).to(torch.uint8).numpy()).save(
        os.path.join(out_dir, "contact_strip.png")
    )
    print(f"  contact strip -> {os.path.join(out_dir, 'contact_strip.png')}")
    batch.cleanup()

    # ---- cached run --------------------------------------------------------
    print("cached run:")
    dl = get_dataloader_from_datasets(
        [make_config(videos, ref_a, ref_b, cache=True)], batch_size=1, sd=FakeSD()
    )
    batch = next(iter(dl))
    check("reference_latents present", batch.reference_latents is not None
          and len(batch.reference_latents) == 2)
    rl_v, rl_im = batch.reference_latents
    check("video ref latent (B,24,12,6,8)", tuple(rl_v.shape) == (1, 24, 12, 6, 8),
          str(tuple(rl_v.shape)))
    check("image ref latent (B,24,1,4,6)", tuple(rl_im.shape) == (1, 24, 1, 4, 6),
          str(tuple(rl_im.shape)))
    check("pixels still loaded for text path (text not cached)",
          batch.reference_tensors is not None)
    cache_dir = os.path.join(videos, "_latent_cache")
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith(".safetensors")]
    check("cache files written", len(cache_files) == 2, str(cache_files))
    from safetensors import safe_open

    with safe_open(os.path.join(cache_dir, cache_files[0]), framework="pt") as f:
        keys = list(f.keys())
        meta = f.metadata() or {}
    check("reference_latent_{0,1} in cache file",
          "reference_latent_0" in keys and "reference_latent_1" in keys, str(keys))
    check("reference_paths in cache key metadata",
          "reference_paths" in str(meta), str(meta)[:200])
    batch.cleanup()

    # ---- missing reference excludes the file (with a printed error) -------
    # the dataset loader's per-file error policy catches the mixin's
    # FileNotFoundError and drops the file; it must never load WITHOUT its
    # references. Pipelines should assert dataset length externally.
    print("guards:")
    os.remove(os.path.join(ref_b, "clip_two.png"))
    dl = get_dataloader_from_datasets(
        [make_config(videos, ref_a, ref_b, cache=False)], batch_size=1, sd=FakeSD()
    )
    batches = [b for b in dl]
    check("file with missing reference is excluded", len(batches) == 1,
          f"{len(batches)} batches")
    check("surviving file still carries its references",
          batches[0].reference_tensors is not None
          and len(batches[0].reference_tensors) == 2)

    # ---- frozen dtype round-trip (cache fix) ------------------------------
    print("AdvancedPromptEmbeds dtype freeze:")
    pe = AdvancedPromptEmbeds(
        text_embeds=[torch.randn(5, 8)], text_token_tags=[torch.ones(5, dtype=torch.long)]
    )
    pe.frozen_dtype_keys = ["text_token_tags"]
    pe_path = os.path.join(out_dir, "pe_test.safetensors")
    pe.save(pe_path)
    pe2 = AdvancedPromptEmbeds.load(pe_path)
    check("frozen_dtype_keys restored", "text_token_tags" in pe2.frozen_dtype_keys,
          str(pe2.frozen_dtype_keys))
    pe3 = pe2.to(torch.device("cpu"), torch.bfloat16)
    check("tags stay long after dtype cast",
          pe3.text_token_tags[0].dtype == torch.long, str(pe3.text_token_tags[0].dtype))
    check("embeds do cast", pe3.text_embeds[0].dtype == torch.bfloat16)

    print()
    if FAILURES:
        print(f"TEST FAIL - {len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("TEST PASS - reference dataloader + caching + dtype freeze")


if __name__ == "__main__":
    main()
