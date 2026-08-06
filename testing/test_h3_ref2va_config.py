"""Config dry-run + preview-pipeline smoke for MiniMax H3 ref2va.

Parses config/examples/train_lora_minimax_h3_ref2va.yaml, constructs
DatasetConfig / ModelConfig from it and checks the ref2va-specific keys
land. Then runs the sampling pipeline with reference videos on a tiny
random transformer (VAE stubbed) and checks it denoises to the target
shape with the references in the sequence.

Usage:  python testing/test_h3_ref2va_config.py
"""

import os
import sys

import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import DatasetConfig, ModelConfig

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def main():
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "examples", "train_lora_minimax_h3_ref2va.yaml",
    )
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    process = raw["config"]["process"][0]

    ds = DatasetConfig(**process["datasets"][0])
    check("reference_path parsed as list",
          isinstance(ds.reference_path, list) and len(ds.reference_path) == 2,
          str(ds.reference_path))
    check("num_frames on the 17n+5 grid", ds.num_frames % 17 == 5, str(ds.num_frames))
    check("latent caching on", ds.cache_latents_to_disk is True)

    mc = ModelConfig(**process["model"])
    check("arch minimax_h3", mc.arch == "minimax_h3", mc.arch)
    check("partition ref2va", mc.model_kwargs.get("partition") == "ref2va")

    check("audio loss zeroed",
          float(process["train"].get("audio_loss_multiplier", 1.0)) == 0.0)
    check("text embeds cached", process["train"].get("cache_text_embeddings") is True)

    # ---- preview pipeline smoke with references ---------------------------
    print("pipeline ref smoke:")
    from types import MethodType

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from testing.test_h3_reference_forward import make_model, make_embeds

    torch.manual_seed(5)
    model = make_model("ref2va")
    model.model = model.model.inner  # unwrap the recorder; pipeline calls kwargs too

    def fake_encode_images(self, image_list, device=None, dtype=None):
        out = []
        for item in image_list:
            if item.ndim == 3:
                item = item.unsqueeze(1)  # (C, 1, H, W)
            else:
                item = item.permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)
            c, t, h, w = item.shape
            lt = 1 if t == 1 else (t - 5) // 17 * 5 + 2
            out.append(torch.randn(24, lt, h // 16, w // 16))
        return torch.stack(out)

    def fake_decode_latents(self, latents, device=None, dtype=None):
        b, c, t, h, w = latents.shape
        pixel_t = 1 if t == 1 else (t - 2) // 5 * 17 + 5
        return torch.zeros(b, 3, pixel_t, h * 16, w * 16)

    model.encode_images = MethodType(fake_encode_images, model)
    model.decode_latents = MethodType(fake_decode_latents, model)

    from extensions_built_in.diffusion_models.minimax_h3.src.pipeline import (
        MiniMaxH3Pipeline,
    )

    # transformer property: BaseModel exposes .transformer -> unet/model
    pipe = MiniMaxH3Pipeline(model)
    embeds = make_embeds()
    refs = [torch.rand(22, 3, 64, 96), torch.rand(1, 3, 64, 96)]  # video + image
    result = pipe(
        conditional_embeds=embeds,
        height=96,
        width=128,
        num_frames=22,
        num_inference_steps=3,
        generator=torch.Generator().manual_seed(0),
        ref_videos=refs,
        with_audio=False,
    )
    check("pipeline returns video dict", isinstance(result, dict) and "video" in result)
    check("video shape (T, H, W, C)",
          tuple(result["video"].shape) == (22, 96, 128, 3),
          str(tuple(result["video"].shape)))

    try:
        pipe(
            conditional_embeds=embeds, height=96, width=128, num_frames=22,
            num_inference_steps=2, ref_videos=refs,
            ctrl_img=Image.new("RGB", (128, 96)),
        )
        check("ctrl_img + refs rejected", False, "no ValueError")
    except ValueError:
        check("ctrl_img + refs rejected", True)

    print()
    if FAILURES:
        print(f"TEST FAIL - {len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("TEST PASS - ref2va example config + preview pipeline")


if __name__ == "__main__":
    main()
