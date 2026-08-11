"""R5c: per-sample reference dropout + precomputed caption-variant selection.

Before this, ``has_references`` was fixed per file item — every sample saw
every reference, every step — so no dropout curriculum was expressible at all
(h3-crossview open question 5). This covers the properties the mechanism has
to have to be trusted:

  1. unconfigured dropout is a bit-exact no-op
  2. the empirical drop rate matches the configured probability, PER STREAM
     (a scalar could only ever say "train unconditioned x% of the time", which
     is the wrong regime when the source is mandatory and the warp is not)
  3. the linear anneal moves the rate over steps and HOLDS past the horizon —
     this is what makes a warp-scaffold curriculum expressible
  4. a dropped stream selects the caption variant PRECOMPUTED for that
     reference set (dropping <Video 3> renumbers ordinals, so a caption still
     naming it trains a tag that is not there), and a missing variant is loud
  5. deterministic in (seed, path, step), independent across samples

Usage:  python testing/test_h3_reference_dropout.py
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import DatasetConfig
from toolkit.dataloader_mixins import ReferenceFileItemDTOMixin

FULL = "FULL CAPTION naming <Video 3>"
STRIPPED = "STRIPPED CAPTION with no reference 3"


class _Base:
    # the mixin cooperatively calls super().__init__(**kwargs); in the real
    # FileItemDTO the next class in the MRO absorbs those, and object does not
    def __init__(self, *args, **kwargs):
        pass


class FakeItem(ReferenceFileItemDTOMixin, _Base):
    """Minimal stand-in: the mixin needs only .path, .dataset_config and a
    caption hook. Driving the real FileItemDTO would drag in bucketing, latent
    caching and a model, none of which this behaviour touches."""

    def __init__(self, path, dataset_config):
        self.path = path
        self.dataset_config = dataset_config
        self.raw_caption = FULL
        self.caption = FULL
        super().__init__(dataset_config=dataset_config)

    def get_caption(self, **kwargs):
        return self.raw_caption


def build_workspace(root):
    refs = []
    for name in ("source_ref", "raymap_ref", "warp_ref"):
        rd = os.path.join(root, name)
        os.makedirs(rd)
        for stem in ("pair0", "pair1"):
            open(os.path.join(rd, stem + ".mp4"), "wb").write(b"\x00")
        refs.append(rd)
    targets = os.path.join(root, "targets")
    os.makedirs(targets)
    for stem in ("pair0", "pair1"):
        open(os.path.join(targets, stem + ".mp4"), "wb").write(b"\x00")
        with open(os.path.join(targets, stem + ".nowarp.txt"), "w", encoding="utf-8") as f:
            f.write(STRIPPED)
    return os.path.join(targets, "pair0.mp4"), os.path.join(targets, "pair1.mp4"), refs


def cfg(refs, **kw):
    return DatasetConfig(folder_path=os.path.dirname(refs[0]), reference_path=refs, **kw)


def main():
    root = tempfile.mkdtemp(prefix="h3_refdrop_")
    try:
        train, other, refs = build_workspace(root)

        # 1 -- unconfigured dropout is a no-op
        item = FakeItem(train, cfg(refs))
        assert item.has_references and len(item.reference_paths) == 3
        assert item.select_references(step=0, seed=0) == []
        assert item.active_reference_paths == item.reference_paths
        assert item.caption == FULL
        print("  [ok] unconfigured dropout is a bit-exact no-op")

        # 2 -- per-stream empirical rate
        c = cfg(refs, reference_dropout=[0.0, 0.0, 0.3])
        n, counts = 4000, [0, 0, 0]
        for step in range(n):
            for i in FakeItem(train, c).select_references(step=step, seed=7):
                counts[i] += 1
        assert counts[0] == 0 and counts[1] == 0, f"mandatory streams dropped: {counts}"
        rate = counts[2] / n
        assert abs(rate - 0.3) < 0.02, f"warp drop rate {rate:.4f} != 0.3"
        print(f"  [ok] per-stream rate: source 0, raymap 0, warp {rate:.4f} (want 0.30)")

        # 3 -- anneal
        c = cfg(refs, reference_dropout=[0.0, 0.0, 0.0],
                reference_dropout_end=[0.0, 0.0, 0.8],
                reference_dropout_anneal_steps=1000)

        def rate_at(step, draws=2000):
            hits = 0
            for k in range(draws):
                if 2 in FakeItem(train, c).select_references(step=step, seed=k):
                    hits += 1
            return hits / draws

        r0, r500, r1000, r5000 = rate_at(0), rate_at(500), rate_at(1000), rate_at(5000)
        assert r0 == 0.0, r0
        assert abs(r500 - 0.4) < 0.03, r500
        assert abs(r1000 - 0.8) < 0.03, r1000
        assert abs(r5000 - 0.8) < 0.03, f"extrapolated past the horizon: {r5000}"
        print(f"  [ok] anneal 0->0.8 over 1000 steps: {r0:.3f} {r500:.3f} "
              f"{r1000:.3f}, holds at {r5000:.3f} past the horizon")

        # 4 -- caption variant selection
        c = cfg(refs, reference_dropout=[0.0, 0.0, 1.0],
                reference_caption_variants=[None, None, "nowarp"])
        item = FakeItem(train, c)
        assert item.select_references(step=0, seed=1) == [2]
        assert len(item.active_reference_paths) == 2
        assert "warp_ref" not in " ".join(item.active_reference_paths)
        assert item.caption == STRIPPED and "<Video 3>" not in item.caption
        print("  [ok] dropped stream loads the matching precomputed caption variant")

        c = cfg(refs, reference_dropout=[0.0, 0.0, 1.0],
                reference_caption_variants=[None, None, "doesnotexist"])
        item = FakeItem(train, c)
        item.select_references(step=0, seed=1)
        assert item.caption == FULL, "silently trained a caption naming a dropped ref"
        print("  [ok] missing variant warns and keeps the full caption (loud, not silent)")

        # 5 -- determinism + independence
        c = cfg(refs, reference_dropout=[0.0, 0.0, 0.5])
        a = [FakeItem(train, c).select_references(step=s, seed=3) for s in range(50)]
        b = [FakeItem(train, c).select_references(step=s, seed=3) for s in range(50)]
        d = [FakeItem(train, c).select_references(step=s, seed=4) for s in range(50)]
        assert a == b, "not reproducible at a fixed seed"
        assert a != d, "seed is ignored (a constant generator would pass the above)"
        same = sum(
            (2 in FakeItem(train, c).select_references(step=s, seed=11))
            == (2 in FakeItem(other, c).select_references(step=s, seed=11))
            for s in range(400)
        ) / 400
        assert 0.35 < same < 0.65, f"draws correlated across samples: {same}"
        print(f"  [ok] deterministic under a fixed seed; independent across "
              f"samples (agreement {same:.3f}, chance 0.5)")

        print("TEST PASS - per-sample reference dropout + caption-variant selection")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
