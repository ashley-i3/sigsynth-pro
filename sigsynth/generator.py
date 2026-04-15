from __future__ import annotations

from pathlib import Path

import numpy as np

from sigsynth.models import AppConfig


def _attempt_torchsig_generation(config: AppConfig) -> bool:
    """Best-effort torchsig smoke path.

    TorchSig APIs differ across versions, so this checks import and reports availability.
    Returns True if torchsig is importable; otherwise False.
    """
    try:
        import torchsig  # noqa: F401
    except Exception:
        return False
    return True


def generate_dataset(config: AppConfig) -> dict[str, int | str | bool]:
    output_dir = Path(config.dataset.output_dir)
    train_count = int(config.dataset.total_samples * config.dataset.train_ratio)
    val_count = config.dataset.total_samples - train_count

    layout = [
        output_dir / "train" / "raw",
        output_dir / "train" / "impaired",
        output_dir / "val" / "raw",
        output_dir / "val" / "impaired",
    ]
    for folder in layout:
        folder.mkdir(parents=True, exist_ok=True)

    torchsig_available = _attempt_torchsig_generation(config)

    rng = np.random.default_rng(seed=53)
    sample_len = int(config.global_params.get("sample_len", 1024))

    for split, count in (("train", train_count), ("val", val_count)):
        for idx in range(count):
            raw = rng.standard_normal(sample_len) + 1j * rng.standard_normal(sample_len)
            impaired = raw * (1 + 0.01 * rng.standard_normal(sample_len))

            np.save(output_dir / split / "raw" / f"sample_{idx:06d}.npy", raw)
            np.save(output_dir / split / "impaired" / f"sample_{idx:06d}.npy", impaired)

    return {
        "output_dir": str(output_dir),
        "train_samples": train_count,
        "val_samples": val_count,
        "torchsig_available": torchsig_available,
    }
