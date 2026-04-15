from __future__ import annotations

import io
import shutil
from pathlib import Path
import zipfile

import numpy as np

from sigsynth.models import AppConfig
from sigsynth.hdf5_export import write_config_yaml, write_torchsig_compatible_hdf5
from sigsynth.paths import sanitize_output_dir
from sigsynth.registry import GENERATOR_REGISTRY
from sigsynth.registry import resolve_generator_name


def _build_torchsig_metadata(config: AppConfig):
    """Create TorchSig 2.0 metadata using the documented defaults object."""
    try:
        from torchsig.utils.defaults import TorchSigDefaults  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on external submodule state
        raise RuntimeError("TorchSig defaults module is unavailable.") from exc

    generator_tags = {
        tag
        for name in config.generators
        for tag in GENERATOR_REGISTRY.get(
            resolve_generator_name(name) or name, GENERATOR_REGISTRY["BPSK"]
        ).tags
    }

    sample_rate = int(config.global_params.get("sample_rate", 1_000_000))
    sample_len = int(config.global_params.get("sample_len", 1024))
    nyquist = max(1, sample_rate // 2)
    frequency_limit = max(1, nyquist - 1)
    metadata = TorchSigDefaults().default_dataset_metadata
    metadata["sample_rate"] = sample_rate
    metadata["num_iq_samples_dataset"] = sample_len
    metadata["bandwidth_max"] = min(int(metadata.get("bandwidth_max", frequency_limit)), frequency_limit)
    metadata["bandwidth_min"] = min(int(metadata.get("bandwidth_min", metadata["bandwidth_max"])), metadata["bandwidth_max"])
    metadata["signal_center_freq_min"] = max(
        int(metadata.get("signal_center_freq_min", -frequency_limit)),
        -frequency_limit,
    )
    metadata["signal_center_freq_max"] = min(
        int(metadata.get("signal_center_freq_max", frequency_limit)),
        frequency_limit,
    )
    metadata["frequency_min"] = max(int(metadata.get("frequency_min", -frequency_limit)), -frequency_limit)
    metadata["frequency_max"] = min(int(metadata.get("frequency_max", frequency_limit)), frequency_limit)
    duration_cap = max(1, sample_len)
    duration_min = max(
        1,
        min(
            duration_cap,
            int(metadata.get("signal_duration_in_samples_min", duration_cap)),
        ),
    )
    duration_max = max(
        duration_min,
        min(
            duration_cap,
            int(metadata.get("signal_duration_in_samples_max", duration_cap)),
        ),
    )
    metadata["signal_duration_in_samples_min"] = duration_min
    metadata["signal_duration_in_samples_max"] = duration_max

    if "wideband" in generator_tags:
        metadata["num_signals_min"] = max(int(metadata.get("num_signals_min", 1)), 1)
        metadata["num_signals_max"] = max(int(metadata.get("num_signals_max", 1)), 3)
    else:
        metadata["num_signals_min"] = 1
        metadata["num_signals_max"] = 1

    return metadata


def _attempt_torchsig_generation(config: AppConfig, output_dir: Path) -> tuple[bool, str | None]:
    """Best-effort generation through TorchSig APIs with fallback behavior."""
    try:
        from torch.utils.data import DataLoader  # type: ignore
        from torchsig.datasets.datasets import TorchSigIterableDataset  # type: ignore
        from torchsig.utils.writer import DatasetCreator  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on external submodule state
        return False, f"TorchSig generate API unavailable: {exc}"

    try:
        metadata = _build_torchsig_metadata(config)
        batch_size = max(1, min(256, int(config.dataset.total_samples)))
        dataset = TorchSigIterableDataset(
            metadata=metadata,
            transforms=[],
            component_transforms=[],
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=0,
        )
        creator = DatasetCreator(
            dataloader=dataloader,
            dataset_length=int(config.dataset.total_samples),
            root=str(output_dir),
            overwrite=True,
        )
        creator.create()
        return True, None
    except Exception as exc:  # pragma: no cover - depends on torchsig version
        return False, str(exc)


def _reset_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _generate_numpy_dataset(config: AppConfig, output_dir: Path) -> None:
    rng = np.random.default_rng(seed=53)
    sample_len = int(config.global_params.get("sample_len", 1024))
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

    for split, count in (("train", train_count), ("val", val_count)):
        for idx in range(count):
            raw = rng.standard_normal(sample_len) + 1j * rng.standard_normal(sample_len)
            impaired = raw * (1 + 0.01 * rng.standard_normal(sample_len))

            np.save(output_dir / split / "raw" / f"sample_{idx:06d}.npy", raw)
            np.save(output_dir / split / "impaired" / f"sample_{idx:06d}.npy", impaired)


def generate_dataset(config: AppConfig) -> dict[str, int | str | bool]:
    output_dir = sanitize_output_dir(config.dataset.output_dir)
    _reset_output_dir(output_dir)

    torchsig_generated = False
    torchsig_error = ""

    if config.dataset.output_format == "hdf5":
        train_count = int(config.dataset.total_samples)
        val_count = 0
        torchsig_generated, torchsig_error = _attempt_torchsig_generation(config, output_dir)
        if not torchsig_generated:
            write_torchsig_compatible_hdf5(output_dir, config, int(config.dataset.total_samples))
    else:
        train_count = int(config.dataset.total_samples * config.dataset.train_ratio)
        val_count = config.dataset.total_samples - train_count
        _generate_numpy_dataset(config, output_dir)

    write_config_yaml(output_dir, config)

    return {
        "output_dir": str(output_dir),
        "output_format": config.dataset.output_format,
        "train_samples": train_count,
        "val_samples": val_count,
        "torchsig_generated": torchsig_generated,
        "torchsig_error": torchsig_error or "",
    }


def build_dataset_zip_bytes(output_dir: str | Path) -> bytes:
    root = sanitize_output_dir(output_dir)
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in root.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(root))
    memory_file.seek(0)
    return memory_file.getvalue()
