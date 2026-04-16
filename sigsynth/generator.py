from __future__ import annotations

import gc
import io
import resource
import shutil
from pathlib import Path
import zipfile

import numpy as np

from sigsynth.numpy_synth import synthesize_sample
from sigsynth.models import AppConfig
from sigsynth.hdf5_export import write_config_yaml, write_torchsig_compatible_hdf5
from sigsynth.paths import sanitize_output_dir
from sigsynth.post_transforms import NUMPY_POST_TRANSFORMS, apply_post_transform
from sigsynth.registry import GENERATOR_REGISTRY
from sigsynth.registry import resolve_generator_name
from sigsynth.registry import to_torchsig_generator_name


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
    class_list = config.global_params.get("class_list", "all")
    class_distribution = config.global_params.get("class_distribution", "uniform")
    nyquist = max(1, sample_rate // 2)
    frequency_limit = max(1, nyquist - 1)
    metadata = TorchSigDefaults().default_dataset_metadata
    metadata["sample_rate"] = sample_rate
    metadata["num_iq_samples_dataset"] = sample_len
    metadata["class_list"] = class_list
    metadata["class_distribution"] = class_distribution

    # Set SNR range from config (default to -2 to 30 for Sig53 compatibility)
    snr_db = config.global_params.get("snr_db", [-2, 30])
    if isinstance(snr_db, (list, tuple)) and len(snr_db) >= 2:
        metadata["snr_db_min"] = float(snr_db[0])
        metadata["snr_db_max"] = float(snr_db[1])
    else:
        metadata["snr_db_min"] = -2.0
        metadata["snr_db_max"] = 30.0

    # NOTE: eb_no parameter existed in TorchSig v0.1.0 to control whether SNR is defined
    # as Eb/No (energy per bit, making higher-order modulations more powerful) or
    # Es/No (energy per symbol, the canonical Sig53 approach). TorchSig v2.1.0 does not
    # appear to expose this parameter in the metadata-driven API. The canonical Sig53
    # datasets used Es/No (eb_no=False). Our numpy fallback also uses Es/No.
    metadata["bandwidth_max"] = min(int(metadata.get("bandwidth_max", frequency_limit)), frequency_limit)
    metadata["bandwidth_min"] = min(int(metadata.get("bandwidth_min", metadata["bandwidth_max"])), metadata["bandwidth_max"])

    # For Sig53 compatibility, use the narrower range from original implementation
    # Original used RandomFrequencyShift((-.16, .16)) which is ±16% of sample rate
    # Allow override via config
    center_freq_range_factor = config.global_params.get("signal_center_freq_range_factor", 0.16)
    default_center_freq_min = -int(sample_rate * center_freq_range_factor)
    default_center_freq_max = int(sample_rate * center_freq_range_factor)

    metadata["signal_center_freq_min"] = max(
        int(metadata.get("signal_center_freq_min", default_center_freq_min)),
        -frequency_limit,
    )
    metadata["signal_center_freq_max"] = min(
        int(metadata.get("signal_center_freq_max", default_center_freq_max)),
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

    explicit_num_signals_min = config.global_params.get("num_signals_min")
    explicit_num_signals_max = config.global_params.get("num_signals_max")
    if explicit_num_signals_min is not None or explicit_num_signals_max is not None:
        metadata["num_signals_min"] = int(explicit_num_signals_min or 1)
        metadata["num_signals_max"] = int(explicit_num_signals_max or metadata["num_signals_min"])
    elif "wideband" in generator_tags:
        metadata["num_signals_min"] = max(int(metadata.get("num_signals_min", 1)), 3)
        metadata["num_signals_max"] = max(int(metadata.get("num_signals_max", 1)), 5)
    else:
        metadata["num_signals_min"] = 1
        metadata["num_signals_max"] = 1

    return metadata


def _check_disk_space(output_dir: Path, config: AppConfig) -> None:
    """Check if sufficient disk space is available before starting generation."""
    # Estimate dataset size: sample_len * 8 bytes (complex64) * 2 (raw + impaired for numpy)
    sample_size_bytes = int(config.global_params.get("sample_len", 1024)) * 8
    if config.dataset.output_format == "numpy":
        sample_size_bytes *= 2  # Both raw and impaired for numpy
        # NumPy files are uncompressed
        compression_factor = 1.0
    else:
        # HDF5 with gzip level 6 typically achieves 2-4x compression on IQ data
        # Use conservative estimate of 2.5x for disk space check
        compression_factor = 0.4  # Means: final size = 40% of uncompressed

    total_samples = config.dataset.total_samples
    estimated_bytes = int(sample_size_bytes * total_samples * compression_factor)

    # Add overhead for metadata (10%)
    estimated_bytes = int(estimated_bytes * 1.1)

    # Check available space on target path
    # Walk up the directory tree until we find an existing directory
    try:
        check_path = output_dir
        while not check_path.exists():
            if check_path.parent == check_path:
                # Reached filesystem root without finding existing directory
                raise RuntimeError(f"Cannot determine disk space for {output_dir}: no parent directory exists")
            check_path = check_path.parent

        stat = shutil.disk_usage(check_path)
        available_bytes = stat.free
    except Exception as e:
        print(f"Warning: Could not check disk space: {e}")
        return

    # Require 20% safety margin
    required_bytes = int(estimated_bytes * 1.2)

    if available_bytes < required_bytes:
        raise RuntimeError(
            f"Insufficient disk space on {output_dir}:\n"
            f"  Available: {available_bytes / 1e9:.1f} GB\n"
            f"  Required:  {required_bytes / 1e9:.1f} GB (estimated + 20% margin)\n"
            f"  Shortfall: {(required_bytes - available_bytes) / 1e9:.1f} GB"
        )

    print(f"Disk space check passed: {available_bytes / 1e9:.1f} GB available, {required_bytes / 1e9:.1f} GB required")


def _attempt_torchsig_generation(config: AppConfig, output_dir: Path) -> tuple[bool, str | None]:
    """Best-effort generation through TorchSig APIs with fallback behavior."""
    try:
        from torch.utils.data import DataLoader  # type: ignore
        from torchsig.datasets.datasets import TorchSigIterableDataset  # type: ignore
        from torchsig.utils.writer import DatasetCreator  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on external submodule state
        return False, f"TorchSig generate API unavailable: {exc}"

    dataloader = None
    dataset = None
    try:
        metadata = _build_torchsig_metadata(config)
        batch_size = config.dataset.create_batch_size
        torchsig_generators = [
            to_torchsig_generator_name(name) or name.lower()
            for name in config.generators
        ]
        seed = config.global_params.get("seed")
        dataset = TorchSigIterableDataset(
            signal_generators=torchsig_generators,
            metadata=metadata,
            transforms=[],
            component_transforms=[],
            seed=seed,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=config.dataset.create_num_workers,
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
    finally:
        # Clean up DataLoader workers to prevent memory leaks
        if dataloader is not None:
            del dataloader
        if dataset is not None:
            del dataset
        gc.collect()


def _reset_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _split_counts(config: AppConfig) -> tuple[int, int, str]:
    split_mode = str(config.dataset.split_mode or "split").lower()
    total_samples = int(config.dataset.total_samples)

    if split_mode == "train_only":
        return total_samples, 0, split_mode
    if split_mode == "val_only":
        return 0, total_samples, split_mode

    train_count = int(total_samples * config.dataset.train_ratio)
    val_count = total_samples - train_count
    return train_count, val_count, "split"


def _generate_numpy_dataset(config: AppConfig, output_dir: Path) -> None:
    train_count, val_count, split_mode = _split_counts(config)
    apply_post_transforms = bool(config.global_params.get("apply_post_transforms", False))
    post_transform_warnings: set[str] = set()

    layout = []
    if train_count > 0:
        layout.extend(
            [
                output_dir / "train" / "raw",
                output_dir / "train" / "impaired",
            ]
        )
    if val_count > 0:
        layout.extend(
            [
                output_dir / "val" / "raw",
                output_dir / "val" / "impaired",
            ]
        )
    for folder in layout:
        folder.mkdir(parents=True, exist_ok=True)

    # GC when we hit EITHER threshold (more frequent = safer):
    # - At least every 100 samples (avoid excessive overhead for tiny samples)
    # - At most every 8 GB of data (prevent memory buildup for large samples)
    # With 64-128 GB memory limits, 8 GB gives good balance between GC overhead and memory safety
    sample_len = int(config.global_params.get("sample_len", 1024))
    bytes_per_sample = sample_len * 8 * 2  # complex64 = 8 bytes, raw + impaired = 2x
    gc_interval_bytes = 8 * 1024 * 1024 * 1024  # 8 GB max memory between GCs
    gc_interval_samples = max(100, gc_interval_bytes // max(1, bytes_per_sample))  # At least 100 samples

    bytes_written_since_gc = 0

    for split, count in (("train", train_count), ("val", val_count)):
        if count <= 0:
            continue
        for idx in range(count):
            if split_mode == "split" and split == "val":
                sample_index = train_count + idx
            else:
                sample_index = idx
            sample = synthesize_sample(config, sample_index)
            impaired = sample.impaired
            if apply_post_transforms:
                for step in config.transforms:
                    if not step.enabled:
                        continue
                    if step.name in NUMPY_POST_TRANSFORMS:
                        impaired = apply_post_transform(step.name, impaired, config)
                    elif step.name not in post_transform_warnings:
                        post_transform_warnings.add(step.name)

            np.save(output_dir / split / "raw" / f"sample_{idx:06d}.npy", sample.clean)
            np.save(output_dir / split / "impaired" / f"sample_{idx:06d}.npy", impaired)

            # Memory management: explicit cleanup after each sample
            del impaired
            del sample

            bytes_written_since_gc += bytes_per_sample

            # GC based on data volume (~1 GB) rather than sample count
            # This avoids excessive overhead for small samples and ensures cleanup for large samples
            if bytes_written_since_gc >= gc_interval_bytes or (idx + 1) % gc_interval_samples == 0:
                gc.collect()
                bytes_written_since_gc = 0
                total_so_far = (train_count if split == "val" else 0) + idx + 1
                print(f"NumPy Progress: {total_so_far}/{train_count + val_count} samples ({total_so_far / (train_count + val_count) * 100:.1f}%)")

    if post_transform_warnings:
        config.global_params["post_transform_warnings"] = sorted(post_transform_warnings)
    else:
        config.global_params.pop("post_transform_warnings", None)


def generate_dataset(config: AppConfig) -> dict[str, int | str | bool]:
    # Apply memory limit if configured
    if config.dataset.max_memory_mb:
        try:
            max_bytes = config.dataset.max_memory_mb * 1024 * 1024
            # Use RLIMIT_DATA (heap) instead of RLIMIT_AS (virtual memory)
            # This avoids issues with memory-mapped HDF5 files
            resource.setrlimit(resource.RLIMIT_DATA, (max_bytes, max_bytes))
        except (ValueError, OSError, AttributeError) as e:
            # Some systems don't support RLIMIT_DATA, try RLIMIT_AS as fallback
            try:
                resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
                print(f"Warning: Using RLIMIT_AS instead of RLIMIT_DATA (may affect HDF5)")
            except (ValueError, OSError):
                print(f"Warning: Could not set memory limit: {e}")

    output_dir = sanitize_output_dir(config.dataset.output_dir)

    # Check disk space BEFORE starting generation
    try:
        _check_disk_space(output_dir, config)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return {
            "output_dir": str(output_dir),
            "output_format": config.dataset.output_format,
            "train_samples": 0,
            "val_samples": 0,
            "torchsig_generated": False,
            "torchsig_error": str(e),
            "post_transform_warnings": [],
        }

    _reset_output_dir(output_dir)

    config.global_params.pop("post_transform_warnings", None)
    train_count, val_count, _ = _split_counts(config)
    torchsig_generated = False
    torchsig_error = ""

    if config.dataset.output_format == "hdf5":
        torchsig_generated, torchsig_error = _attempt_torchsig_generation(config, output_dir)
        if not torchsig_generated:
            write_torchsig_compatible_hdf5(output_dir, config, int(config.dataset.total_samples))
    else:
        _generate_numpy_dataset(config, output_dir)

    write_config_yaml(output_dir, config)

    return {
        "output_dir": str(output_dir),
        "output_format": config.dataset.output_format,
        "train_samples": train_count,
        "val_samples": val_count,
        "torchsig_generated": torchsig_generated,
        "torchsig_error": torchsig_error or "",
        "post_transform_warnings": config.global_params.get("post_transform_warnings", []),
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
