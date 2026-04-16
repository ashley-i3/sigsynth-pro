from __future__ import annotations

import gc
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from sigsynth.models import AppConfig
from sigsynth.numpy_synth import synthesize_sample


def _write_scalar_dataset(group: h5py.Group, key: str, value) -> None:
    if value is None:
        return

    if isinstance(value, np.ndarray):
        group.create_dataset(key, data=value)
        return

    if isinstance(value, (np.generic, int, float, bool)):
        group.create_dataset(key, data=value)
        return

    if isinstance(value, (list, tuple)):
        if not value:
            group.create_dataset(key, data=json.dumps(value), dtype=h5py.string_dtype("utf-8"))
            return
        if all(isinstance(item, (str, bytes)) for item in value):
            serialized = json.dumps(list(value))
            group.create_dataset(key, data=serialized, dtype=h5py.string_dtype("utf-8"))
            return
        if all(isinstance(item, dict) for item in value):
            serialized = json.dumps(value, sort_keys=True)
            group.create_dataset(key, data=serialized, dtype=h5py.string_dtype("utf-8"))
            return
        group.create_dataset(key, data=np.asarray(value))
        return

    if isinstance(value, dict):
        serialized = json.dumps(value, sort_keys=True)
        group.create_dataset(key, data=serialized, dtype=h5py.string_dtype("utf-8"))
        return

    group.create_dataset(key, data=str(value), dtype=h5py.string_dtype("utf-8"))


def write_torchsig_compatible_hdf5(output_dir: str | Path, config: AppConfig, total_samples: int) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    datapath = root / "data.h5"

    sample_rate = int(config.global_params.get("sample_rate", 1_000_000))
    sample_len = int(config.global_params.get("sample_len", 1024))

    # Performance optimizations for large sequential writes:
    # - rdcc_nbytes: Raw data chunk cache size (64 MB, up from default 1 MB)
    # - rdcc_nslots: Number of chunk slots in cache (521 is prime number for good hash distribution)
    # - rdcc_w0: Chunk preemption policy (0 = pure LRU, 1 = fully read chunks always preempted)
    with h5py.File(
        datapath,
        "w",
        libver="latest",
        rdcc_nbytes=64 * 1024 * 1024,  # 64 MB chunk cache
        rdcc_nslots=521,  # Prime number for good hash distribution
        rdcc_w0=0.75,  # Balanced preemption policy
    ) as h5:
        h5.attrs["created_by"] = "sigsynth"
        h5.attrs["dataset_format"] = "torchsig_compatible_hdf5"
        h5.attrs["output_format"] = config.dataset.output_format
        h5.attrs["sample_count"] = total_samples
        h5.attrs["sample_rate"] = sample_rate
        h5.attrs["sample_len"] = sample_len
        h5.attrs["config_json"] = json.dumps(config.to_dict(), sort_keys=True)

        data_group = h5.create_group("data")
        metadata_group = h5.create_group("metadata")
        index_group = h5.create_group("index")
        h5.create_group("component_signals")

        # Calculate appropriate chunk size for HDF5
        # For large samples (>1MB), use the full sample as chunk
        # For small samples, use default chunking
        sample_size_mb = (sample_len * 8) / (1024 * 1024)  # complex64 = 8 bytes
        if sample_size_mb >= 1.0:
            # Large samples: chunk = full sample (better for random access)
            chunk_shape = (sample_len,)
        else:
            # Small samples: let HDF5 choose, or use sensible default
            chunk_shape = None  # Auto-chunking

        # Get compression level from config (default 0 = disabled for speed)
        compression_level = getattr(config.dataset, "compression_level", 0)
        compression_level = max(0, min(9, compression_level))  # Clamp to valid range

        # GC when we hit EITHER threshold (more frequent = safer):
        # - At least every 100 samples (avoid excessive overhead for tiny samples)
        # - At most every 8 GB of data (prevent memory buildup for large samples)
        # With 64-128 GB memory limits, 8 GB gives good balance between GC overhead and memory safety
        bytes_per_sample = sample_len * 8  # complex64 = 8 bytes
        gc_interval_bytes = 8 * 1024 * 1024 * 1024  # 8 GB max memory between GCs
        gc_interval_samples = max(100, gc_interval_bytes // max(1, bytes_per_sample))  # At least 100 samples
        bytes_written_since_gc = 0

        for sample_index in range(total_samples):
            sample_id = f"sample_{sample_index:06d}"
            sample = synthesize_sample(config, sample_index)
            sample_data = sample.impaired
            data_group.create_dataset(
                sample_id,
                data=sample_data,
                compression="gzip",
                compression_opts=compression_level,
                shuffle=True,
                fletcher32=True,
                chunks=chunk_shape,
            )

            sample_metadata = metadata_group.create_group(sample_id)
            metadata_values = {
                "sample_id": sample_id,
                "sample_index": sample_index,
                "generator": sample.generator,
                "sample_rate": sample_rate,
                "sample_len": sample_len,
                "output_format": config.dataset.output_format,
                "backend": "fallback_h5py",
                "generators": config.generators,
                "transforms": [step.name for step in config.transforms if step.enabled],
                "global_params": config.global_params,
                "class_list": config.global_params.get("class_list"),
                "class_distribution": config.global_params.get("class_distribution"),
                "num_components": sample.metadata.get("num_components"),
                "components": sample.metadata.get("components", []),
                "impairments": sample.metadata.get("impairments", {}),
            }
            for key, value in metadata_values.items():
                _write_scalar_dataset(sample_metadata, key, value)

            component_group = h5["component_signals"].create_group(sample_id)
            for component_index, component in enumerate(sample.metadata.get("components", [])):
                sub_group = component_group.create_group(f"component_{component_index:03d}")
                for key, value in component.items():
                    _write_scalar_dataset(sub_group, key, value)

            index_group.create_dataset(str(sample_index), data=sample_id, dtype=h5py.string_dtype("utf-8"))

            # Memory management: explicit cleanup after each sample
            del sample_data
            del sample

            bytes_written_since_gc += bytes_per_sample

            # Flush and GC based on data volume (~1 GB) rather than sample count
            # This avoids excessive overhead for small samples and ensures cleanup for large samples
            if bytes_written_since_gc >= gc_interval_bytes or (sample_index + 1) % gc_interval_samples == 0:
                h5.flush()
                gc.collect()
                bytes_written_since_gc = 0
                print(f"HDF5 Progress: {sample_index + 1}/{total_samples} samples ({(sample_index + 1) / total_samples * 100:.1f}%)")

    dataset_info = {
        "format": "torchsig_compatible_hdf5",
        "output_format": config.dataset.output_format,
        "sample_count": total_samples,
        "sample_rate": sample_rate,
        "sample_len": sample_len,
        "generators": config.generators,
        "transforms": [step.name for step in config.transforms if step.enabled],
    }
    with (root / "dataset_info.yaml").open("w", encoding="utf-8") as fp:
        yaml.safe_dump(dataset_info, fp, sort_keys=False)

    writer_info = {
        "root": str(root),
        "overwrite": True,
        "batch_size": config.dataset.create_batch_size,
        "num_workers": config.dataset.create_num_workers,
        "complete": True,
        "backend": "fallback_h5py",
    }
    with (root / "writer_info.yaml").open("w", encoding="utf-8") as fp:
        yaml.safe_dump(writer_info, fp, sort_keys=False)

    write_config_yaml(root, config)
    return datapath


def write_config_yaml(output_dir: str | Path, config: AppConfig) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.yaml"
    with config_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(config.to_dict(), fp, sort_keys=False)
    return config_path
