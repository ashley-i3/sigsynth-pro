from __future__ import annotations

import gc
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from sigsynth.models import AppConfig
from sigsynth.numpy_synth import synthesize_sample
from sigsynth.registry import resolve_generator_name


def _metadata_component_list(sample_metadata: dict[str, object]) -> list[dict[str, object]]:
    components = sample_metadata.get("components", [])
    if isinstance(components, list):
        return [component for component in components if isinstance(component, dict)]
    return []


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
        try:
            array_value = np.asarray(value)
        except ValueError:
            array_value = np.asarray(value, dtype=object)
        if array_value.dtype == object:
            serialized = json.dumps(list(value), sort_keys=True, default=str)
            group.create_dataset(key, data=serialized, dtype=h5py.string_dtype("utf-8"))
            return
        group.create_dataset(key, data=array_value)
        return

    if isinstance(value, dict):
        serialized = json.dumps(value, sort_keys=True)
        group.create_dataset(key, data=serialized, dtype=h5py.string_dtype("utf-8"))
        return

    group.create_dataset(key, data=str(value), dtype=h5py.string_dtype("utf-8"))


def _signal_component_ids(sample_id: str, component_count: int) -> list[str]:
    return [f"{sample_id}__component_{index:03d}" for index in range(component_count)]


def _coerce_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, np.generic):
        return float(value.item())
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _build_dataset_metadata_entry(config: AppConfig) -> dict[str, object]:
    try:
        from sigsynth.generator import _build_torchsig_metadata

        metadata = dict(_build_torchsig_metadata(config))
    except Exception:
        sample_rate = int(config.global_params.get("sample_rate", 1_000_000))
        sample_len = int(config.global_params.get("sample_len", 1024))
        frequency_limit = max(1, sample_rate // 2 - 1)
        snr_db = config.global_params.get("snr_db", [0, 30])
        if isinstance(snr_db, (list, tuple)) and len(snr_db) >= 2:
            snr_db_min = float(snr_db[0])
            snr_db_max = float(snr_db[1])
        else:
            snr_db_min = 0.0
            snr_db_max = 30.0
        metadata = {
            "num_iq_samples_dataset": sample_len,
            "num_signals_min": int(config.global_params.get("num_signals_min", 1)),
            "num_signals_max": int(config.global_params.get("num_signals_max", 1)),
            "fft_size": min(512, sample_len),
            "fft_stride": min(512, sample_len),
            "sample_rate": sample_rate,
            "noise_power_db": 0.0,
            "snr_db_min": snr_db_min,
            "snr_db_max": snr_db_max,
            "cochannel_overlap_probability": float(config.global_params.get("cochannel_overlap_probability", 0.2)),
            "signal_duration_in_samples_min": sample_len,
            "signal_duration_in_samples_max": sample_len,
            "bandwidth_min": min(max(1, sample_rate // 32), frequency_limit),
            "bandwidth_max": frequency_limit,
            "signal_center_freq_min": -int(sample_rate * float(config.global_params.get("signal_center_freq_range_factor", 0.16))),
            "signal_center_freq_max": int(sample_rate * float(config.global_params.get("signal_center_freq_range_factor", 0.16))),
            "frequency_min": -frequency_limit,
            "frequency_max": frequency_limit,
            "class_list": config.global_params.get("class_list", "all"),
            "class_distribution": config.global_params.get("class_distribution", "uniform"),
        }

    metadata["class_names"] = list(
        dict.fromkeys(str(generator).lower() for generator in config.generators if generator)
    )
    return metadata


def _build_generator_metadata_entry(generator_name: str, class_index: int, dataset_metadata_id: str) -> dict[str, object]:
    class_name = str(generator_name).lower()
    metadata: dict[str, object] = {
        "class_index": int(class_index),
        "class_name": class_name,
        "parent_metadata_id": dataset_metadata_id,
    }

    canonical_name = resolve_generator_name(generator_name) or generator_name
    if canonical_name in {"BPSK", "QPSK", "8PSK", "PSK", "QAM16", "QAM64", "QAM", "ASK", "OOK"}:
        metadata["constellation_name"] = class_name
    elif class_name.startswith("ofdm-"):
        try:
            metadata["num_subcarriers"] = int(class_name.split("-", 1)[1])
        except ValueError:
            pass
    elif class_name.startswith("lfm-"):
        metadata["lfm_type"] = class_name.split("-", 1)[1]
    elif class_name.startswith("am-"):
        metadata["am_mode"] = class_name.split("-", 1)[1]
    else:
        match = None
        try:
            import re

            match = re.fullmatch(r"(\d+)(g?fsk|g?msk)", class_name)
        except Exception:
            match = None
        if match:
            metadata["constellation_size"] = int(match.group(1))
            metadata["fsk_type"] = match.group(2)

    return metadata


def _write_signal_entry(
    data_group: h5py.Group,
    metadata_group: h5py.Group,
    component_group: h5py.Group,
    signal_id: str,
    signal_data: np.ndarray,
    metadata_values: dict[str, object],
    child_signal_ids: list[str],
    *,
    compression_name: str | None,
    compression_opts: int | None,
    shuffle_enabled: bool,
    chunk_shape: tuple[int, ...] | None,
) -> None:
    data_group.create_dataset(
        signal_id,
        data=signal_data,
        compression=compression_name,
        compression_opts=compression_opts,
        shuffle=shuffle_enabled,
        fletcher32=True,
        chunks=chunk_shape,
    )

    signal_metadata = metadata_group.create_group(signal_id)
    for key, value in metadata_values.items():
        _write_scalar_dataset(signal_metadata, key, value)

    if child_signal_ids:
        component_group.create_dataset(
            signal_id,
            data=np.asarray(child_signal_ids, dtype=h5py.string_dtype("utf-8")),
        )


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
        try:
            from torchsig import __version__ as torchsig_version  # type: ignore
        except Exception:
            torchsig_version = None

        h5.attrs["created_by"] = "sigsynth"
        h5.attrs["dataset_format"] = "torchsig_compatible_hdf5"
        h5.attrs["output_format"] = config.dataset.output_format
        h5.attrs["sample_count"] = total_samples
        h5.attrs["sample_rate"] = sample_rate
        h5.attrs["sample_len"] = sample_len
        h5.attrs["config_json"] = json.dumps(config.to_dict(), sort_keys=True)
        h5.attrs["compression"] = "gzip" if int(getattr(config.dataset, "compression_level", 0)) > 0 else "none"
        if torchsig_version is not None:
            h5.attrs["torchsig_version"] = str(torchsig_version)

        data_group = h5.create_group("data")
        metadata_group = h5.create_group("metadata")
        index_group = h5.create_group("index")
        h5.create_group("component_signals")

        dataset_metadata_id = "__dataset_metadata__"
        dataset_metadata_values = _build_dataset_metadata_entry(config)
        metadata_group.create_group(dataset_metadata_id)
        for key, value in dataset_metadata_values.items():
            _write_scalar_dataset(metadata_group[dataset_metadata_id], key, value)

        generator_metadata_ids: dict[str, str] = {}
        for class_index, generator_name in enumerate(dict.fromkeys(config.generators)):
            generator_metadata_id = f"__generator_metadata__::{str(generator_name).lower()}"
            generator_metadata_ids[str(generator_name).lower()] = generator_metadata_id
            metadata_group.create_group(generator_metadata_id)
            for key, value in _build_generator_metadata_entry(
                str(generator_name),
                class_index,
                dataset_metadata_id,
            ).items():
                _write_scalar_dataset(metadata_group[generator_metadata_id], key, value)

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
        compression_enabled = compression_level > 0
        compression_name = "gzip" if compression_enabled else None
        compression_opts = compression_level if compression_enabled else None
        shuffle_enabled = compression_enabled

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
            component_metadata = _metadata_component_list(sample.metadata)
            component_signal_ids = _signal_component_ids(sample_id, len(sample.component_signals))
            sample_bandwidth = max(
                [0.0]
                + [
                    _coerce_float(component.get("bandwidth", 0.0))
                    for component in component_metadata
                ]
            )
            metadata_values = {
                "sample_id": sample_id,
                "sample_index": sample_index,
                "parent_metadata_id": dataset_metadata_id,
                "generator": sample.generator,
                "sample_rate": sample_rate,
                "sample_len": sample_len,
                "num_iq_samples_dataset": sample_len,
                "dataset_length": total_samples,
                "output_format": config.dataset.output_format,
                "backend": "fallback_h5py",
                "generators": config.generators,
                "transforms": [step.name for step in config.transforms if step.enabled],
                "global_params": config.global_params,
                "class_list": config.global_params.get("class_list"),
                "class_distribution": config.global_params.get("class_distribution"),
                "num_components": sample.metadata.get("num_components"),
                "num_signals_min": int(config.global_params.get("num_signals_min", 1)),
                "num_signals_max": int(config.global_params.get("num_signals_max", max(1, len(component_metadata)))),
                "center_freq": 0.0,
                "bandwidth": sample_bandwidth,
                "duration_in_samples": sample_len,
                "components": component_metadata,
                "impairments": sample.metadata.get("impairments", {}),
            }
            _write_signal_entry(
                data_group,
                metadata_group,
                h5["component_signals"],
                sample_id,
                sample_data,
                metadata_values,
                component_signal_ids,
                compression_name=compression_name,
                compression_opts=compression_opts,
                shuffle_enabled=shuffle_enabled,
                chunk_shape=chunk_shape,
            )

            for component_id, component_signal in zip(component_signal_ids, sample.component_signals):
                component_metadata_values = dict(component_signal.metadata)
                component_metadata_values["parent_metadata_id"] = generator_metadata_ids.get(
                    str(component_metadata_values.get("class_name", "")).lower(),
                    dataset_metadata_id,
                )
                _write_signal_entry(
                    data_group,
                    metadata_group,
                    h5["component_signals"],
                    component_id,
                    component_signal.data,
                    component_metadata_values,
                    [],
                    compression_name=compression_name,
                    compression_opts=compression_opts,
                    shuffle_enabled=shuffle_enabled,
                    chunk_shape=chunk_shape,
                )

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
