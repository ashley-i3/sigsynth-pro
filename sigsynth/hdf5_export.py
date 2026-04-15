from __future__ import annotations

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

    with h5py.File(datapath, "w", libver="latest") as h5:
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

        for sample_index in range(total_samples):
            sample_id = f"sample_{sample_index:06d}"
            sample = synthesize_sample(config, sample_index)
            sample_data = sample.impaired
            data_group.create_dataset(
                sample_id,
                data=sample_data,
                compression="gzip",
                compression_opts=6,
                shuffle=True,
                fletcher32=True,
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
                "burst": sample.metadata.get("burst", {}),
                "impairments": sample.metadata.get("impairments", {}),
            }
            for key, value in metadata_values.items():
                _write_scalar_dataset(sample_metadata, key, value)

            index_group.create_dataset(str(sample_index), data=sample_id, dtype=h5py.string_dtype("utf-8"))

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
        "batch_size": 1,
        "num_workers": 0,
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
