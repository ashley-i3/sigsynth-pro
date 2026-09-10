from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class GeneratorMeta:
    name: str
    produces: list[str]
    requires: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    parameter_groups: list[str] = field(default_factory=list)


@dataclass
class TransformMeta:
    name: str
    accepts: list[str]
    produces: list[str]
    modifies: list[str] = field(default_factory=list)
    constraints: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class TransformStep:
    name: str
    enabled: bool = True


@dataclass
class DatasetConfig:
    total_samples: int = 1000
    train_ratio: float = 0.8
    output_dir: str = "output/dataset"
    output_format: str = "hdf5"
    split_mode: str = "split"
    create_batch_size: int = 256
    create_num_workers: int = 0  # HDF5 writing is not thread-safe, use single process
    max_memory_mb: int | None = None
    compression_level: int = 0  # gzip compression (0-9). 0=disabled for speed, enable for space savings


@dataclass
class AppConfig:
    schema_version: str = "1.0"
    generators: list[str] = field(default_factory=list)
    global_params: dict[str, Any] = field(default_factory=dict)
    generator_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    transforms: list[TransformStep] = field(default_factory=list)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    @property
    def output_path(self) -> Path:
        return Path(self.dataset.output_dir)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AppConfig":
        transforms_payload = payload.get("transforms", [])
        transform_field_names = {field.name for field in fields(TransformStep)}
        transform_objs: list[TransformStep] = []
        for step in transforms_payload if isinstance(transforms_payload, list) else []:
            if isinstance(step, str):
                transform_objs.append(TransformStep(name=step))
                continue
            if isinstance(step, dict):
                filtered_step = {
                    key: value for key, value in step.items() if key in transform_field_names
                }
                if "name" in filtered_step:
                    transform_objs.append(TransformStep(**filtered_step))

        dataset_payload = payload.get("dataset", {})
        dataset_field_names = {field.name for field in fields(DatasetConfig)}
        filtered_dataset = {
            key: value for key, value in dataset_payload.items() if key in dataset_field_names
        } if isinstance(dataset_payload, dict) else {}
        dataset = DatasetConfig(**filtered_dataset)

        generators_payload = payload.get("generators", [])
        generator_overrides_payload = payload.get("generator_overrides", {})
        global_params_payload = payload.get("global_params", {})
        return cls(
            schema_version=payload.get("schema_version", "1.0"),
            generators=generators_payload if isinstance(generators_payload, list) else [],
            global_params=global_params_payload if isinstance(global_params_payload, dict) else {},
            generator_overrides=(
                generator_overrides_payload if isinstance(generator_overrides_payload, dict) else {}
            ),
            transforms=transform_objs,
            dataset=dataset,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generators": self.generators,
            "global_params": self.global_params,
            "generator_overrides": self.generator_overrides,
            "transforms": [step.__dict__ for step in self.transforms],
            "dataset": self.dataset.__dict__,
        }
