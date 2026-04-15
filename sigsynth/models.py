from __future__ import annotations

from dataclasses import dataclass, field
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
        transform_objs = [TransformStep(**step) for step in payload.get("transforms", [])]
        dataset = DatasetConfig(**payload.get("dataset", {}))
        return cls(
            schema_version=payload.get("schema_version", "1.0"),
            generators=payload.get("generators", []),
            global_params=payload.get("global_params", {}),
            generator_overrides=payload.get("generator_overrides", {}),
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
