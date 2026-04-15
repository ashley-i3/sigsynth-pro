from __future__ import annotations

from sigsynth.models import AppConfig
from sigsynth.registry import (
    GENERATOR_REGISTRY,
    TRANSFORM_REGISTRY,
    resolve_generator_name,
    resolve_transform_name,
)


REQUIRED_GLOBALS = {"sample_rate", "duration", "snr_db"}
VALID_OUTPUT_FORMATS = {"hdf5", "numpy"}


def validate_config(config: AppConfig) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not config.generators:
        errors.append("Select at least one generator.")

    if config.dataset.output_format not in VALID_OUTPUT_FORMATS:
        errors.append(
            f"Unsupported output format '{config.dataset.output_format}'. Choose hdf5 or numpy."
        )

    missing_globals = REQUIRED_GLOBALS - set(config.global_params.keys())
    if missing_globals:
        errors.append(f"Missing required global parameters: {', '.join(sorted(missing_globals))}")

    for generator_name in config.generators:
        canonical_generator_name = resolve_generator_name(generator_name) or generator_name
        meta = GENERATOR_REGISTRY.get(canonical_generator_name)
        if not meta:
            errors.append(f"Generator '{generator_name}' is not registered.")
            continue
        for group in meta.parameter_groups:
            has_group = group in config.global_params or group in config.generator_overrides.get(canonical_generator_name, {})
            if not has_group:
                errors.append(f"Generator '{generator_name}' requires parameter group '{group}'.")

    enabled_transforms = [step for step in config.transforms if step.enabled]
    if config.generators and enabled_transforms:
        current_types = set()
        for name in config.generators:
            canonical_generator_name = resolve_generator_name(name) or name
            generator = GENERATOR_REGISTRY.get(canonical_generator_name)
            if generator:
                current_types.update(generator.produces)

        generator_tags = {
            tag
            for name in config.generators
            for tag in GENERATOR_REGISTRY.get(
                resolve_generator_name(name) or name, GENERATOR_REGISTRY["BPSK"]
            ).tags
        }

        for step in enabled_transforms:
            canonical_transform_name = resolve_transform_name(step.name) or step.name
            transform = TRANSFORM_REGISTRY.get(canonical_transform_name)
            if not transform:
                errors.append(f"Transform '{step.name}' is not registered.")
                continue

            if not current_types.intersection(transform.accepts):
                errors.append(
                    f"Transform '{step.name}' expects {transform.accepts} but pipeline currently has {sorted(current_types)}."
                )
            else:
                current_types = set(transform.produces)

            incompatible_tags = set(transform.constraints.get("incompatible_with", []))
            conflicting = incompatible_tags.intersection(generator_tags)
            if conflicting:
                warnings.append(
                    f"Transform '{step.name}' is incompatible with generator tags: {sorted(conflicting)}."
                )

    if config.dataset.total_samples < 2:
        errors.append("Total samples must be >= 2.")
    if config.dataset.output_format == "numpy":
        if not 0.0 < config.dataset.train_ratio < 1.0:
            errors.append("Train ratio must be between 0 and 1 for NumPy output.")
        else:
            train_count = int(config.dataset.total_samples * config.dataset.train_ratio)
            val_count = config.dataset.total_samples - train_count
            if train_count == 0 or val_count == 0:
                warnings.append(
                    "Train/validation split produces a zero-sized partition "
                    f"(train={train_count}, val={val_count})."
                )

    return errors, warnings
