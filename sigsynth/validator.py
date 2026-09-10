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
VALID_SPLIT_MODES = {"split", "train_only", "val_only"}


def _matching_generator_overrides(
    config: AppConfig,
    generator_name: str,
    canonical_generator_name: str,
) -> list[dict[str, object]]:
    matching: list[dict[str, object]] = []
    seen_keys: set[str] = set()

    for candidate_key in (generator_name, canonical_generator_name):
        override = config.generator_overrides.get(candidate_key)
        if isinstance(override, dict) and candidate_key not in seen_keys:
            matching.append(override)
            seen_keys.add(candidate_key)

    for key, override in config.generator_overrides.items():
        if key in seen_keys or not isinstance(override, dict):
            continue
        if resolve_generator_name(key) == canonical_generator_name:
            matching.append(override)

    return matching


def validate_config(config: AppConfig) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not config.generators:
        errors.append("Select at least one generator.")

    if config.dataset.output_format not in VALID_OUTPUT_FORMATS:
        errors.append(
            f"Unsupported output format '{config.dataset.output_format}'. Choose hdf5 or numpy."
        )

    if config.dataset.split_mode not in VALID_SPLIT_MODES:
        errors.append(
            f"Unsupported split mode '{config.dataset.split_mode}'. Choose split, train_only, or val_only."
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
        matching_overrides = _matching_generator_overrides(
            config,
            generator_name,
            canonical_generator_name,
        )
        for group in meta.parameter_groups:
            has_group = group in config.global_params or any(
                group in override for override in matching_overrides
            )
            if not has_group:
                errors.append(f"Generator '{generator_name}' requires parameter group '{group}'.")

    enabled_transforms = [step for step in config.transforms if step.enabled]
    if config.generators and enabled_transforms:
        current_types = set()
        registered_generator_tags: set[str] = set()
        for name in config.generators:
            canonical_generator_name = resolve_generator_name(name) or name
            generator = GENERATOR_REGISTRY.get(canonical_generator_name)
            if generator:
                current_types.update(generator.produces)
                registered_generator_tags.update(generator.tags)

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
                break

            current_types = set(transform.produces)

            incompatible_tags = set(transform.constraints.get("incompatible_with", []))
            conflicting = incompatible_tags.intersection(registered_generator_tags)
            if conflicting:
                warnings.append(
                    f"Transform '{step.name}' is incompatible with generator tags: {sorted(conflicting)}."
                )

    if config.dataset.total_samples < 1:
        errors.append("Total samples must be >= 1.")
    if config.dataset.output_format == "numpy" and config.dataset.split_mode == "split":
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
