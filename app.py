from __future__ import annotations

from pathlib import Path

import streamlit as st

from sigsynth.generator import build_dataset_zip_bytes, generate_dataset
from sigsynth.macro_manager import MacroManager
from sigsynth.models import AppConfig, DatasetConfig, TransformStep
from sigsynth.paths import sanitize_output_dir
from sigsynth.preview import build_transform_preview, render_preview_figure
from sigsynth.registry import (
    GENERATOR_REGISTRY,
    TORCHSIG_CONCRETE_GENERATORS,
    is_torchsig_concrete_generator,
    TRANSFORM_REGISTRY,
    resolve_generator_name,
    resolve_transform_name,
    to_torchsig_generator_name,
)
from sigsynth.validator import validate_config

st.set_page_config(page_title="TorchSig Dataset Builder", layout="wide")
st.title("TorchSig Dataset Builder")
st.caption("Build and validate RF synthesis macros, including Sig53 and Wideband Sig53 starter presets.")

macro_manager = MacroManager("macros")

if "config" not in st.session_state:
    st.session_state.config = AppConfig(
        generators=["BPSK", "QPSK"],
        global_params={"sample_rate": 1_000_000, "duration": 0.001, "snr_db": [0, 30], "sample_len": 1024},
        transforms=[TransformStep(name="AWGN", enabled=True)],
        dataset=DatasetConfig(total_samples=100, train_ratio=0.8, output_dir="output/dataset"),
    )
if "download_zip" not in st.session_state:
    st.session_state.download_zip = None
if "download_name" not in st.session_state:
    st.session_state.download_name = "dataset.zip"

config: AppConfig = st.session_state.config

FREQUENCY_UNITS = {
    "Hz": 1.0,
    "kHz": 1_000.0,
    "MHz": 1_000_000.0,
    "GHz": 1_000_000_000.0,
}

TORCHSIG_GENERATOR_GROUPS: list[tuple[str, list[str]]] = [
    ("Tones and chirps", ["tone", "chirpss", "lfm-data", "lfm-radar"]),
    ("OFDM", [name for name in TORCHSIG_CONCRETE_GENERATORS if name.startswith("ofdm-")]),
    ("PSK", [name for name in TORCHSIG_CONCRETE_GENERATORS if name.endswith("psk")]),
    ("QAM", [name for name in TORCHSIG_CONCRETE_GENERATORS if "qam" in name]),
    ("FSK / MSK", [name for name in TORCHSIG_CONCRETE_GENERATORS if name.endswith("fsk") or name.endswith("msk")]),
    ("AM / FM / OOK", ["fm", "ook", "am-dsb", "am-dsb-sc", "am-usb", "am-lsb"]),
]


def render_remap_form(
    form_key: str,
    legacy_names: list[str],
    options: list[str],
    labels: dict[str, str],
    suggestion_map: dict[str, str | None],
) -> tuple[dict[str, str], bool]:
    remaps: dict[str, str] = {}
    if not legacy_names:
        return remaps, False

    suggestion_button_key = f"{form_key}::use_suggestions"
    with st.form(form_key):
        st.caption("Pick replacements for unresolved names to migrate older macros.")
        for index, legacy_name in enumerate(legacy_names):
            suggested = suggestion_map.get(legacy_name)
            default_index = 0
            if suggested in options:
                default_index = options.index(suggested) + 1
            remaps[legacy_name] = st.selectbox(
                labels.get(legacy_name, legacy_name),
                options=["<keep unresolved>"] + options,
                index=default_index,
                key=f"{form_key}::{index}::{legacy_name}",
            )
        submitted = st.form_submit_button("Apply remaps")

    if st.button("Use suggestions for all", key=suggestion_button_key, use_container_width=True):
        for index, legacy_name in enumerate(legacy_names):
            suggested = suggestion_map.get(legacy_name)
            target_value = suggested if suggested in options else "<keep unresolved>"
            st.session_state[f"{form_key}::{index}::{legacy_name}"] = target_value
        st.rerun()

    return remaps, submitted


def render_grouped_generator_selector(
    group_specs: list[tuple[str, list[str]]],
    resolved_generators: list[str],
) -> list[str]:
    selected: list[str] = []

    st.caption("Grouped by modulation family so the concrete TorchSig catalog is easier to scan.")
    for group_name, group_options in group_specs:
        available_defaults = [name for name in resolved_generators if name in group_options]
        with st.expander(group_name, expanded=group_name in {"PSK", "QAM"}):
            group_selected = st.multiselect(
                group_name,
                options=group_options,
                default=available_defaults,
                key=f"torchsig_group::{group_name}",
                label_visibility="collapsed",
            )
            selected.extend(group_selected)

    deduped: list[str] = []
    for name in selected:
        if name not in deduped:
            deduped.append(name)
    return deduped


def torchsig_runtime_available() -> bool:
    try:
        from torchsig.utils.defaults import TorchSigDefaults  # noqa: F401
        from torchsig.datasets.datasets import TorchSigIterableDataset  # noqa: F401
        from torchsig.utils.writer import DatasetCreator  # noqa: F401
    except Exception:
        return False
    return True


left, right = st.columns([2, 1])

with right:
    st.subheader("Macros")
    macros = macro_manager.list_macros()
    selected_macro = st.selectbox("Available macros", options=["<none>"] + macros)

    if st.button("Load macro", use_container_width=True) and selected_macro != "<none>":
        st.session_state.config = macro_manager.load(selected_macro)
        st.rerun()

    save_name = st.text_input("Save macro as", value="new_macro.yaml")
    if st.button("Save current macro", use_container_width=True):
        candidate_name = save_name if save_name.endswith(".yaml") else f"{save_name}.yaml"
        try:
            path = macro_manager.save(candidate_name, config)
            st.success(f"Saved {path}")
        except ValueError as exc:
            st.error(str(exc))

    st.markdown("---")
    st.subheader("Dataset")
    if torchsig_runtime_available():
        st.success("Runtime mode: TorchSig backend available for local HDF5 generation.")
    else:
        st.info("Runtime mode: TorchSig is unavailable, so HDF5 output will use the local h5py fallback.")
    output_format_label = st.radio(
        "Output format",
        options=["TorchSig-compatible HDF5", "NumPy split folders"],
        index=0 if config.dataset.output_format == "hdf5" else 1,
        horizontal=True,
    )
    config.dataset.output_format = "hdf5" if output_format_label == "TorchSig-compatible HDF5" else "numpy"
    generator_catalog_default = "TorchSig concrete labels" if any(
        is_torchsig_concrete_generator(name) for name in config.generators
    ) else "Simplified families"
    generator_catalog = st.radio(
        "Generator catalog",
        options=["Simplified families", "TorchSig concrete labels"],
        index=0 if generator_catalog_default == "Simplified families" else 1,
        horizontal=True,
    )
    config.global_params["generator_catalog"] = generator_catalog
    total_samples = st.number_input("Total samples", min_value=2, value=config.dataset.total_samples)
    train_ratio = st.slider("Train ratio", 0.1, 0.95, float(config.dataset.train_ratio), 0.05)
    if config.dataset.output_format == "hdf5":
        st.caption("Train ratio is retained in the config, but it only affects NumPy split-folder output.")
    output_dir = st.text_input("Output directory", value=config.dataset.output_dir)
    config.dataset = DatasetConfig(total_samples=int(total_samples), train_ratio=float(train_ratio), output_dir=output_dir)
    config.dataset.output_format = "hdf5" if output_format_label == "TorchSig-compatible HDF5" else "numpy"

with left:
    st.subheader("1) Generators")
    generator_catalog = config.global_params.get("generator_catalog", "Simplified families")
    resolved_generators = []
    unknown_generators = []
    if generator_catalog == "TorchSig concrete labels":
        for name in config.generators:
            if is_torchsig_concrete_generator(name):
                if name not in resolved_generators:
                    resolved_generators.append(name)
            else:
                mapped_name = to_torchsig_generator_name(name)
                if mapped_name and mapped_name not in resolved_generators:
                    resolved_generators.append(mapped_name)
                elif not mapped_name:
                    unknown_generators.append(name)
        selected_generators = render_grouped_generator_selector(
            group_specs=TORCHSIG_GENERATOR_GROUPS,
            resolved_generators=resolved_generators,
        )
        if unknown_generators:
            st.warning(
                "Macro contains legacy or unknown generator names that need attention: "
                f"{', '.join(unknown_generators)}"
            )
            generator_remaps, generator_remaps_applied = render_remap_form(
                form_key="torchsig_generator_remap_form",
                legacy_names=unknown_generators,
                options=TORCHSIG_CONCRETE_GENERATORS,
                labels={name: f"Replace '{name}' with" for name in unknown_generators},
                suggestion_map={name: to_torchsig_generator_name(name) for name in unknown_generators},
            )
            for legacy_name, replacement in generator_remaps.items():
                if replacement != "<keep unresolved>" and replacement not in selected_generators:
                    selected_generators.append(replacement)
                elif replacement == "<keep unresolved>":
                    selected_generators.append(legacy_name)
            if generator_remaps_applied:
                st.success("Applied generator remaps.")
    else:
        generator_options = sorted(GENERATOR_REGISTRY.keys())
        for name in config.generators:
            canonical_name = resolve_generator_name(name)
            if canonical_name and canonical_name not in resolved_generators:
                resolved_generators.append(canonical_name)
        if unknown_generators:
            st.warning(
                "Macro contains legacy or unknown generator names that need attention: "
                f"{', '.join(unknown_generators)}"
            )
        selected_generators = st.multiselect(
            "Choose signal generators",
            options=generator_options,
            default=resolved_generators,
        )
        generator_suggestions = {
            name: resolve_generator_name(name)
            for name in unknown_generators
        }
        generator_remaps, generator_remaps_applied = render_remap_form(
            form_key="generator_remap_form",
            legacy_names=unknown_generators,
            options=generator_options,
            labels={name: f"Replace '{name}' with" for name in unknown_generators},
            suggestion_map=generator_suggestions,
        )
        remapped_generators = list(selected_generators)
        for legacy_name, replacement in generator_remaps.items():
            if replacement != "<keep unresolved>" and replacement not in remapped_generators:
                remapped_generators.append(replacement)
            elif replacement == "<keep unresolved>":
                remapped_generators.append(legacy_name)

        selected_generators = remapped_generators
        if generator_remaps_applied and unknown_generators:
            st.success("Applied generator remaps.")

    config.generators = selected_generators

    st.subheader("2) Global parameters")
    selected_generator_families = {resolve_generator_name(name) or name for name in config.generators}
    default_frequency_unit = "MHz"
    current_sample_rate = int(config.global_params.get("sample_rate", 1_000_000))
    if current_sample_rate >= 1_000_000_000:
        default_frequency_unit = "GHz"
    elif current_sample_rate >= 1_000_000:
        default_frequency_unit = "MHz"
    elif current_sample_rate >= 1_000:
        default_frequency_unit = "kHz"

    frequency_unit = st.selectbox(
        "Frequency unit",
        options=list(FREQUENCY_UNITS.keys()),
        index=list(FREQUENCY_UNITS.keys()).index(default_frequency_unit),
        key="frequency_unit",
    )
    frequency_scale = FREQUENCY_UNITS[frequency_unit]
    st.caption("Values are displayed in the selected unit but stored internally in Hz.")
    sample_rate_display = current_sample_rate / frequency_scale
    sample_rate_display = st.number_input(
        f"Sample rate ({frequency_unit})",
        min_value=1_000 / frequency_scale,
        value=float(sample_rate_display),
        step=max(0.001, 1_000.0 / frequency_scale),
        format="%.6f",
    )
    sample_rate = int(sample_rate_display * frequency_scale)

    center_frequency_default = float(config.global_params.get("center_frequency_hz", 0))
    center_frequency_limit = sample_rate / 2.0
    center_frequency_default = max(-center_frequency_limit, min(center_frequency_default, center_frequency_limit))
    center_frequency_display = center_frequency_default / frequency_scale
    center_frequency_hz = st.number_input(
        f"Band center frequency ({frequency_unit})",
        min_value=-center_frequency_limit / frequency_scale,
        max_value=center_frequency_limit / frequency_scale,
        value=float(center_frequency_display),
        step=max(0.001, max(1_000.0, sample_rate / 100.0) / frequency_scale),
        format="%.6f",
    )
    center_frequency_hz = int(center_frequency_hz * frequency_scale)
    duration = st.number_input(
        "Duration (sec)", min_value=0.000001, value=float(config.global_params.get("duration", 0.001)), format="%.6f"
    )
    snr_min, snr_max = st.slider(
        "SNR range (dB)",
        min_value=-20,
        max_value=60,
        value=tuple(config.global_params.get("snr_db", [0, 30])),
    )
    sample_len = st.number_input("Sample length", min_value=128, value=int(config.global_params.get("sample_len", 1024)), step=128)

    config.global_params.update(
        {
            "sample_rate": int(sample_rate),
            "center_frequency_hz": int(center_frequency_hz),
            "duration": float(duration),
            "snr_db": [int(snr_min), int(snr_max)],
            "sample_len": int(sample_len),
        }
    )

    if "LFM" in selected_generator_families:
        st.info("LFM selected: chirp parameters are required.")
        sweep_hz = st.number_input("LFM sweep bandwidth (Hz)", min_value=1_000, value=50_000)
        config.generator_overrides.setdefault("LFM", {})["chirp"] = {"sweep_hz": int(sweep_hz)}

    st.subheader("3) Transform pipeline")
    transform_options = sorted(TRANSFORM_REGISTRY.keys())
    resolved_transforms = []
    unknown_transforms = []
    for step in config.transforms:
        if not step.enabled:
            continue
        canonical_name = resolve_transform_name(step.name)
        if canonical_name:
            if canonical_name not in resolved_transforms:
                resolved_transforms.append(canonical_name)
        else:
            unknown_transforms.append(step.name)

    if unknown_transforms:
        st.warning(
            "Macro contains legacy or unknown transform names that need attention: "
            f"{', '.join(unknown_transforms)}"
        )
    enabled_transforms = st.multiselect(
        "Enable transforms in order",
        options=transform_options,
        default=resolved_transforms,
    )
    transform_suggestions = {
        name: resolve_transform_name(name)
        for name in unknown_transforms
    }
    transform_remaps, transform_remaps_applied = render_remap_form(
        form_key="transform_remap_form",
        legacy_names=unknown_transforms,
        options=transform_options,
        labels={name: f"Replace transform '{name}' with" for name in unknown_transforms},
        suggestion_map=transform_suggestions,
    )
    remapped_transforms = list(enabled_transforms)
    for legacy_name, replacement in transform_remaps.items():
        if replacement != "<keep unresolved>" and replacement not in remapped_transforms:
            remapped_transforms.append(replacement)
        elif replacement == "<keep unresolved>":
            remapped_transforms.append(legacy_name)

    config.transforms = [TransformStep(name=name, enabled=True) for name in remapped_transforms]
    if transform_remaps_applied and unknown_transforms:
        st.success("Applied transform remaps.")

    st.subheader("4) Transform preview")
    preview_transforms = [step.name for step in config.transforms if step.enabled and step.name in TRANSFORM_REGISTRY]
    if preview_transforms:
        preview_stages = build_transform_preview(config, preview_transforms)
        st.caption(
            "Approximate preview built from a synthetic sample. This uses the selected band center so the preview matches the exported RF placement."
        )
        with st.expander("Show preview", expanded=True):
            st.pyplot(render_preview_figure(preview_stages), clear_figure=True)
    else:
        st.info("Add at least one recognized transform to see a live preview.")

output_dir_error = None
safe_output_dir = None
try:
    safe_output_dir = sanitize_output_dir(config.dataset.output_dir)
except ValueError as exc:
    output_dir_error = str(exc)

errors, warnings = validate_config(config)
train_count = int(config.dataset.total_samples * config.dataset.train_ratio)
val_count = config.dataset.total_samples - train_count
split_has_zero_partition = train_count == 0 or val_count == 0
if config.dataset.output_format != "numpy":
    split_has_zero_partition = False
output_dir_non_empty = False
if safe_output_dir is not None:
    output_dir_non_empty = safe_output_dir.exists() and safe_output_dir.is_dir() and any(safe_output_dir.iterdir())

st.subheader("Validation")
if warnings:
    for item in warnings:
        st.warning(item)
if errors:
    for item in errors:
        st.error(item)
if output_dir_error:
    st.error(output_dir_error)
if not errors and not output_dir_error:
    st.success("Configuration is valid.")

confirm_non_empty_output = True
if output_dir_non_empty:
    st.warning(
        "Output directory is not empty. Files that are not overwritten can taint your dataset unexpectedly."
    )
    confirm_non_empty_output = st.checkbox(
        "I understand and want to generate into this non-empty directory.",
        value=False,
    )

confirm_zero_split = True
if split_has_zero_partition:
    confirm_zero_split = st.checkbox(
        f"I understand the split creates a zero-sized partition (train={train_count}, val={val_count}) and want to continue.",
        value=False,
    )

generate_disabled = bool(errors) or bool(output_dir_error) or not confirm_non_empty_output or not confirm_zero_split
if st.button("Generate dataset", type="primary", disabled=generate_disabled):
    results = generate_dataset(config)
    st.session_state.download_zip = build_dataset_zip_bytes(results["output_dir"])
    st.session_state.download_name = f"{Path(results['output_dir']).name or 'dataset'}.zip"
    if results["output_format"] == "hdf5":
        st.success(
            "Generated TorchSig-compatible HDF5 dataset at "
            f"{results['output_dir']} with {config.dataset.total_samples} samples."
        )
        if results["torchsig_generated"]:
            st.success("TorchSig generation backend was used.")
        else:
            st.info(
                "TorchSig was unavailable, so the app wrote a compatible data.h5 file with h5py instead."
                f" Reason: {results['torchsig_error'] or 'unknown'}"
            )
    else:
        st.success(
            "Generated NumPy split-folder dataset at "
            f"{results['output_dir']} (train={results['train_samples']}, val={results['val_samples']})."
        )

if st.session_state.download_zip:
    st.download_button(
        "Download generated dataset (.zip)",
        data=st.session_state.download_zip,
        file_name=st.session_state.download_name,
        mime="application/zip",
        use_container_width=True,
    )

st.markdown("---")
st.subheader("Current Config Snapshot")
st.json(config.to_dict())

if Path("README.md").exists():
    with st.expander("Project spec excerpt"):
        st.write("See README.md for full functional specification and phased roadmap.")
