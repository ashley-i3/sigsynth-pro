from __future__ import annotations

from copy import deepcopy
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
        global_params={
            "sample_rate": 1_000_000,
            "duration": 0.001024,
            "snr_db": [0, 30],
            "sample_len": 1024,
            "seed": 1234567890,
        },
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

SEED_PRESETS = {
    "Custom": None,
    "Original train seed (1234567890)": 1234567890,
    "Original val seed (1234567891)": 1234567891,
    "Original impaired train seed (1234567892)": 1234567892,
    "Original impaired val seed (1234567893)": 1234567893,
}

DATASET_PRESETS = {
    "Custom": None,
    "Sig53 clean train": {
        "seed": 1234567890,
        "total_samples": 1_060_000,
        "split_mode": "train_only",
        "sample_rate": 10_000_000,
        "sample_len": 4096,
        "duration": 0.0004096,
        "snr_db": [100, 100],
        "center_frequency_hz": 0,
        "class_list": "all",
        "class_distribution": "uniform",
        "num_signals_min": 1,
        "num_signals_max": 1,
        "impairment_level": 0,
        "eb_no": False,
        "output_format": "hdf5",
        "generator_catalog": "TorchSig concrete labels",
    },
    "Sig53 clean val": {
        "seed": 1234567891,
        "total_samples": 106_000,
        "split_mode": "val_only",
        "sample_rate": 10_000_000,
        "sample_len": 4096,
        "duration": 0.0004096,
        "snr_db": [100, 100],
        "center_frequency_hz": 0,
        "class_list": "all",
        "class_distribution": "uniform",
        "num_signals_min": 1,
        "num_signals_max": 1,
        "impairment_level": 0,
        "eb_no": False,
        "output_format": "hdf5",
        "generator_catalog": "TorchSig concrete labels",
    },
    "Sig53 impaired train": {
        "seed": 1234567892,
        "total_samples": 5_300_000,
        "split_mode": "train_only",
        "sample_rate": 10_000_000,
        "sample_len": 4096,
        "duration": 0.0004096,
        "snr_db": [-2, 30],
        "center_frequency_hz": 0,
        "class_list": "all",
        "class_distribution": "uniform",
        "num_signals_min": 1,
        "num_signals_max": 1,
        "impairment_level": 2,
        "eb_no": False,
        "output_format": "hdf5",
        "generator_catalog": "TorchSig concrete labels",
    },
    "Sig53 impaired val": {
        "seed": 1234567893,
        "total_samples": 106_000,
        "split_mode": "val_only",
        "sample_rate": 10_000_000,
        "sample_len": 4096,
        "duration": 0.0004096,
        "snr_db": [-2, 30],
        "center_frequency_hz": 0,
        "class_list": "all",
        "class_distribution": "uniform",
        "num_signals_min": 1,
        "num_signals_max": 1,
        "impairment_level": 2,
        "eb_no": False,
        "output_format": "hdf5",
        "generator_catalog": "TorchSig concrete labels",
    },
    # EbNo variants removed - TorchSig v2.1.0 does not expose eb_no parameter
    # The v0.1.0 API supported eb_no to control Eb/No vs Es/No SNR definition,
    # but this is not available in the v2.1.0 metadata-driven API.
    # All datasets now use Es/No (energy per symbol), which matches canonical Sig53.
    "Wideband clean train": {
        "seed": 1234567890,
        "total_samples": 250_000,
        "split_mode": "train_only",
        "sample_rate": 100_000_000,
        "sample_len": 262_144,
        "duration": 0.00262144,
        "snr_db": [100, 100],
        "center_frequency_hz": 0,
        "class_list": "all",
        "class_distribution": "uniform",
        "num_signals_min": 3,
        "num_signals_max": 5,
        "cochannel_overlap_probability": 0.1,
        "impairment_level": 0,
        "eb_no": False,
        "output_format": "hdf5",
        "generator_catalog": "TorchSig concrete labels",
    },
    "Wideband clean val": {
        "seed": 1234567891,
        "total_samples": 25_000,
        "split_mode": "val_only",
        "sample_rate": 100_000_000,
        "sample_len": 262_144,
        "duration": 0.00262144,
        "snr_db": [100, 100],
        "center_frequency_hz": 0,
        "class_list": "all",
        "class_distribution": "uniform",
        "num_signals_min": 3,
        "num_signals_max": 5,
        "cochannel_overlap_probability": 0.1,
        "impairment_level": 0,
        "eb_no": False,
        "output_format": "hdf5",
        "generator_catalog": "TorchSig concrete labels",
    },
    "Wideband impaired train": {
        "seed": 1234567892,
        "total_samples": 250_000,
        "split_mode": "train_only",
        "sample_rate": 100_000_000,
        "sample_len": 262_144,
        "duration": 0.00262144,
        "snr_db": [-2, 30],
        "center_frequency_hz": 0,
        "class_list": "all",
        "class_distribution": "uniform",
        "num_signals_min": 3,
        "num_signals_max": 5,
        "cochannel_overlap_probability": 0.1,
        "impairment_level": 2,
        "eb_no": False,
        "output_format": "hdf5",
        "generator_catalog": "TorchSig concrete labels",
    },
    "Wideband impaired val": {
        "seed": 1234567893,
        "total_samples": 25_000,
        "split_mode": "val_only",
        "sample_rate": 100_000_000,
        "sample_len": 262_144,
        "duration": 0.00262144,
        "snr_db": [-2, 30],
        "center_frequency_hz": 0,
        "class_list": "all",
        "class_distribution": "uniform",
        "num_signals_min": 3,
        "num_signals_max": 5,
        "cochannel_overlap_probability": 0.1,
        "impairment_level": 2,
        "eb_no": False,
        "output_format": "hdf5",
        "generator_catalog": "TorchSig concrete labels",
    },
}

DATASET_PRESET_PAIRS = {
    "Sig53 clean train": ("Sig53 clean val", "sig53_clean"),
    "Sig53 clean val": ("Sig53 clean train", "sig53_clean"),
    "Sig53 impaired train": ("Sig53 impaired val", "sig53_impaired"),
    "Sig53 impaired val": ("Sig53 impaired train", "sig53_impaired"),
    # EbNo variants removed from pairs - TorchSig v2.1.0 doesn't support eb_no parameter
    "Wideband clean train": ("Wideband clean val", "wideband_clean"),
    "Wideband clean val": ("Wideband clean train", "wideband_clean"),
    "Wideband impaired train": ("Wideband impaired val", "wideband_impaired"),
    "Wideband impaired val": ("Wideband impaired train", "wideband_impaired"),
}

OFFICIAL_DATASET_FAMILIES = {
    "Sig53 official set": [
        "Sig53 clean train",
        "Sig53 clean val",
        "Sig53 impaired train",
        "Sig53 impaired val",
        # Note: EbNo variants removed - TorchSig v2.1.0 does not expose eb_no parameter
        # The eb_no distinction (Eb/No vs Es/No) was available in v0.1.0 but not in v2.1.0
        # All datasets now use Es/No (energy per symbol), which matches canonical Sig53
    ],
    "Wideband Sig53 official set": [
        "Wideband clean train",
        "Wideband clean val",
        "Wideband impaired train",
        "Wideband impaired val",
    ],
}

OFFICIAL_DATASET_FAMILY_SLUGS = {
    "Sig53 official set": "sig53_official",
    "Wideband Sig53 official set": "wideband_sig53_official",
}

LENGTH_WIDGET_KEYS = {
    "sample_rate_hz": "sample_rate_hz",
    "sample_rate_display": "sample_rate_input",
    "duration": "duration_input",
    "sample_len": "sample_len_input",
}

TORCHSIG_GENERATOR_GROUPS: list[tuple[str, list[str]]] = [
    ("Tones and chirps", ["tone", "chirpss", "lfm-data", "lfm-radar"]),
    ("OFDM", [name for name in TORCHSIG_CONCRETE_GENERATORS if name.startswith("ofdm-")]),
    ("PSK", [name for name in TORCHSIG_CONCRETE_GENERATORS if name.endswith("psk")]),
    ("ASK", [name for name in TORCHSIG_CONCRETE_GENERATORS if name.endswith("ask")]),
    ("QAM", [name for name in TORCHSIG_CONCRETE_GENERATORS if "qam" in name]),
    ("FSK / MSK", [name for name in TORCHSIG_CONCRETE_GENERATORS if name.endswith("fsk") or name.endswith("msk")]),
    ("AM / FM / OOK", ["fm", "ook", "am-dsb", "am-dsb-sc", "am-usb", "am-lsb"]),
]


def apply_dataset_preset(preset_name: str) -> None:
    preset = DATASET_PRESETS.get(preset_name)
    if not preset:
        config.global_params["dataset_preset"] = "Custom"
        return

    config.global_params.update(
        {
            "dataset_preset": preset_name,
            "seed": preset["seed"],
            "sample_rate": preset["sample_rate"],
            "center_frequency_hz": preset["center_frequency_hz"],
            "duration": preset["duration"],
            "snr_db": preset["snr_db"],
            "sample_len": preset["sample_len"],
            "class_list": preset["class_list"],
            "class_distribution": preset["class_distribution"],
            "num_signals_min": preset["num_signals_min"],
            "num_signals_max": preset["num_signals_max"],
            "impairment_level": preset["impairment_level"],
            "eb_no": preset["eb_no"],
            "generator_catalog": preset["generator_catalog"],
        }
    )
    if "cochannel_overlap_probability" in preset:
        config.global_params["cochannel_overlap_probability"] = preset["cochannel_overlap_probability"]
    config.dataset.total_samples = preset["total_samples"]
    config.dataset.split_mode = preset["split_mode"]
    config.dataset.output_format = preset["output_format"]
    seed_length_widget_state(config, force=True)


def build_config_for_dataset_preset(preset_name: str, output_dir: Path, test_mode: bool = False) -> AppConfig:
    preset = DATASET_PRESETS[preset_name]
    preset_config = deepcopy(st.session_state.config)
    preset_config.global_params.update(
        {
            "dataset_preset": preset_name,
            "seed": preset["seed"],
            "sample_rate": preset["sample_rate"],
            "center_frequency_hz": preset["center_frequency_hz"],
            "duration": preset["duration"],
            "snr_db": preset["snr_db"],
            "sample_len": preset["sample_len"],
            "class_list": preset["class_list"],
            "class_distribution": preset["class_distribution"],
            "num_signals_min": preset["num_signals_min"],
            "num_signals_max": preset["num_signals_max"],
            "impairment_level": preset["impairment_level"],
            "eb_no": preset["eb_no"],
            "generator_catalog": preset["generator_catalog"],
        }
    )
    if "cochannel_overlap_probability" in preset:
        preset_config.global_params["cochannel_overlap_probability"] = preset["cochannel_overlap_probability"]
    preset_config.generators = list(config.generators)
    preset_config.generator_overrides = deepcopy(config.generator_overrides)
    preset_config.transforms = deepcopy(config.transforms)

    # Apply test mode reduction if requested
    total_samples = preset["total_samples"]
    if test_mode:
        total_samples = max(100, int(total_samples * 0.1))

    preset_config.dataset = DatasetConfig(
        total_samples=total_samples,
        train_ratio=config.dataset.train_ratio,
        output_dir=str(output_dir),
        output_format=preset["output_format"],
        split_mode=preset["split_mode"],
        create_batch_size=config.dataset.create_batch_size,
        create_num_workers=config.dataset.create_num_workers,
        max_memory_mb=config.dataset.max_memory_mb,
        compression_level=getattr(config.dataset, "compression_level", 6),
    )
    return preset_config


def _default_frequency_unit(sample_rate_hz: int) -> str:
    if sample_rate_hz >= 1_000_000_000:
        return "GHz"
    if sample_rate_hz >= 1_000_000:
        return "MHz"
    if sample_rate_hz >= 1_000:
        return "kHz"
    return "Hz"


def _frequency_scale_from_session() -> float:
    frequency_unit = st.session_state.get("frequency_unit", "Hz")
    return FREQUENCY_UNITS.get(frequency_unit, 1.0)


def _format_storage_size(gb: float) -> str:
    """Format storage size with appropriate units (MB or GB)."""
    if gb < 0.1:  # Less than 100 MB
        return f"{gb * 1024:.0f} MB"
    elif gb < 1.0:  # Less than 1 GB
        return f"{gb * 1024:.1f} MB"
    elif gb < 10.0:  # 1-10 GB
        return f"{gb:.1f} GB"
    else:  # 10+ GB
        return f"{gb:.0f} GB"


def seed_length_widget_state(config_obj: AppConfig, force: bool = False) -> None:
    sample_rate_hz = int(config_obj.global_params.get("sample_rate", 1_000_000))
    frequency_unit = _default_frequency_unit(sample_rate_hz)
    frequency_scale = FREQUENCY_UNITS[frequency_unit]
    defaults = {
        "frequency_unit": frequency_unit,
        LENGTH_WIDGET_KEYS["sample_rate_hz"]: sample_rate_hz,
        LENGTH_WIDGET_KEYS["sample_rate_display"]: sample_rate_hz / frequency_scale,
        LENGTH_WIDGET_KEYS["duration"]: float(config_obj.global_params.get("duration", 0.001024)),
        LENGTH_WIDGET_KEYS["sample_len"]: int(config_obj.global_params.get("sample_len", 1024)),
    }
    for key, value in defaults.items():
        if force or key not in st.session_state:
            st.session_state[key] = value
    if force or "length_sync_focus" not in st.session_state:
        st.session_state["length_sync_focus"] = "duration"


def sync_length_widget_state(source: str) -> None:
    sample_rate = max(1.0, float(st.session_state.get(LENGTH_WIDGET_KEYS["sample_rate_hz"], 1_000_000)))
    duration = max(1e-9, float(st.session_state.get(LENGTH_WIDGET_KEYS["duration"], 0.001024)))
    sample_len = max(1, int(round(float(st.session_state.get(LENGTH_WIDGET_KEYS["sample_len"], 1024)))))

    if source == "duration":
        st.session_state[LENGTH_WIDGET_KEYS["sample_len"]] = max(1, int(round(sample_rate * duration)))
        st.session_state["length_sync_focus"] = "duration"
    elif source == "sample_len":
        st.session_state[LENGTH_WIDGET_KEYS["duration"]] = sample_len / sample_rate
        st.session_state["length_sync_focus"] = "sample_len"
    elif source == "sample_rate":
        frequency_scale = _frequency_scale_from_session()
        sample_rate = max(
            1.0,
            float(st.session_state.get(LENGTH_WIDGET_KEYS["sample_rate_display"], 1_000_000 / frequency_scale)) * frequency_scale,
        )
        st.session_state[LENGTH_WIDGET_KEYS["sample_rate_hz"]] = int(round(sample_rate))
        focus = st.session_state.get("length_sync_focus", "duration")
        if focus == "sample_len":
            st.session_state[LENGTH_WIDGET_KEYS["duration"]] = sample_len / sample_rate
        else:
            st.session_state[LENGTH_WIDGET_KEYS["sample_len"]] = max(1, int(round(sample_rate * duration)))
            st.session_state["length_sync_focus"] = "duration"


def sync_sample_rate_display_to_unit() -> None:
    sample_rate_hz = max(1.0, float(st.session_state.get(LENGTH_WIDGET_KEYS["sample_rate_hz"], 1_000_000)))
    frequency_scale = _frequency_scale_from_session()
    st.session_state[LENGTH_WIDGET_KEYS["sample_rate_display"]] = sample_rate_hz / frequency_scale


def build_demo_preview_config(base_config: AppConfig) -> AppConfig:
    demo_config = deepcopy(base_config)
    demo_config.generators = ["Tone"]
    demo_config.global_params.update(
        {
            "sample_rate": 44_100,
            "center_frequency_hz": 66.6,
            "duration": 1.0,
            "sample_len": 44_100,
            "seed": 666,
        }
    )
    demo_config.dataset.total_samples = 1
    return demo_config


def ensure_lfm_chirp_override(config_obj: AppConfig) -> None:
    sample_rate = int(config_obj.global_params.get("sample_rate", 1_000_000))
    chirp = config_obj.generator_overrides.setdefault("LFM", {}).setdefault("chirp", {})
    chirp.setdefault("sweep_hz", max(1_000, sample_rate // 4))


seed_length_widget_state(config)


def generate_paired_dataset_set(selected_preset: str, base_output_dir: Path, test_mode: bool = False) -> tuple[dict[str, int | str | bool], dict[str, int | str | bool], Path]:
    companion_preset, group_slug = DATASET_PRESET_PAIRS[selected_preset]
    group_root = base_output_dir / group_slug
    primary_root = group_root / selected_preset.lower().replace(" ", "_")
    companion_root = group_root / companion_preset.lower().replace(" ", "_")

    primary_results = generate_dataset(build_config_for_dataset_preset(selected_preset, primary_root, test_mode))
    companion_results = generate_dataset(build_config_for_dataset_preset(companion_preset, companion_root, test_mode))
    return primary_results, companion_results, group_root


def generate_official_family_set(family_name: str, base_output_dir: Path, test_mode: bool = False) -> tuple[list[tuple[str, dict[str, int | str | bool]]], Path]:
    family_root = base_output_dir / OFFICIAL_DATASET_FAMILY_SLUGS[family_name]
    results: list[tuple[str, dict[str, int | str | bool]]] = []
    for preset_name in OFFICIAL_DATASET_FAMILIES[family_name]:
        preset_root = family_root / preset_name.lower().replace(" ", "_")
        results.append((preset_name, generate_dataset(build_config_for_dataset_preset(preset_name, preset_root, test_mode))))
    return results, family_root


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
        seed_length_widget_state(st.session_state.config, force=True)
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
    generation_scope = st.radio(
        "Generation scope",
        options=["Single split", "Paired official split", "Full official family"],
        horizontal=True,
    )
    st.caption(
        "Single split generates just the selected preset. Paired official split generates the selected split plus its companion. "
        "Full official family generates every official split in the chosen Sig53 or Wideband family. "
        "Note: Large datasets (>1 GB) cannot be downloaded via browser and must be accessed from the output directory."
    )
    current_dataset_preset = config.global_params.get("dataset_preset", "Custom")
    if current_dataset_preset not in DATASET_PRESETS:
        current_dataset_preset = "Custom"
    selected_dataset_preset = st.selectbox(
        "Dataset preset",
        options=list(DATASET_PRESETS.keys()),
        index=list(DATASET_PRESETS.keys()).index(current_dataset_preset),
        disabled=generation_scope == "Full official family",
    )
    if selected_dataset_preset != current_dataset_preset:
        apply_dataset_preset(selected_dataset_preset)
        current_dataset_preset = selected_dataset_preset
    if selected_dataset_preset != "Custom":
        preset = DATASET_PRESETS[selected_dataset_preset]
        st.caption(
            f"Applied official split preset: seed `{preset['seed']}`, "
            f"samples `{preset['total_samples']}`, split mode `{preset['split_mode']}`."
        )
        st.caption("Preset fields are locked; switch back to Custom to edit them manually.")
    selected_official_family = None
    family_presets: list[str] = []
    if generation_scope == "Full official family":
        family_default = "Sig53 official set"
        if selected_dataset_preset in OFFICIAL_DATASET_FAMILIES["Wideband Sig53 official set"]:
            family_default = "Wideband Sig53 official set"
        elif selected_dataset_preset in OFFICIAL_DATASET_FAMILIES["Sig53 official set"]:
            family_default = "Sig53 official set"
        selected_official_family = st.selectbox(
            "Official family",
            options=list(OFFICIAL_DATASET_FAMILIES.keys()),
            index=list(OFFICIAL_DATASET_FAMILIES.keys()).index(family_default),
        )
        family_presets = OFFICIAL_DATASET_FAMILIES[selected_official_family]
        family_preview_preset = family_presets[0]
        if config.global_params.get("dataset_preset") != family_preview_preset:
            apply_dataset_preset(family_preview_preset)
        ensure_lfm_chirp_override(config)
        if config.generators != list(TORCHSIG_CONCRETE_GENERATORS):
            config.generators = list(TORCHSIG_CONCRETE_GENERATORS)
        st.info(
            "This will generate every split in the family below. Each split gets its own directory with a single HDF5 file "
            "(HDF5 is hierarchical internally, not a directory tree). All splits will be written to the output directory on disk. "
            "Note: Full official families are too large for browser download and must be accessed from the filesystem."
        )
        st.caption(
            f"Full official family mode previews `{family_preview_preset}` in the UI, then overrides the dataset preset, output format, generator catalog, sample count, split mode, and seed fields below when you generate."
        )
        st.caption(
            "The preview split fields are locked to the first split in the chosen family so the UI shows a concrete starting point before generation."
        )
        with st.expander("Show generated splits", expanded=True):
            for preset_name in family_presets:
                preset = DATASET_PRESETS[preset_name]
                st.write(
                    f"- {preset_name}: {preset['total_samples']} samples, "
                    f"{preset['split_mode']}, seed {preset['seed']}"
                )
    preset_locked = selected_dataset_preset != "Custom" or generation_scope == "Full official family"

    output_format_label = st.radio(
        "Output format",
        options=["TorchSig-compatible HDF5", "NumPy split folders"],
        index=0 if config.dataset.output_format == "hdf5" else 1,
        horizontal=True,
        disabled=preset_locked,
    )
    if selected_dataset_preset == "Custom":
        config.dataset.output_format = "hdf5" if output_format_label == "TorchSig-compatible HDF5" else "numpy"
    generator_catalog_default = config.global_params.get(
        "generator_catalog",
        "TorchSig concrete labels"
        if any(is_torchsig_concrete_generator(name) for name in config.generators)
        else "Simplified families",
    )
    generator_catalog = st.radio(
        "Generator catalog",
        options=["Simplified families", "TorchSig concrete labels"],
        index=0 if generator_catalog_default == "Simplified families" else 1,
        horizontal=True,
        disabled=preset_locked,
    )
    if selected_dataset_preset == "Custom":
        config.global_params["generator_catalog"] = generator_catalog
    total_samples = st.number_input(
        "Total samples",
        min_value=1,
        value=config.dataset.total_samples,
        disabled=preset_locked,
    )
    split_mode = st.radio(
        "Split mode",
        options=["split", "train_only", "val_only"],
        index=["split", "train_only", "val_only"].index(config.dataset.split_mode)
        if config.dataset.split_mode in {"split", "train_only", "val_only"}
        else 0,
        horizontal=True,
        help="Split mode controls whether NumPy output is split into train/val folders or written as a single train-only / val-only partition.",
        disabled=preset_locked,
    )
    config.dataset.split_mode = split_mode
    if split_mode == "split":
        train_ratio = st.slider(
            "Train ratio",
            0.1,
            0.95,
            float(config.dataset.train_ratio),
            0.05,
            disabled=preset_locked,
        )
    else:
        train_ratio = float(config.dataset.train_ratio)
        st.caption("Train ratio is ignored outside split mode.")
    output_dir = st.text_input("Output directory", value=config.dataset.output_dir)

    st.caption("**Data creation performance settings**")
    create_batch_size = st.number_input(
        "Data creation batch size",
        min_value=1,
        max_value=1024,
        value=int(config.dataset.create_batch_size if hasattr(config.dataset, "create_batch_size") else 256),
        help="Batch size for data creation. Higher values use more memory but may be faster. Only applies to TorchSig generation path.",
    )
    create_num_workers = st.number_input(
        "Data creation workers",
        min_value=0,
        max_value=32,
        value=int(config.dataset.create_num_workers if hasattr(config.dataset, "create_num_workers") else 0),
        help="Number of parallel workers for data creation. RECOMMENDED: Keep at 0 for HDF5 (not thread-safe). Only applies to TorchSig generation path.",
    )
    max_memory_gb_val = None
    if hasattr(config.dataset, "max_memory_mb") and config.dataset.max_memory_mb:
        max_memory_gb_val = config.dataset.max_memory_mb / 1024
    max_memory_gb = st.number_input(
        "Maximum memory (GB, optional)",
        min_value=0,
        max_value=500,
        value=int(max_memory_gb_val) if max_memory_gb_val else 0,
        help="Cap memory usage to prevent server crashes. Set to 0 for no limit.",
    )

    with st.expander("Advanced HDF5 settings", expanded=False):
        compression_level = st.slider(
            "HDF5 compression level",
            min_value=0,
            max_value=9,
            value=int(getattr(config.dataset, "compression_level", 0)),
            help="gzip compression level for HDF5 output. 0=no compression (fastest, RECOMMENDED), 9=maximum compression (slowest). "
                 "Compression adds significant overhead. Enable only if disk space is critical. "
                 "Only applies to fallback h5py generation path.",
        )
        st.caption(
            "**Compression vs Speed:**\n"
            "- Level 0: **No compression (RECOMMENDED)** - fastest generation, larger files\n"
            "- Level 1-3: Fast compression (~1.5x slower, ~2-3x smaller)\n"
            "- Level 6: Balanced compression (~2-3x slower, ~3-4x smaller)\n"
            "- Level 9: Maximum compression (~3-4x slower, marginal improvement over 6)"
        )

    test_run_mode = st.checkbox(
        "Test run mode (10% of samples)",
        value=False,
        help="Generate only 10% of the total samples (minimum 100) to validate configuration before full run. Output directory will be suffixed with '_test'.",
    )
    if test_run_mode:
        test_sample_count = max(100, int(total_samples * 0.1))
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test Run Samples", f"{test_sample_count:,}", delta=f"-{int(total_samples) - test_sample_count:,}", delta_color="off")
        with col2:
            st.metric("Full Run Samples", f"{int(total_samples):,}")

        if split_mode == "split":
            test_train = int(test_sample_count * train_ratio)
            test_val = test_sample_count - test_train
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"**Test split:** {test_train:,} train + {test_val:,} val")
            with col2:
                st.caption(f"**Full split:** {train_count:,} train + {val_count:,} val")
        elif split_mode == "train_only":
            st.caption(f"**Test:** {test_sample_count:,} train samples | **Full:** {int(total_samples):,} train samples")
        elif split_mode == "val_only":
            st.caption(f"**Test:** {test_sample_count:,} val samples | **Full:** {int(total_samples):,} val samples")

    current_seed = config.global_params.get("seed")
    if selected_dataset_preset == "Custom":
        seed_preset = next(
            (label for label, preset_seed in SEED_PRESETS.items() if preset_seed == current_seed),
            "Custom",
        )
        selected_seed_preset = st.selectbox(
            "Seed preset",
            options=list(SEED_PRESETS.keys()),
            index=list(SEED_PRESETS.keys()).index(seed_preset),
        )
        preset_seed = SEED_PRESETS[selected_seed_preset]
        if preset_seed is None:
            st.caption("Use the global seed control to set a custom value.")
        else:
            config.global_params["seed"] = int(preset_seed)
            st.caption(f"Using preset seed `{preset_seed}` from the original TorchSig settings.")
    else:
        seed = int(config.global_params.get("seed", 1234567890))
        st.caption(f"Using dataset preset seed `{seed}` from the original TorchSig settings.")

    # Apply test run mode if enabled
    # Strip any existing _test suffix to prevent multiple appends
    base_output_dir = output_dir.rstrip("/").rstrip("\\")
    if base_output_dir.endswith("_test"):
        base_output_dir = base_output_dir[:-5]

    # Always save original values to config - test mode reduction happens at generation time only
    config.dataset = DatasetConfig(
        total_samples=int(total_samples),  # Always save original, not reduced
        train_ratio=float(train_ratio),
        output_dir=base_output_dir,  # Always save base dir without _test
        output_format="hdf5" if output_format_label == "TorchSig-compatible HDF5" else "numpy",
        split_mode=split_mode,
        create_batch_size=int(create_batch_size),
        create_num_workers=int(create_num_workers),
        max_memory_mb=int(max_memory_gb * 1024) if max_memory_gb > 0 else None,
        compression_level=int(compression_level),
    )
    if selected_dataset_preset != "Custom":
        config.dataset.output_format = DATASET_PRESETS[selected_dataset_preset]["output_format"]

with left:
    st.subheader("1) Generators")
    generator_catalog = config.global_params.get("generator_catalog", "Simplified families")
    resolved_generators = []
    unknown_generators = []
    full_family_mode = generation_scope == "Full official family"
    if full_family_mode:
        selected_generators = list(TORCHSIG_CONCRETE_GENERATORS)
        config.generators = selected_generators
        st.info(
            f"Full official family mode is using the full TorchSig concrete generator catalog ({len(config.generators)} generators)."
        )
        with st.expander("Selected generators", expanded=False):
            st.write(", ".join(config.generators))
        st.caption("The family preset controls generators in this mode.")
    elif generator_catalog == "TorchSig concrete labels":
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
    if st.session_state.get("demo_preview_mode", False) and selected_generators != ["Tone"]:
        st.session_state["demo_preview_mode"] = False
        st.rerun()

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

    # Initialize frequency_unit in session state if not present
    if "frequency_unit" not in st.session_state:
        st.session_state["frequency_unit"] = default_frequency_unit

    frequency_unit = st.selectbox(
        "Frequency unit",
        options=list(FREQUENCY_UNITS.keys()),
        key="frequency_unit",
        on_change=sync_sample_rate_display_to_unit,
        disabled=preset_locked,
    )
    frequency_scale = FREQUENCY_UNITS[frequency_unit]
    st.caption("Values are displayed in the selected unit but stored internally in Hz.")
    sample_rate_display = st.number_input(
        f"Sample rate ({frequency_unit})",
        min_value=1_000 / frequency_scale,
        step=max(0.001, 1_000.0 / frequency_scale),
        format="%.6f",
        key=LENGTH_WIDGET_KEYS["sample_rate_display"],
        on_change=sync_length_widget_state,
        args=("sample_rate",),
        disabled=preset_locked,
    )
    sample_rate = int(round(float(st.session_state[LENGTH_WIDGET_KEYS["sample_rate_display"]]) * frequency_scale))
    st.session_state[LENGTH_WIDGET_KEYS["sample_rate_hz"]] = sample_rate

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
        disabled=preset_locked,
    )
    center_frequency_hz = int(center_frequency_hz * frequency_scale)
    duration = st.number_input(
        "Duration (sec)",
        min_value=0.000001,
        format="%.6f",
        key=LENGTH_WIDGET_KEYS["duration"],
        on_change=sync_length_widget_state,
        args=("duration",),
        disabled=preset_locked,
    )
    snr_min, snr_max = st.slider(
        "SNR range (dB)",
        min_value=-20,
        max_value=60,
        value=tuple(config.global_params.get("snr_db", [0, 30])),
        disabled=preset_locked,
    )
    sample_len = st.number_input(
        "Sample length",
        min_value=128,
        step=1,
        key=LENGTH_WIDGET_KEYS["sample_len"],
        on_change=sync_length_widget_state,
        args=("sample_len",),
        disabled=preset_locked,
    )
    seed_value = config.global_params.get("seed")
    seed = st.number_input(
        "Seed",
        value=int(seed_value) if seed_value is not None else 123456789,
        step=1,
        help="Set this to make generation and previews repeatable.",
        key="global_seed",
        disabled=preset_locked,
    )

    center_freq_range_factor = st.number_input(
        "Center frequency range (fraction of Fs)",
        min_value=0.05,
        max_value=0.5,
        value=float(config.global_params.get("signal_center_freq_range_factor", 0.16)),
        step=0.01,
        help="Uniform range for signal center frequencies as fraction of sample rate. Original Sig53 used 0.16 (±1.6 MHz for 10 MHz sample rate).",
        disabled=preset_locked,
    )

    config.global_params.update(
        {
            "sample_rate": int(sample_rate),
            "center_frequency_hz": int(center_frequency_hz),
            "duration": float(st.session_state[LENGTH_WIDGET_KEYS["duration"]]),
            "snr_db": [int(snr_min), int(snr_max)],
            "sample_len": int(st.session_state[LENGTH_WIDGET_KEYS["sample_len"]]),
            "seed": int(seed),
            "signal_center_freq_range_factor": float(center_freq_range_factor),
        }
    )

    if "LFM" in selected_generator_families:
        st.info("LFM selected: chirp parameters are required.")
        # LFM parameters are locked in "Full official family" mode to preserve canonical datasets
        lfm_locked = generation_scope == "Full official family"
        sweep_hz = st.number_input(
            "LFM sweep bandwidth (Hz)",
            min_value=1_000,
            value=50_000,
            disabled=lfm_locked,
            help="Locked in 'Full official family' mode to preserve canonical datasets." if lfm_locked else None,
        )
        config.generator_overrides.setdefault("LFM", {})["chirp"] = {"sweep_hz": int(sweep_hz)}

    st.subheader("3) Transform pipeline")
    st.caption(
        "These transforms are post-generation augmentations, not TorchSig impairments. "
        "They are previewed for every backend, but only applied to NumPy output when enabled."
    )
    apply_post_transforms = st.checkbox(
        "Apply post-generation transforms to NumPy output",
        value=bool(config.global_params.get("apply_post_transforms", config.dataset.output_format == "numpy")),
        disabled=config.dataset.output_format != "numpy" or preset_locked and generation_scope == "Full official family",
        help="When enabled, the supported post-generation transforms are applied to the NumPy fallback output only.",
    )
    config.global_params["apply_post_transforms"] = bool(apply_post_transforms)
    if config.dataset.output_format != "numpy":
        st.caption("TorchSig-compatible HDF5 generation does not apply the post-generation transform chain.")
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
    demo_preview_mode = st.checkbox(
        "Use demo preview tone",
        value=bool(st.session_state.get("demo_preview_mode", True)),
        help="When enabled, the preview is forced to a 66.6 Hz tone demo with a 44.1 kHz sample rate, 44,100 samples, 1.0 second duration, band center at 66.6 Hz, and seed 666.",
        key="demo_preview_mode",
    )
    preview_config = config
    if demo_preview_mode:
        preview_config = build_demo_preview_config(config)
        st.caption(
            "Demo preview mode is forcing Tone at 44.1 kHz, 66.6 Hz center frequency, 1.0 s duration, 44,100 samples, and seed 666."
        )
    preview_transforms = [step.name for step in config.transforms if step.enabled and step.name in TRANSFORM_REGISTRY]
    if demo_preview_mode:
        if "AWGN" not in preview_transforms:
            preview_transforms = ["AWGN", *preview_transforms]
        else:
            preview_transforms = ["AWGN", *[name for name in preview_transforms if name != "AWGN"]]
    if preview_transforms:
        preview_stages = build_transform_preview(preview_config, preview_transforms)
        st.caption(
            "Approximate preview built from a clean tone, then the selected preview transforms, then a single spectrogram view that is zoomed to the active band. "
            "Demo mode injects AWGN once so the tone remains visible in the preview."
        )
        with st.expander("Show preview", expanded=True):
            st.pyplot(render_preview_figure(preview_stages, preview_config), clear_figure=True)
    else:
        st.info("Add at least one recognized transform to see a live preview.")

output_dir_error = None
safe_output_dir = None
try:
    safe_output_dir = sanitize_output_dir(config.dataset.output_dir)
except ValueError as exc:
    output_dir_error = str(exc)

errors, warnings = validate_config(config)
if config.dataset.split_mode == "train_only":
    train_count, val_count = config.dataset.total_samples, 0
elif config.dataset.split_mode == "val_only":
    train_count, val_count = 0, config.dataset.total_samples
else:
    train_count = int(config.dataset.total_samples * config.dataset.train_ratio)
    val_count = config.dataset.total_samples - train_count
split_has_zero_partition = config.dataset.split_mode == "split" and (
    train_count == 0 or val_count == 0
)
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
paired_generate_disabled = (
    selected_dataset_preset == "Custom"
    or selected_dataset_preset not in DATASET_PRESET_PAIRS
    or bool(errors)
    or bool(output_dir_error)
    or not confirm_non_empty_output
)
if generation_scope == "Full official family":
    family_generate_disabled = bool(errors) or bool(output_dir_error) or not confirm_non_empty_output
    family_button_label = "Generate full official family set (test 10%)" if test_run_mode else "Generate full official family set"
    if st.button(family_button_label, type="primary", disabled=family_generate_disabled):
        # For test mode, update the base_root to add _test suffix
        # Strip any existing _test suffix to prevent multiple appends
        clean_output_dir = config.dataset.output_dir.rstrip("/").rstrip("\\")
        if clean_output_dir.endswith("_test"):
            clean_output_dir = clean_output_dir[:-5]
        base_output_dir = f"{clean_output_dir}_test" if test_run_mode else clean_output_dir
        base_root = sanitize_output_dir(base_output_dir)
        family_results, family_root = generate_official_family_set(
            selected_official_family or "Sig53 official set",
            base_root,
            test_mode=test_run_mode,
        )
        st.session_state.download_zip = build_dataset_zip_bytes(family_root)
        st.session_state.download_name = f"{family_root.name or 'official_family'}.zip"
        success_prefix = "Test run: Generated" if test_run_mode else "Generated"
        st.success(
            f"{success_prefix} the full official family at "
            f"{family_root} ({selected_official_family})."
        )
        if test_run_mode:
            total_full_samples = sum(DATASET_PRESETS[name]["total_samples"] for name in family_presets)
            st.info(f"This was a test run with 10% samples per split. Full family would have {total_full_samples:,} total samples.")
        hdf5_count = sum(1 for _, result in family_results if result["output_format"] == "hdf5")
        if hdf5_count:
            torchsig_count = sum(1 for _, result in family_results if result["torchsig_generated"])
            if torchsig_count:
                st.success(f"TorchSig generation backend was used for {torchsig_count} split(s).")
            else:
                st.info("TorchSig was unavailable, so the family used the local h5py fallback for HDF5 splits.")
elif selected_dataset_preset != "Custom":
    paired_button_label = "Generate paired official train/val set (test 10%)" if test_run_mode else "Generate paired official train/val set"
    if st.button(paired_button_label, disabled=paired_generate_disabled, use_container_width=True):
        # For test mode, update the base_root to add _test suffix
        # Strip any existing _test suffix to prevent multiple appends
        clean_output_dir = config.dataset.output_dir.rstrip("/").rstrip("\\")
        if clean_output_dir.endswith("_test"):
            clean_output_dir = clean_output_dir[:-5]
        base_output_dir = f"{clean_output_dir}_test" if test_run_mode else clean_output_dir
        base_root = sanitize_output_dir(base_output_dir)
        primary_results, companion_results, group_root = generate_paired_dataset_set(
            selected_dataset_preset,
            base_root,
            test_mode=test_run_mode,
        )
        st.session_state.download_zip = build_dataset_zip_bytes(group_root)
        st.session_state.download_name = f"{group_root.name or 'dataset_pair'}.zip"
        success_prefix = "Test run: Generated" if test_run_mode else "Generated"
        st.success(
            f"{success_prefix} paired official datasets at "
            f"{group_root} ({selected_dataset_preset} + {DATASET_PRESET_PAIRS[selected_dataset_preset][0]})."
        )
        if test_run_mode:
            preset_config = DATASET_PRESETS[selected_dataset_preset]
            companion_preset = DATASET_PRESET_PAIRS[selected_dataset_preset][0]
            companion_config = DATASET_PRESETS[companion_preset]
            st.info(
                f"This was a test run with 10% samples. Full would be: "
                f"{preset_config['total_samples']:,} ({selected_dataset_preset}) + "
                f"{companion_config['total_samples']:,} ({companion_preset})"
            )
        if primary_results["output_format"] == "hdf5" and companion_results["output_format"] == "hdf5":
            if primary_results["torchsig_generated"] or companion_results["torchsig_generated"]:
                st.success("TorchSig generation backend was used for at least one split.")
            else:
                st.info("TorchSig was unavailable for both splits, so h5py fallback files were written.")
else:
    button_label = "Generate test dataset (10%)" if test_run_mode else "Generate dataset"
    button_type = "secondary" if test_run_mode else "primary"
    if st.button(button_label, type=button_type, disabled=generate_disabled):
        # Apply test mode modifications right before generation
        if test_run_mode:
            config.dataset.output_dir = f"{config.dataset.output_dir}_test"
            config.dataset.total_samples = max(100, int(config.dataset.total_samples * 0.1))
        results = generate_dataset(config)
        st.session_state.download_zip = build_dataset_zip_bytes(results["output_dir"])
        st.session_state.download_name = f"{Path(results['output_dir']).name or 'dataset'}.zip"

        success_prefix = "Test run: Generated" if test_run_mode else "Generated"
        if results["output_format"] == "hdf5":
            st.success(
                f"{success_prefix} TorchSig-compatible HDF5 dataset at "
                f"{results['output_dir']} with {config.dataset.total_samples} samples "
                f"({config.dataset.split_mode})."
            )
            if test_run_mode:
                st.info(f"This was a test run. Full dataset would have {int(total_samples):,} samples.")
            if results["torchsig_generated"]:
                st.success("TorchSig generation backend was used.")
            else:
                st.info(
                    "TorchSig was unavailable, so the app wrote a compatible data.h5 file with h5py instead."
                    f" Reason: {results['torchsig_error'] or 'unknown'}"
                )
        else:
            st.success(
                f"{success_prefix} NumPy split-folder dataset at "
                f"{results['output_dir']} (train={results['train_samples']}, val={results['val_samples']})."
            )
            if test_run_mode:
                full_train = int(int(total_samples) * train_ratio)
                full_val = int(total_samples) - full_train
                st.info(f"This was a test run. Full dataset would have {full_train:,} train and {full_val:,} val samples.")

if st.session_state.download_zip:
    # Only allow download for datasets < 1GB to prevent memory issues
    dataset_size_gb = len(st.session_state.download_zip) / (1024**3)

    if dataset_size_gb < 1.0:
        st.download_button(
            "Download generated dataset (.zip)",
            data=st.session_state.download_zip,
            file_name=st.session_state.download_name,
            mime="application/zip",
            use_container_width=True,
        )
    else:
        st.warning(
            f"Dataset is too large to download via browser ({dataset_size_gb:.1f} GB). "
            f"Files are available on disk - check the output directory path above."
        )
        # Free memory immediately
        st.session_state.download_zip = None

st.markdown("---")
st.subheader("Current Config Snapshot")
st.json(config.to_dict())

if Path("README.md").exists():
    with st.expander("Project spec excerpt"):
        st.write("See README.md for full functional specification and phased roadmap.")
