from __future__ import annotations

from pathlib import Path

import streamlit as st

from sigsynth.generator import generate_dataset
from sigsynth.macro_manager import MacroManager
from sigsynth.models import AppConfig, DatasetConfig, TransformStep
from sigsynth.registry import GENERATOR_REGISTRY, TRANSFORM_REGISTRY
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

config: AppConfig = st.session_state.config

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
        if not save_name.endswith(".yaml"):
            save_name += ".yaml"
        path = macro_manager.save(save_name, config)
        st.success(f"Saved {path}")

    st.markdown("---")
    st.subheader("Dataset")
    total_samples = st.number_input("Total samples", min_value=2, value=config.dataset.total_samples)
    train_ratio = st.slider("Train ratio", 0.1, 0.95, float(config.dataset.train_ratio), 0.05)
    output_dir = st.text_input("Output directory", value=config.dataset.output_dir)
    config.dataset = DatasetConfig(total_samples=int(total_samples), train_ratio=float(train_ratio), output_dir=output_dir)

with left:
    st.subheader("1) Generators")
    selected_generators = st.multiselect(
        "Choose signal generators",
        options=sorted(GENERATOR_REGISTRY.keys()),
        default=config.generators,
    )
    config.generators = selected_generators

    st.subheader("2) Global parameters")
    sample_rate = st.number_input(
        "Sample rate (Hz)", min_value=1_000, value=int(config.global_params.get("sample_rate", 1_000_000))
    )
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
            "duration": float(duration),
            "snr_db": [int(snr_min), int(snr_max)],
            "sample_len": int(sample_len),
        }
    )

    if "LFM" in config.generators:
        st.info("LFM selected: chirp parameters are required.")
        sweep_hz = st.number_input("LFM sweep bandwidth (Hz)", min_value=1_000, value=50_000)
        config.generator_overrides.setdefault("LFM", {})["chirp"] = {"sweep_hz": int(sweep_hz)}

    st.subheader("3) Transform pipeline")
    enabled_transforms = st.multiselect(
        "Enable transforms in order",
        options=sorted(TRANSFORM_REGISTRY.keys()),
        default=[step.name for step in config.transforms if step.enabled],
    )
    config.transforms = [TransformStep(name=name, enabled=True) for name in enabled_transforms]

errors, warnings = validate_config(config)

st.subheader("Validation")
if warnings:
    for item in warnings:
        st.warning(item)
if errors:
    for item in errors:
        st.error(item)
else:
    st.success("Configuration is valid.")

if st.button("Generate dataset", type="primary", disabled=bool(errors)):
    results = generate_dataset(config)
    st.success(
        "Generated dataset at "
        f"{results['output_dir']} (train={results['train_samples']}, val={results['val_samples']})."
    )
    if not results["torchsig_available"]:
        st.info("TorchSig import not available in this environment; synthetic NumPy placeholder generation was used.")

st.markdown("---")
st.subheader("Current Config Snapshot")
st.json(config.to_dict())

if Path("README.md").exists():
    with st.expander("Project spec excerpt"):
        st.write("See README.md for full functional specification and phased roadmap.")
