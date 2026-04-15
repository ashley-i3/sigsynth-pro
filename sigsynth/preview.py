from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_MPL_CACHE_DIR = Path(os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib"))
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

import matplotlib.pyplot as plt

from sigsynth.models import AppConfig
from sigsynth.post_transforms import (
    apply_awgn,
    apply_chirp_flatten,
    apply_complex_to_real_magnitude,
    apply_freq_offset,
    apply_iq_imbalance,
    apply_spectrogram,
)
from sigsynth.registry import resolve_generator_name


@dataclass
class PreviewStage:
    name: str
    data: np.ndarray
    kind: str


def _sample_rng(config: AppConfig) -> np.random.Generator:
    base_seed = config.global_params.get("seed")
    try:
        seed_offset = 0 if base_seed is None else int(base_seed) * 2654435761
    except (TypeError, ValueError):
        seed_offset = 0
    seed = seed_offset + int(config.dataset.total_samples) * 17 + int(config.global_params.get("sample_len", 1024))
    return np.random.default_rng(seed=seed)


def _make_base_signal(config: AppConfig) -> np.ndarray:
    sample_len = int(config.global_params.get("sample_len", 1024))
    sample_rate = int(config.global_params.get("sample_rate", 1_000_000))
    center_frequency_hz = float(config.global_params.get("center_frequency_hz", 0.0))
    t = np.arange(sample_len, dtype=float) / sample_rate
    resolved_generators = {resolve_generator_name(name) or name for name in config.generators}

    def _demo_tone() -> np.ndarray:
        carrier_hz = 66.6
        phase = 2.0 * np.pi * carrier_hz * t
        return np.exp(1j * phase)

    if "LFM" in resolved_generators or "ChirpSS" in resolved_generators:
        sweep_hz = float(config.generator_overrides.get("LFM", {}).get("chirp", {}).get("sweep_hz", sample_rate / 4))
        chirp_rate = sweep_hz / max(t[-1], 1e-9)
        phase = 2 * np.pi * (0.5 * chirp_rate * t**2)
        base = np.exp(1j * phase)
    else:
        base = _demo_tone()

    base *= np.exp(1j * 2 * np.pi * center_frequency_hz * t)
    return base.astype(np.complex64)


def _apply_awgn(signal: np.ndarray, config: AppConfig) -> np.ndarray:
    return apply_awgn(signal, config)


def _apply_freq_offset(signal: np.ndarray, config: AppConfig) -> np.ndarray:
    return apply_freq_offset(signal, config)


def _apply_iq_imbalance(signal: np.ndarray, config: AppConfig) -> np.ndarray:
    return apply_iq_imbalance(signal, config)


def _apply_chirp_flatten(signal: np.ndarray, config: AppConfig) -> np.ndarray:
    return apply_chirp_flatten(signal, config)


def _apply_complex_to_real_magnitude(signal: np.ndarray) -> np.ndarray:
    return apply_complex_to_real_magnitude(signal)


def _apply_spectrogram(signal: np.ndarray, config: AppConfig) -> np.ndarray:
    return apply_spectrogram(_upsample_for_spectrogram(signal), config)


def apply_preview_transform(name: str, signal: np.ndarray, config: AppConfig) -> np.ndarray:
    if name == "AWGN":
        return _apply_awgn(signal, config)
    if name == "FreqOffset":
        return _apply_freq_offset(signal, config)
    if name == "IQImbalance":
        return _apply_iq_imbalance(signal, config)
    if name == "ChirpFlatten":
        return _apply_chirp_flatten(signal, config)
    if name == "ComplexToRealMagnitude":
        return _apply_complex_to_real_magnitude(signal)
    if name == "Spectrogram":
        return _apply_spectrogram(signal, config)
    return signal


def build_transform_preview(config: AppConfig, transform_names: list[str], max_stages: int = 6) -> list[PreviewStage]:
    base = _make_base_signal(config)
    stages = [PreviewStage(name="Clean Tone", data=base, kind="complex")]
    current = base

    for name in transform_names[: max_stages - 2]:
        current = apply_preview_transform(name, current, config)
        kind = "spectrogram" if name == "Spectrogram" else ("real" if np.isrealobj(current) else "complex")
        label = f"After {name}" if name != "Spectrogram" else "Spectrogram"
        stages.append(PreviewStage(name=label, data=current, kind=kind))
        if kind == "spectrogram":
            break

    if not any(name == "Spectrogram" for name in transform_names):
        current = apply_spectrogram(_upsample_for_spectrogram(current), config)
        stages.append(PreviewStage(name="Preview Spectrogram", data=current, kind="spectrogram"))
    return stages


def _zoom_spectrogram(data: np.ndarray, target_fraction: float = 0.35) -> np.ndarray:
    if data.ndim != 2 or data.shape[0] < 8:
        return data

    energy = np.mean(np.abs(data), axis=1)
    center = int(np.argmax(energy))
    half_span = max(16, int(data.shape[0] * target_fraction / 2.0))
    start = max(0, center - half_span)
    stop = min(data.shape[0], center + half_span)
    if stop - start < 8:
        start = max(0, center - 4)
        stop = min(data.shape[0], center + 4)
    return data[start:stop, :]


def _upsample_for_spectrogram(signal: np.ndarray, target_len: int = 16384) -> np.ndarray:
    if len(signal) >= target_len:
        return signal
    repeats = int(np.ceil(target_len / max(1, len(signal))))
    return np.tile(signal, repeats)[:target_len]


def render_preview_figure(stages: list[PreviewStage]):
    rows = len(stages)
    fig, axes = plt.subplots(rows, 1, figsize=(10, max(2.5, 2.2 * rows)), constrained_layout=True)
    if rows == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages):
        data = stage.data
        if data.ndim == 2:
            data = _zoom_spectrogram(data)
            finite = np.isfinite(data)
            if np.any(finite):
                vmin = float(np.nanpercentile(data[finite], 5))
                vmax = float(np.nanpercentile(data[finite], 95))
            else:
                vmin, vmax = -80.0, 0.0
            if np.isclose(vmin, vmax):
                vmin, vmax = vmin - 1.0, vmax + 1.0
            ax.imshow(data, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_ylabel("Bins")
        elif np.iscomplexobj(data):
            limit = min(len(data), 256)
            x = np.arange(limit)
            ax.plot(x, data.real[:limit], label="real", linewidth=1.0)
            ax.plot(x, data.imag[:limit], label="imag", linewidth=1.0, alpha=0.8)
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right", fontsize="small")
        else:
            limit = min(len(data), 256)
            x = np.arange(limit)
            ax.plot(x, data[:limit], color="#2b6cb0", linewidth=1.0)
            ax.set_ylabel("Amplitude")
        ax.set_title(stage.name)
        ax.set_xlabel("Sample" if data.ndim == 1 else "Time bin")

    return fig
