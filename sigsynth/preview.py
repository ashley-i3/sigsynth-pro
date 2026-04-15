from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_MPL_CACHE_DIR = Path(os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib"))
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

import matplotlib.pyplot as plt
from scipy.signal import stft

from sigsynth.models import AppConfig
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
    rng = _sample_rng(config)
    sample_len = int(config.global_params.get("sample_len", 1024))
    sample_rate = int(config.global_params.get("sample_rate", 1_000_000))
    center_frequency_hz = float(config.global_params.get("center_frequency_hz", 0.0))
    t = np.arange(sample_len, dtype=float) / sample_rate
    resolved_generators = {resolve_generator_name(name) or name for name in config.generators}

    def _fm_demo_tone() -> np.ndarray:
        carrier_hz = 66.6
        modulation = 8.0 * np.sin(2.0 * np.pi * 0.75 * t) + 2.5 * np.sin(2.0 * np.pi * 3.25 * t)
        instantaneous_freq = carrier_hz + modulation
        phase = 2.0 * np.pi * np.cumsum(instantaneous_freq) / sample_rate
        return np.exp(1j * phase)

    if "LFM" in resolved_generators or "ChirpSS" in resolved_generators:
        sweep_hz = float(config.generator_overrides.get("LFM", {}).get("chirp", {}).get("sweep_hz", sample_rate / 4))
        chirp_rate = sweep_hz / max(t[-1], 1e-9)
        phase = 2 * np.pi * (0.5 * chirp_rate * t**2)
        base = np.exp(1j * phase)
    else:
        base = _fm_demo_tone()

    base *= np.exp(1j * 2 * np.pi * center_frequency_hz * t)
    base += 0.03 * (rng.standard_normal(sample_len) + 1j * rng.standard_normal(sample_len))
    return base.astype(np.complex64)


def _apply_awgn(signal: np.ndarray, config: AppConfig) -> np.ndarray:
    snr_db = config.global_params.get("snr_db", [0, 30])
    if isinstance(snr_db, (list, tuple)) and len(snr_db) >= 2:
        snr_mid = float(snr_db[0] + snr_db[1]) / 2.0
    else:
        snr_mid = 15.0
    rng = _sample_rng(config)
    power = np.mean(np.abs(signal) ** 2)
    noise_power = power / (10 ** (snr_mid / 10))
    noise = np.sqrt(noise_power / 2.0) * (rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal)))
    return (signal + noise).astype(np.complex64)


def _apply_freq_offset(signal: np.ndarray, config: AppConfig) -> np.ndarray:
    sample_rate = int(config.global_params.get("sample_rate", 1_000_000))
    sample_len = len(signal)
    offset_hz = min(max(sample_rate * 0.02, 1_000.0), sample_rate / 8.0)
    t = np.arange(sample_len, dtype=float) / sample_rate
    return (signal * np.exp(1j * 2 * np.pi * offset_hz * t)).astype(np.complex64)


def _apply_iq_imbalance(signal: np.ndarray, config: AppConfig) -> np.ndarray:
    rng = _sample_rng(config)
    amplitude = 1.0 + rng.uniform(-0.15, 0.15)
    phase = rng.uniform(-0.08, 0.08)
    i = signal.real * amplitude
    q = signal.imag * (2.0 - amplitude)
    rotated_i = i * np.cos(phase) - q * np.sin(phase)
    rotated_q = i * np.sin(phase) + q * np.cos(phase)
    dc = 0.03 * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))
    return (rotated_i + 1j * rotated_q + dc).astype(np.complex64)


def _apply_chirp_flatten(signal: np.ndarray, config: AppConfig) -> np.ndarray:
    sample_rate = int(config.global_params.get("sample_rate", 1_000_000))
    t = np.arange(len(signal), dtype=float) / sample_rate
    flatten_rate = sample_rate * 0.02
    return (signal * np.exp(-1j * 2 * np.pi * (0.5 * flatten_rate * t**2))).astype(np.complex64)


def _apply_complex_to_real_magnitude(signal: np.ndarray) -> np.ndarray:
    return np.abs(signal).astype(np.float32)


def _apply_spectrogram(signal: np.ndarray, config: AppConfig) -> np.ndarray:
    fft_size = int(config.global_params.get("sample_len", 1024))
    fft_size = max(32, min(256, fft_size))
    _, _, spec = stft(
        signal,
        nperseg=fft_size,
        noverlap=fft_size // 2,
        boundary=None,
        return_onesided=False,
    )
    return np.abs(spec).astype(np.float32)


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
    stages = [PreviewStage(name="Input", data=_make_base_signal(config), kind="complex")]
    current = stages[0].data

    for name in transform_names[: max_stages - 1]:
        current = apply_preview_transform(name, current, config)
        kind = "spectrogram" if name == "Spectrogram" else ("real" if np.isrealobj(current) else "complex")
        stages.append(PreviewStage(name=name, data=current, kind=kind))
        if kind == "spectrogram":
            break

    if stages[-1].name != "Final":
        stages.append(PreviewStage(name="Final", data=current, kind="spectrogram" if current.ndim == 2 else ("real" if np.isrealobj(current) else "complex")))
    return stages


def render_preview_figure(stages: list[PreviewStage]):
    rows = len(stages)
    fig, axes = plt.subplots(rows, 1, figsize=(10, max(2.5, 2.2 * rows)), constrained_layout=True)
    if rows == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages):
        data = stage.data
        if data.ndim == 2:
            ax.imshow(data, aspect="auto", origin="lower", cmap="magma")
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
