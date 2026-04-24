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
from sigsynth.numpy_synth import synthesize_sample
from sigsynth.post_transforms import (
    apply_awgn,
    apply_chirp_flatten,
    apply_complex_to_real_magnitude,
    apply_freq_offset,
    apply_iq_imbalance,
    apply_random_phase_shift,
    apply_random_resample,
    apply_random_time_shift,
    apply_rayleigh_fading,
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


def _make_base_signal(config: AppConfig, use_demo: bool = False) -> np.ndarray:
    """Generate base signal - either a real dataset sample or a demo tone.

    Args:
        config: Application configuration
        use_demo: If True, generate simple demo tone. If False, generate real dataset sample.
    """
    sample_len = int(config.global_params.get("sample_len", 1024))
    sample_rate = int(config.global_params.get("sample_rate", 1_000_000))

    if not use_demo and config.generators:
        # Generate a real dataset sample (sample_index=0 for repeatability)
        try:
            sample = synthesize_sample(config, sample_index=0)
            return sample.clean.astype(np.complex64)
        except Exception:
            # Fall back to demo tone if generation fails
            pass

    # Demo mode: simple tone or chirp
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
    if name == "RandomPhaseShift":
        return apply_random_phase_shift(signal, config)
    if name == "RandomTimeShift":
        return apply_random_time_shift(signal, config)
    if name == "RayleighFadingChannel":
        return apply_rayleigh_fading(signal, config)
    if name == "RandomResample":
        return apply_random_resample(signal, config)
    return signal


def build_transform_preview(config: AppConfig, transform_names: list[str], max_stages: int = 6, use_demo: bool = False) -> list[PreviewStage]:
    """Build preview stages showing signal transformations.

    Args:
        config: Application configuration
        transform_names: List of transform names to apply
        max_stages: Maximum number of stages to show
        use_demo: If True, use demo tone. If False, use real dataset sample.
    """
    base = _make_base_signal(config, use_demo=use_demo)
    label = "Demo Tone" if use_demo else "Clean Signal"
    stages = [PreviewStage(name=label, data=base, kind="complex")]
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
    """
    Zoom spectrogram to show signal with context.

    For narrowband signals, shows more surrounding empty frequency space
    to illustrate the true bandwidth characteristics.
    """
    if data.ndim != 2 or data.shape[0] < 8:
        return data

    energy = np.mean(np.abs(data), axis=1)
    center = int(np.argmax(energy))

    # Detect if signal is narrowband by measuring energy concentration
    # Calculate what fraction of total energy is in the top 20% of frequency bins
    sorted_energy = np.sort(energy)[::-1]
    top_20_percent_bins = max(1, int(len(energy) * 0.2))
    energy_concentration = np.sum(sorted_energy[:top_20_percent_bins]) / max(np.sum(energy), 1e-10)

    # If > 80% of energy is in 20% of bins, it's narrowband - show more context
    if energy_concentration > 0.8:
        # Narrowband: show 70% of spectrum to see surrounding empty space
        target_fraction = 0.7
    else:
        # Wideband: use tighter zoom to focus on signal detail
        target_fraction = 0.35

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


def _format_frequency_axis(freq_hz: float) -> tuple[float, str]:
    """Convert frequency to appropriate unit and return (value, unit)."""
    if abs(freq_hz) >= 1e9:
        return freq_hz / 1e9, "GHz"
    elif abs(freq_hz) >= 1e6:
        return freq_hz / 1e6, "MHz"
    elif abs(freq_hz) >= 1e3:
        return freq_hz / 1e3, "kHz"
    else:
        return freq_hz, "Hz"


def _format_time_axis(time_s: float) -> tuple[float, str]:
    """Convert time to appropriate unit and return (value, unit)."""
    if time_s >= 1.0:
        return time_s, "s"
    elif time_s >= 1e-3:
        return time_s * 1e3, "ms"
    elif time_s >= 1e-6:
        return time_s * 1e6, "μs"
    else:
        return time_s * 1e9, "ns"


def render_preview_figure(stages: list[PreviewStage], config: AppConfig):
    rows = len(stages)
    fig, axes = plt.subplots(rows, 1, figsize=(10, max(2.5, 2.2 * rows)), constrained_layout=True)
    if rows == 1:
        axes = [axes]

    sample_rate = int(config.global_params.get("sample_rate", 1_000_000))
    sample_len = int(config.global_params.get("sample_len", 1024))

    for ax, stage in zip(axes, stages):
        data = stage.data
        if data.ndim == 2:
            # Spectrogram: frequency on y-axis, time on x-axis
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

            # Convert y-axis (frequency bins) to actual frequencies
            freq_bins = data.shape[0]
            nyquist = sample_rate / 2.0
            freq_range = np.linspace(-nyquist, nyquist, freq_bins)
            _, freq_unit = _format_frequency_axis(nyquist)
            freq_scale = 1e9 if freq_unit == "GHz" else 1e6 if freq_unit == "MHz" else 1e3 if freq_unit == "kHz" else 1.0

            # Set y-axis ticks and labels (both at once to avoid warning)
            y_tick_positions = ax.get_yticks()
            y_labels = []
            valid_ticks = []
            for tick in y_tick_positions:
                if 0 <= tick < freq_bins:
                    valid_ticks.append(tick)
                    freq_hz = freq_range[int(tick)]
                    y_labels.append(f"{freq_hz / freq_scale:.1f}")
            if valid_ticks:
                ax.set_yticks(valid_ticks, y_labels)
            ax.set_ylabel(f"Frequency ({freq_unit})")

            # Convert x-axis (time bins) to actual time
            time_bins = data.shape[1]
            duration_s = sample_len / sample_rate
            _, time_unit = _format_time_axis(duration_s)
            time_scale = 1.0 if time_unit == "s" else 1e-3 if time_unit == "ms" else 1e-6 if time_unit == "μs" else 1e-9
            ax.set_xlabel(f"Time ({time_unit})")

            # Set x-axis ticks and labels (both at once to avoid warning)
            x_tick_positions = ax.get_xticks()
            x_labels = []
            valid_ticks = []
            for tick in x_tick_positions:
                if 0 <= tick < time_bins:
                    valid_ticks.append(tick)
                    time_s = (tick / time_bins) * duration_s
                    x_labels.append(f"{time_s / time_scale:.1f}")
            if valid_ticks:
                ax.set_xticks(valid_ticks, x_labels)

        elif np.iscomplexobj(data):
            # Time-domain IQ plot
            limit = min(len(data), 256)
            time_s = np.arange(limit) / sample_rate
            time_val, time_unit = _format_time_axis(time_s[-1] if len(time_s) > 0 else 1e-6)
            time_scale = 1.0 if time_unit == "s" else 1e-3 if time_unit == "ms" else 1e-6 if time_unit == "μs" else 1e-9
            time_axis = time_s / time_scale

            ax.plot(time_axis, data.real[:limit], label="real", linewidth=1.0)
            ax.plot(time_axis, data.imag[:limit], label="imag", linewidth=1.0, alpha=0.8)

            # Set y-axis limits based on actual data range for better visibility
            real_range = np.ptp(data.real[:limit])
            imag_range = np.ptp(data.imag[:limit])
            if real_range > 0 or imag_range > 0:
                data_min = min(np.min(data.real[:limit]), np.min(data.imag[:limit]))
                data_max = max(np.max(data.real[:limit]), np.max(data.imag[:limit]))
                margin = (data_max - data_min) * 0.1
                ax.set_ylim(data_min - margin, data_max + margin)

            ax.set_ylabel("Amplitude")
            ax.set_xlabel(f"Time ({time_unit})")
            ax.legend(loc="upper right", fontsize="small")
        else:
            # Real time-domain plot
            limit = min(len(data), 256)
            time_s = np.arange(limit) / sample_rate
            time_val, time_unit = _format_time_axis(time_s[-1] if len(time_s) > 0 else 1e-6)
            time_scale = 1.0 if time_unit == "s" else 1e-3 if time_unit == "ms" else 1e-6 if time_unit == "μs" else 1e-9
            time_axis = time_s / time_scale

            ax.plot(time_axis, data[:limit], color="#2b6cb0", linewidth=1.0)

            # Set y-axis limits based on actual data range for better visibility
            data_range = np.ptp(data[:limit])
            if data_range > 0:
                data_min = np.min(data[:limit])
                data_max = np.max(data[:limit])
                margin = (data_max - data_min) * 0.1
                ax.set_ylim(data_min - margin, data_max + margin)

            ax.set_ylabel("Amplitude")
            ax.set_xlabel(f"Time ({time_unit})")
        ax.set_title(stage.name)

    return fig
