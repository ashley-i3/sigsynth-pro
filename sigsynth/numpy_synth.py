from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from sigsynth.models import AppConfig
from sigsynth.registry import resolve_generator_name


@dataclass(frozen=True)
class SynthSample:
    generator: str
    clean: np.ndarray
    impaired: np.ndarray
    metadata: dict[str, object]


def _sample_rng(config: AppConfig, sample_index: int, salt: int = 0) -> np.random.Generator:
    seed = (
        int(config.dataset.total_samples) * 1009
        + int(config.global_params.get("sample_len", 1024)) * 917
        + sample_index * 65537
        + salt * 131071
    )
    return np.random.default_rng(seed=seed)


def _choose_generator(config: AppConfig, rng: np.random.Generator) -> str:
    generators = [resolve_generator_name(name) or name for name in config.generators]
    generators = [name for name in generators if name]
    if not generators:
        return "BPSK"
    return generators[int(rng.integers(0, len(generators)))]


def _constellation_for(generator: str) -> np.ndarray:
    if generator == "BPSK":
        return np.array([-1.0, 1.0], dtype=np.complex64)
    if generator == "QPSK":
        return np.array(
            [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j],
            dtype=np.complex64,
        ) / np.sqrt(2.0)
    if generator == "8PSK":
        angles = np.arange(8, dtype=float) * (2.0 * np.pi / 8.0)
        return np.exp(1j * angles).astype(np.complex64)
    if generator == "QAM16":
        levels = np.array([-3, -1, 1, 3], dtype=float)
        constellation = [complex(i, q) for i in levels for q in levels]
        arr = np.asarray(constellation, dtype=np.complex64)
        return arr / np.sqrt(np.mean(np.abs(arr) ** 2))
    if generator == "QAM64":
        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=float)
        constellation = [complex(i, q) for i in levels for q in levels]
        arr = np.asarray(constellation, dtype=np.complex64)
        return arr / np.sqrt(np.mean(np.abs(arr) ** 2))
    return np.array([-1.0, 1.0], dtype=np.complex64)


def _symbol_rate_for(generator: str, sample_rate: int, rng: np.random.Generator) -> int:
    if generator == "OFDM":
        return max(8, sample_rate // 128)
    if generator == "LFM":
        return max(8, sample_rate // 256)
    if generator in {"QAM64", "QAM16"}:
        return max(8, sample_rate // int(rng.integers(48, 96)))
    return max(8, sample_rate // int(rng.integers(64, 160)))


def _rrc_taps(samples_per_symbol: int, span: int = 8, beta: float = 0.35) -> np.ndarray:
    num_taps = span * samples_per_symbol + 1
    t = np.arange(num_taps, dtype=float) - (num_taps - 1) / 2.0
    t = t / max(samples_per_symbol, 1)
    taps = np.zeros_like(t)

    for idx, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            taps[idx] = 1.0 - beta + (4.0 * beta / np.pi)
        elif np.isclose(abs(ti), 1.0 / (4.0 * beta)):
            taps[idx] = (
                beta
                / np.sqrt(2.0)
                * (
                    (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                    + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
                )
            )
        else:
            numerator = (
                np.sin(np.pi * ti * (1.0 - beta))
                + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta))
            )
            denominator = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            taps[idx] = numerator / denominator

    taps = taps.astype(np.float32)
    taps /= np.sqrt(np.sum(taps**2))
    return taps


def _pulse_shape(symbols: np.ndarray, samples_per_symbol: int) -> np.ndarray:
    upsampled = np.zeros(len(symbols) * samples_per_symbol, dtype=np.complex64)
    upsampled[::samples_per_symbol] = symbols
    taps = _rrc_taps(samples_per_symbol)
    shaped = lfilter(taps, [1.0], upsampled)
    return shaped.astype(np.complex64)


def _generate_baseband(generator: str, sample_len: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    symbol_rate = _symbol_rate_for(generator, sample_rate, rng)
    samples_per_symbol = max(2, sample_rate // symbol_rate)
    num_symbols = max(8, int(np.ceil(sample_len / samples_per_symbol)) + 4)

    if generator == "LFM":
        t = np.arange(sample_len, dtype=float) / sample_rate
        sweep_hz = float(rng.uniform(0.05, 0.28) * sample_rate)
        chirp_rate = sweep_hz / max(t[-1], 1e-9)
        phase = 2.0 * np.pi * (0.5 * chirp_rate * t**2)
        return np.exp(1j * phase).astype(np.complex64)

    if generator == "OFDM":
        n_subcarriers = int(rng.integers(8, 24))
        fft_size = 1
        while fft_size < n_subcarriers * 2:
            fft_size *= 2
        symbols = np.zeros((num_symbols, fft_size), dtype=np.complex64)
        active_bins = np.linspace(1, fft_size // 2 - 1, n_subcarriers, dtype=int)
        for row in symbols:
            row[active_bins] = _constellation_for("QPSK")[rng.integers(0, 4, size=n_subcarriers)]
            row[-active_bins] = np.conj(row[active_bins])
        time_domain = np.fft.ifft(symbols, axis=1).reshape(-1)
        if len(time_domain) < sample_len:
            time_domain = np.pad(time_domain, (0, sample_len - len(time_domain)))
        return time_domain[:sample_len].astype(np.complex64)

    constellation = _constellation_for(generator)
    indices = rng.integers(0, len(constellation), size=num_symbols)
    symbols = constellation[indices]
    shaped = _pulse_shape(symbols, samples_per_symbol)
    if len(shaped) < sample_len:
        shaped = np.pad(shaped, (0, sample_len - len(shaped)))
    return shaped[:sample_len].astype(np.complex64)


def _apply_burst_envelope(signal: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, object]]:
    length = len(signal)
    active_len = int(rng.integers(max(16, length // 2), length))
    start = int(rng.integers(0, max(1, length - active_len + 1)))
    stop = min(length, start + active_len)
    window = np.zeros(length, dtype=np.float32)
    ramp = max(4, min(32, active_len // 8))
    window[start:stop] = 1.0
    if ramp * 2 < active_len:
        fade = np.linspace(0.0, 1.0, ramp, endpoint=False, dtype=np.float32)
        window[start : start + ramp] *= fade
        window[stop - ramp : stop] *= fade[::-1]
    metadata = {"burst_start": start, "burst_stop": stop}
    return (signal * window).astype(np.complex64), metadata


def _upconvert_to_center_frequency(signal: np.ndarray, config: AppConfig) -> tuple[np.ndarray, dict[str, object]]:
    sample_rate = int(config.global_params.get("sample_rate", 1_000_000))
    sample_len = len(signal)
    center_frequency_hz = float(config.global_params.get("center_frequency_hz", 0.0))
    t = np.arange(sample_len, dtype=float) / sample_rate
    shifted = signal * np.exp(1j * 2.0 * np.pi * center_frequency_hz * t)
    return shifted.astype(np.complex64), {"center_frequency_hz": center_frequency_hz}


def _apply_channel_effects(signal: np.ndarray, config: AppConfig, sample_index: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, object]]:
    sample_rate = int(config.global_params.get("sample_rate", 1_000_000))
    sample_len = len(signal)
    t = np.arange(sample_len, dtype=float) / sample_rate

    snr_db = config.global_params.get("snr_db", [0, 30])
    if isinstance(snr_db, (list, tuple)) and len(snr_db) >= 2:
        snr_mid = float(snr_db[0] + snr_db[1]) / 2.0
    else:
        snr_mid = 15.0

    frequency_offset = float(rng.uniform(-0.04, 0.04) * sample_rate)
    phase_offset = float(rng.uniform(-np.pi, np.pi))
    timing_jitter = float(rng.uniform(-0.02, 0.02))

    shifted = signal * np.exp(1j * (2.0 * np.pi * frequency_offset * t + phase_offset))
    shifted = np.roll(shifted, int(timing_jitter * sample_len))

    amplitude_imbalance = 1.0 + rng.uniform(-0.12, 0.12)
    phase_imbalance = rng.uniform(-0.08, 0.08)
    i = shifted.real * amplitude_imbalance
    q = shifted.imag * (2.0 - amplitude_imbalance)
    rotated_i = i * np.cos(phase_imbalance) - q * np.sin(phase_imbalance)
    rotated_q = i * np.sin(phase_imbalance) + q * np.cos(phase_imbalance)
    impaired = rotated_i + 1j * rotated_q

    power = np.mean(np.abs(impaired) ** 2) + 1e-9
    noise_power = power / (10 ** (snr_mid / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(sample_len) + 1j * rng.standard_normal(sample_len)
    )
    impaired = impaired + noise

    clip_level = float(rng.uniform(1.25, 2.5) * np.sqrt(power))
    impaired = np.clip(impaired.real, -clip_level, clip_level) + 1j * np.clip(
        impaired.imag, -clip_level, clip_level
    )

    return impaired.astype(np.complex64), {
        "frequency_offset_hz": frequency_offset,
        "phase_offset_rad": phase_offset,
        "timing_jitter_fraction": timing_jitter,
        "snr_db": snr_mid,
        "clip_level": clip_level,
    }


def synthesize_sample(config: AppConfig, sample_index: int) -> SynthSample:
    rng = _sample_rng(config, sample_index)
    sample_len = int(config.global_params.get("sample_len", 1024))
    sample_rate = int(config.global_params.get("sample_rate", 1_000_000))
    generator = _choose_generator(config, rng)

    clean = _generate_baseband(generator, sample_len, sample_rate, rng)
    clean, burst_meta = _apply_burst_envelope(clean, rng)
    clean, center_meta = _upconvert_to_center_frequency(clean, config)
    impaired, impairment_meta = _apply_channel_effects(clean, config, sample_index, rng)

    metadata = {
        "sample_index": sample_index,
        "generator": generator,
        "sample_rate": sample_rate,
        "sample_len": sample_len,
        "center_frequency_hz": center_meta["center_frequency_hz"],
        "burst": burst_meta,
        "impairments": impairment_meta,
    }

    return SynthSample(generator=generator, clean=clean, impaired=impaired, metadata=metadata)


def synthesize_dataset_pair(config: AppConfig, sample_index: int) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    sample = synthesize_sample(config, sample_index)
    return sample.clean, sample.impaired, sample.metadata
