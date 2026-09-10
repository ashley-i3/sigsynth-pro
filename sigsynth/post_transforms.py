from __future__ import annotations

import numpy as np
from scipy.signal import stft

from sigsynth.models import AppConfig


NUMPY_POST_TRANSFORMS = {
    "AWGN",
    "FreqOffset",
    "IQImbalance",
    "ChirpFlatten",
}


def _transform_rng(config: AppConfig, salt: int, sample_index: int = 0) -> np.random.Generator:
    seed_value = config.global_params.get("seed", 0)
    try:
        base_seed = int(seed_value)
    except (TypeError, ValueError):
        base_seed = 0

    mixed_seed = (
        base_seed * 2654435761
        + max(0, int(sample_index)) * 65537
        + salt * 131071
    )
    return np.random.default_rng(mixed_seed)


def _sample_snr_db(config: AppConfig, sample_index: int = 0, salt: int = 0) -> float:
    snr_db = config.global_params.get("snr_db", [0, 30])
    if isinstance(snr_db, (list, tuple)) and len(snr_db) >= 2:
        snr_min = float(snr_db[0])
        snr_max = float(snr_db[1])
        if snr_min > snr_max:
            snr_min, snr_max = snr_max, snr_min
        if np.isclose(snr_min, snr_max):
            return snr_min
        rng = _transform_rng(config, salt=17 + salt, sample_index=sample_index)
        return float(rng.uniform(snr_min, snr_max))
    if isinstance(snr_db, (int, float)):
        return float(snr_db)
    return 15.0


def apply_awgn(signal: np.ndarray, config: AppConfig, sample_index: int = 0) -> np.ndarray:
    snr_mid = _sample_snr_db(config, sample_index=sample_index, salt=len(signal))
    power = np.mean(np.abs(signal) ** 2)
    noise_power = power / (10 ** (snr_mid / 10))
    rng = _transform_rng(config, salt=len(signal), sample_index=sample_index)
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
    )
    return (signal + noise).astype(np.complex64)


def apply_freq_offset(signal: np.ndarray, config: AppConfig, sample_index: int = 0) -> np.ndarray:
    sample_rate = int(config.global_params.get("sample_rate", 1_000_000))
    sample_len = len(signal)
    offset_limit = min(max(sample_rate * 0.02, 1_000.0), sample_rate / 8.0)
    rng = _transform_rng(config, salt=5 * max(1, sample_len), sample_index=sample_index)
    offset_hz = float(rng.uniform(-offset_limit, offset_limit))
    t = np.arange(sample_len, dtype=float) / sample_rate
    return (signal * np.exp(1j * 2 * np.pi * offset_hz * t)).astype(np.complex64)


def apply_iq_imbalance(signal: np.ndarray, config: AppConfig, sample_index: int = 0) -> np.ndarray:
    rng = _transform_rng(config, salt=len(signal) * 3, sample_index=sample_index)
    i_gain = 1.0 + rng.uniform(-0.15, 0.15)
    q_gain = 1.0 + rng.uniform(-0.15, 0.15)
    phase = rng.uniform(-0.08, 0.08)
    i = signal.real * i_gain
    q = signal.imag * q_gain
    rotated_i = i * np.cos(phase) - q * np.sin(phase)
    rotated_q = i * np.sin(phase) + q * np.cos(phase)
    dc = 0.03 * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))
    return (rotated_i + 1j * rotated_q + dc).astype(np.complex64)


def apply_chirp_flatten(signal: np.ndarray, config: AppConfig, sample_index: int = 0) -> np.ndarray:
    sample_rate = int(config.global_params.get("sample_rate", 1_000_000))
    t = np.arange(len(signal), dtype=float) / sample_rate
    rng = _transform_rng(config, salt=7 * max(1, len(signal)), sample_index=sample_index)
    flatten_rate = float(sample_rate * rng.uniform(0.01, 0.03))
    return (signal * np.exp(-1j * 2 * np.pi * (0.5 * flatten_rate * t**2))).astype(np.complex64)


def apply_complex_to_real_magnitude(signal: np.ndarray) -> np.ndarray:
    return np.abs(signal).astype(np.float32)


def apply_spectrogram(signal: np.ndarray, config: AppConfig) -> np.ndarray:
    signal_len = max(1, len(signal))
    fft_size = int(config.global_params.get("sample_len", 1024))
    fft_size = max(64, min(512, fft_size // 8 if fft_size > 512 else fft_size))
    fft_size = min(fft_size, signal_len)
    if fft_size < 2:
        return np.zeros((1, signal_len), dtype=np.float32)

    noverlap = min(fft_size - 1, max(0, int(fft_size * 0.75)))
    _, _, spec = stft(
        signal,
        nperseg=fft_size,
        noverlap=noverlap,
        boundary=None,
        return_onesided=False,
    )
    if spec.size == 0:
        return np.zeros((1, signal_len), dtype=np.float32)
    magnitude_db = 20.0 * np.log10(np.abs(spec) + 1e-9)
    magnitude_db -= np.max(magnitude_db)
    return magnitude_db.astype(np.float32)


def apply_post_transform(
    name: str,
    signal: np.ndarray,
    config: AppConfig,
    sample_index: int = 0,
) -> np.ndarray:
    if name == "AWGN":
        return apply_awgn(signal, config, sample_index=sample_index)
    if name == "FreqOffset":
        return apply_freq_offset(signal, config, sample_index=sample_index)
    if name == "IQImbalance":
        return apply_iq_imbalance(signal, config, sample_index=sample_index)
    if name == "ChirpFlatten":
        return apply_chirp_flatten(signal, config, sample_index=sample_index)
    if name == "ComplexToRealMagnitude":
        return apply_complex_to_real_magnitude(signal)
    if name == "Spectrogram":
        return apply_spectrogram(signal, config)
    return signal
