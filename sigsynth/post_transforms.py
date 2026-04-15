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


def apply_awgn(signal: np.ndarray, config: AppConfig) -> np.ndarray:
    snr_db = config.global_params.get("snr_db", [0, 30])
    if isinstance(snr_db, (list, tuple)) and len(snr_db) >= 2:
        snr_mid = float(snr_db[0] + snr_db[1]) / 2.0
    else:
        snr_mid = 15.0
    power = np.mean(np.abs(signal) ** 2)
    noise_power = power / (10 ** (snr_mid / 10))
    rng = np.random.default_rng(
        int(config.global_params.get("seed", 0)) * 2654435761 + len(signal)
    )
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
    )
    return (signal + noise).astype(np.complex64)


def apply_freq_offset(signal: np.ndarray, config: AppConfig) -> np.ndarray:
    sample_rate = int(config.global_params.get("sample_rate", 1_000_000))
    sample_len = len(signal)
    offset_hz = min(max(sample_rate * 0.02, 1_000.0), sample_rate / 8.0)
    t = np.arange(sample_len, dtype=float) / sample_rate
    return (signal * np.exp(1j * 2 * np.pi * offset_hz * t)).astype(np.complex64)


def apply_iq_imbalance(signal: np.ndarray, config: AppConfig) -> np.ndarray:
    rng = np.random.default_rng(
        int(config.global_params.get("seed", 0)) * 2654435761 + len(signal) * 3
    )
    amplitude = 1.0 + rng.uniform(-0.15, 0.15)
    phase = rng.uniform(-0.08, 0.08)
    i = signal.real * amplitude
    q = signal.imag * (2.0 - amplitude)
    rotated_i = i * np.cos(phase) - q * np.sin(phase)
    rotated_q = i * np.sin(phase) + q * np.cos(phase)
    dc = 0.03 * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))
    return (rotated_i + 1j * rotated_q + dc).astype(np.complex64)


def apply_chirp_flatten(signal: np.ndarray, config: AppConfig) -> np.ndarray:
    sample_rate = int(config.global_params.get("sample_rate", 1_000_000))
    t = np.arange(len(signal), dtype=float) / sample_rate
    flatten_rate = sample_rate * 0.02
    return (signal * np.exp(-1j * 2 * np.pi * (0.5 * flatten_rate * t**2))).astype(np.complex64)


def apply_complex_to_real_magnitude(signal: np.ndarray) -> np.ndarray:
    return np.abs(signal).astype(np.float32)


def apply_spectrogram(signal: np.ndarray, config: AppConfig) -> np.ndarray:
    fft_size = int(config.global_params.get("sample_len", 1024))
    fft_size = max(64, min(512, fft_size // 8 if fft_size > 512 else fft_size))
    _, _, spec = stft(
        signal,
        nperseg=fft_size,
        noverlap=max(1, int(fft_size * 0.75)),
        boundary=None,
        return_onesided=False,
    )
    magnitude_db = 20.0 * np.log10(np.abs(spec) + 1e-9)
    magnitude_db -= np.max(magnitude_db)
    return magnitude_db.astype(np.float32)


def apply_post_transform(name: str, signal: np.ndarray, config: AppConfig) -> np.ndarray:
    if name == "AWGN":
        return apply_awgn(signal, config)
    if name == "FreqOffset":
        return apply_freq_offset(signal, config)
    if name == "IQImbalance":
        return apply_iq_imbalance(signal, config)
    if name == "ChirpFlatten":
        return apply_chirp_flatten(signal, config)
    if name == "ComplexToRealMagnitude":
        return apply_complex_to_real_magnitude(signal)
    if name == "Spectrogram":
        return apply_spectrogram(signal, config)
    return signal
