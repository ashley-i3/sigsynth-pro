from __future__ import annotations

import numpy as np
from scipy.signal import stft

from sigsynth.models import AppConfig


NUMPY_POST_TRANSFORMS = {
    "AWGN",
    "FreqOffset",
    "IQImbalance",
    "ChirpFlatten",
    "RandomPhaseShift",
    "RandomTimeShift",
    "RayleighFadingChannel",
    "RandomResample",
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


def apply_random_phase_shift(signal: np.ndarray, config: AppConfig) -> np.ndarray:
    """Apply random phase shift to complex IQ signal (Sig53 level 2: -1 to 1 rad)."""
    rng = np.random.default_rng(
        int(config.global_params.get("seed", 0)) * 2654435761 + len(signal) * 5
    )
    phase_shift = rng.uniform(-1.0, 1.0)
    return (signal * np.exp(1j * phase_shift)).astype(np.complex64)


def apply_random_time_shift(signal: np.ndarray, config: AppConfig) -> np.ndarray:
    """Apply random circular time shift (Sig53 level 2: -32 to 32 samples)."""
    rng = np.random.default_rng(
        int(config.global_params.get("seed", 0)) * 2654435761 + len(signal) * 7
    )
    shift_samples = rng.integers(-32, 33)
    return np.roll(signal, shift_samples).astype(np.complex64)


def apply_rayleigh_fading(signal: np.ndarray, config: AppConfig) -> np.ndarray:
    """Apply Rayleigh fading channel (Sig53 level 2: 0.05-0.5 spread, PDP=(1.0, 0.5, 0.1))."""
    rng = np.random.default_rng(
        int(config.global_params.get("seed", 0)) * 2654435761 + len(signal) * 11
    )
    # Sig53 parameters
    spread_fraction = rng.uniform(0.05, 0.5)
    power_delay_profile = np.array([1.0, 0.5, 0.1], dtype=float)

    num_taps = len(power_delay_profile)
    max_delay = max(1, int(spread_fraction * len(signal)))

    # Generate tap delays and Rayleigh-distributed gains
    delays = np.sort(rng.integers(0, max_delay, size=num_taps))
    # Rayleigh distributed gains (magnitude of complex Gaussian)
    gains = (rng.normal(0, 1, size=num_taps) + 1j * rng.normal(0, 1, size=num_taps))
    gains = gains * np.sqrt(power_delay_profile) / np.sqrt(np.sum(power_delay_profile))

    # Apply multipath
    output = np.zeros_like(signal)
    for delay, gain in zip(delays, gains):
        if delay == 0:
            output += gain * signal
        elif delay < len(signal):
            output[delay:] += gain * signal[:-delay]

    return output.astype(np.complex64)


def apply_random_resample(signal: np.ndarray, config: AppConfig) -> np.ndarray:
    """Resample signal by random factor (Sig53 level 2: 0.75-1.5)."""
    from scipy.signal import resample_poly

    rng = np.random.default_rng(
        int(config.global_params.get("seed", 0)) * 2654435761 + len(signal) * 13
    )
    rate = rng.uniform(0.75, 1.5)
    target_len = len(signal)

    # Convert rate to rational approximation
    up = int(rate * 100)
    down = 100

    # Resample
    resampled = resample_poly(signal, up, down)

    # Crop or pad to target length
    if len(resampled) > target_len:
        return resampled[:target_len].astype(np.complex64)
    elif len(resampled) < target_len:
        return np.pad(resampled, (0, target_len - len(resampled)), mode='constant').astype(np.complex64)
    return resampled.astype(np.complex64)


def apply_post_transform(name: str, signal: np.ndarray, config: AppConfig) -> np.ndarray:
    if name == "AWGN":
        return apply_awgn(signal, config)
    if name == "FreqOffset":
        return apply_freq_offset(signal, config)
    if name == "IQImbalance":
        return apply_iq_imbalance(signal, config)
    if name == "ChirpFlatten":
        return apply_chirp_flatten(signal, config)
    if name == "RandomPhaseShift":
        return apply_random_phase_shift(signal, config)
    if name == "RandomTimeShift":
        return apply_random_time_shift(signal, config)
    if name == "RayleighFadingChannel":
        return apply_rayleigh_fading(signal, config)
    if name == "RandomResample":
        return apply_random_resample(signal, config)
    if name == "ComplexToRealMagnitude":
        return apply_complex_to_real_magnitude(signal)
    if name == "Spectrogram":
        return apply_spectrogram(signal, config)
    return signal
