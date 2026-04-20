from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
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
    base_seed = config.global_params.get("seed")
    try:
        seed_offset = 0 if base_seed is None else int(base_seed) * 2654435761
    except (TypeError, ValueError):
        seed_offset = 0
    seed = (
        seed_offset
        + int(config.dataset.total_samples) * 1009
        + int(config.global_params.get("sample_len", 1024)) * 917
        + sample_index * 65537
        + salt * 131071
    )
    return np.random.default_rng(seed=seed)


def _generator_choices(config: AppConfig) -> list[str]:
    generators = list(dict.fromkeys(name for name in config.generators if name))
    if not generators:
        return ["BPSK"]
    return generators


def _generator_weights(config: AppConfig, generators: list[str]) -> np.ndarray | None:
    weights = config.global_params.get("generator_weights")
    if not isinstance(weights, dict):
        return None

    resolved: list[float] = []
    has_weight = False
    for name in generators:
        weight = weights.get(name)
        if weight is None:
            weight = weights.get(_generator_family(name))
        if weight is None:
            weight = 1.0
        resolved.append(float(weight))
        has_weight = has_weight or float(weight) != 1.0

    if not has_weight:
        return None

    arr = np.asarray(resolved, dtype=float)
    total = float(np.sum(arr))
    if total <= 0:
        return None
    return arr / total


def _choose_generator(config: AppConfig, sample_index: int, rng: np.random.Generator) -> str:
    generators = _generator_choices(config)
    weights = _generator_weights(config, generators)
    if weights is not None:
        return str(rng.choice(generators, p=weights))

    class_distribution = str(config.global_params.get("class_distribution", "")).lower()
    if class_distribution == "uniform" and len(generators) > 1:
        schedule = _uniform_generator_schedule(
            tuple(generators),
            int(config.dataset.total_samples),
            _distribution_seed(config),
        )
        if schedule:
            return schedule[sample_index % len(schedule)]
    return generators[int(rng.integers(0, len(generators)))]


def _distribution_seed(config: AppConfig) -> int:
    configured_seed = config.global_params.get("seed")
    if configured_seed is not None:
        try:
            return int(configured_seed)
        except (TypeError, ValueError):
            pass

    seed_material = "|".join(
        [
            str(config.dataset.total_samples),
            str(config.dataset.train_ratio),
            str(config.dataset.output_format),
            str(config.global_params.get("sample_rate", 1_000_000)),
            str(config.global_params.get("sample_len", 1024)),
            str(config.global_params.get("class_distribution", "unset")),
            ",".join(dict.fromkeys(name for name in config.generators if name)),
        ]
    )
    return int.from_bytes(sha256(seed_material.encode("utf-8")).digest()[:8], "little")


@lru_cache(maxsize=64)
def _uniform_generator_schedule(generators: tuple[str, ...], total_samples: int, seed: int) -> tuple[str, ...]:
    if not generators or total_samples <= 0:
        return tuple()

    base_count, remainder = divmod(total_samples, len(generators))
    schedule: list[str] = []
    for index, name in enumerate(generators):
        repeats = base_count + (1 if index < remainder else 0)
        schedule.extend([name] * repeats)

    rng = np.random.default_rng(seed)
    rng.shuffle(schedule)
    return tuple(schedule)


@lru_cache(maxsize=64)
def _uniform_component_schedule(
    generators: tuple[str, ...], total_components: int, seed: int
) -> tuple[str, ...]:
    if not generators or total_components <= 0:
        return tuple()

    base_count, remainder = divmod(total_components, len(generators))
    schedule: list[str] = []
    for index, name in enumerate(generators):
        repeats = base_count + (1 if index < remainder else 0)
        schedule.extend([name] * repeats)

    rng = np.random.default_rng(seed ^ 0x5A5A5A5A)
    rng.shuffle(schedule)
    return tuple(schedule)


@lru_cache(maxsize=64)
def _component_count_schedule(total_samples: int, minimum: int, maximum: int, seed: int) -> tuple[int, ...]:
    if total_samples <= 0:
        return tuple()
    rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
    return tuple(int(x) for x in rng.integers(minimum, maximum + 1, size=total_samples))


def _generator_family(generator: str) -> str:
    return resolve_generator_name(generator) or generator


def _order_from_name(name: str, default: int) -> int:
    digits = "".join(ch for ch in name if ch.isdigit())
    if digits:
        return max(2, int(digits))
    return default


def _constellation_for(generator: str) -> np.ndarray:
    family = _generator_family(generator)
    if family == "BPSK":
        return np.array([-1.0, 1.0], dtype=np.complex64)
    if family == "QPSK":
        return np.array(
            [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j],
            dtype=np.complex64,
        ) / np.sqrt(2.0)
    if family in {"8PSK", "PSK"}:
        order = 8 if family == "8PSK" else _order_from_name(generator, 8)
        angles = np.arange(order, dtype=float) * (2.0 * np.pi / order)
        return np.exp(1j * angles).astype(np.complex64)
    if family in {"QAM16", "QAM64", "QAM"}:
        order = _order_from_name(generator, 64 if family == "QAM64" else 16)
        side = int(np.sqrt(order))
        if side * side != order:
            side = int(2 ** np.ceil(np.log2(np.sqrt(order))))
            order = side * side
        levels = np.arange(-(side - 1), side, 2, dtype=float)
        constellation = [complex(i, q) for i in levels for q in levels]
        arr = np.asarray(constellation, dtype=np.complex64)
        return arr / np.sqrt(np.mean(np.abs(arr) ** 2))
    if family in {"ASK"}:
        order = _order_from_name(generator, 16)
        levels = np.arange(1, order + 1, dtype=float)
        levels = levels - np.mean(levels)
        levels = levels / np.sqrt(np.mean(levels**2))
        return levels.astype(np.complex64)
    if family == "PAM":
        order = _order_from_name(generator, 16)
        # Unipolar: 0 to 1 (like original Sig53 PAM)
        levels = np.linspace(0, 1, order, dtype=float)
        # Normalize to unit power
        levels = levels / np.sqrt(np.mean(levels**2))
        return levels.astype(np.complex64)
    if family in {"Tone", "AM", "FM", "OFDM", "ChirpSS", "LFM"}:
        return np.array([1.0], dtype=np.complex64)
    if family == "OOK":
        return np.array([0.0, 1.0], dtype=np.complex64)
    if family in {"FSK", "GFSK", "MSK", "GMSK"}:
        order = _order_from_name(generator, 4)
        return np.arange(order, dtype=np.float32).astype(np.complex64)
    return np.array([-1.0, 1.0], dtype=np.complex64)


def _symbol_rate_for(generator: str, sample_rate: int, rng: np.random.Generator) -> int:
    family = _generator_family(generator)
    if family == "OFDM":
        return max(8, sample_rate // 128)
    if family in {"LFM", "ChirpSS"}:
        return max(8, sample_rate // 256)
    if family in {"QAM64", "QAM16", "QAM"}:
        return max(8, sample_rate // int(rng.integers(48, 96)))
    if family in {"ASK", "PAM", "PSK", "FSK", "GFSK", "MSK", "GMSK", "AM", "FM", "OOK", "Tone"}:
        return max(8, sample_rate // int(rng.integers(48, 128)))
    return max(8, sample_rate // int(rng.integers(64, 160)))


def _rrc_taps(samples_per_symbol: int, span: int = 11, beta: float = 0.35) -> np.ndarray:
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
    family = _generator_family(generator)
    symbol_rate = _symbol_rate_for(generator, sample_rate, rng)
    samples_per_symbol = max(2, sample_rate // symbol_rate)
    num_symbols = max(8, int(np.ceil(sample_len / samples_per_symbol)) + 4)

    if family == "Tone":
        tone_hz = float(rng.uniform(-0.08, 0.08) * sample_rate)
        t = np.arange(sample_len, dtype=float) / sample_rate
        return np.exp(1j * 2.0 * np.pi * tone_hz * t).astype(np.complex64)

    if family == "FM":
        t = np.arange(sample_len, dtype=float) / sample_rate
        mod = rng.standard_normal(sample_len).astype(np.float32)
        mod = np.convolve(mod, np.ones(16) / 16.0, mode="same")
        deviation_hz = float(rng.uniform(0.02, 0.12) * sample_rate)
        phase = 2.0 * np.pi * np.cumsum(mod) * deviation_hz / sample_rate
        return np.exp(1j * phase).astype(np.complex64)

    if family in {"AM", "OOK"}:
        t = np.arange(sample_len, dtype=float) / sample_rate
        carrier_hz = float(rng.uniform(-0.06, 0.06) * sample_rate)
        symbol_stream = rng.integers(0, 2, size=num_symbols).astype(np.float32)
        if family == "AM":
            symbol_stream = 0.4 + 0.6 * symbol_stream
        shaped = np.repeat(symbol_stream, samples_per_symbol)
        if len(shaped) < sample_len:
            shaped = np.pad(shaped, (0, sample_len - len(shaped)), constant_values=shaped[-1] if len(shaped) else 0.0)
        envelope = shaped[:sample_len]
        return (envelope * np.exp(1j * 2.0 * np.pi * carrier_hz * t)).astype(np.complex64)

    if family == "ASK":
        t = np.arange(sample_len, dtype=float) / sample_rate
        order = _order_from_name(generator, 16)
        amplitudes = np.linspace(0.2, 1.2, order, dtype=np.float32)
        amplitudes = amplitudes - np.mean(amplitudes)
        amplitudes = amplitudes / (np.sqrt(np.mean(amplitudes**2)) + 1e-9)
        symbol_indices = rng.integers(0, order, size=num_symbols)
        envelope = np.repeat(amplitudes[symbol_indices], samples_per_symbol)
        if len(envelope) < sample_len:
            envelope = np.pad(envelope, (0, sample_len - len(envelope)), mode="edge")
        carrier_hz = float(rng.uniform(-0.08, 0.08) * sample_rate)
        return (envelope[:sample_len] * np.exp(1j * 2.0 * np.pi * carrier_hz * t)).astype(np.complex64)

    if family in {"FSK", "GFSK", "MSK", "GMSK"}:
        order = _order_from_name(generator, 4)
        t = np.arange(sample_len, dtype=float) / sample_rate
        symbols = rng.integers(0, order, size=num_symbols).astype(np.float32)
        freq_levels = np.linspace(-0.12, 0.12, order, dtype=np.float32) * sample_rate
        freq = np.repeat(freq_levels[symbols.astype(int)], samples_per_symbol)
        if len(freq) < sample_len:
            freq = np.pad(freq, (0, sample_len - len(freq)), mode="edge")
        freq = freq[:sample_len]
        if family in {"GFSK", "GMSK"}:
            freq = np.convolve(freq, np.ones(9) / 9.0, mode="same")
        phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate
        return np.exp(1j * phase).astype(np.complex64)

    if family == "LFM":
        t = np.arange(sample_len, dtype=float) / sample_rate
        sweep_hz = float(rng.uniform(0.05, 0.28) * sample_rate)
        chirp_rate = sweep_hz / max(t[-1], 1e-9)
        phase = 2.0 * np.pi * (0.5 * chirp_rate * t**2)
        return np.exp(1j * phase).astype(np.complex64)

    if family == "ChirpSS":
        t = np.arange(sample_len, dtype=float) / sample_rate
        sweep_hz = float(rng.uniform(0.08, 0.18) * sample_rate)
        phase = 2.0 * np.pi * (0.5 * (sweep_hz / max(t[-1], 1e-9)) * t**2)
        return np.exp(1j * phase).astype(np.complex64)

    if family == "OFDM":
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
    min_components = int(config.global_params.get("num_signals_min", 1))
    max_components = int(config.global_params.get("num_signals_max", max(1, min_components)))
    if max_components < min_components:
        max_components = min_components

    component_counts = _component_count_schedule(
        int(config.dataset.total_samples),
        min_components,
        max_components,
        _distribution_seed(config),
    )
    component_count = component_counts[sample_index % len(component_counts)] if component_counts else 1

    generators = _generator_choices(config)
    class_distribution = str(config.global_params.get("class_distribution", "")).lower()
    weights = _generator_weights(config, generators)
    start_slot = sum(component_counts[:sample_index]) if component_counts else 0

    if weights is None and class_distribution == "uniform" and len(generators) > 1:
        total_components = sum(component_counts) if component_counts else component_count
        component_schedule = _uniform_component_schedule(
            tuple(generators),
            total_components,
            _distribution_seed(config),
        )
    else:
        component_schedule = tuple()

    clean = np.zeros(sample_len, dtype=np.complex64)
    components: list[dict[str, object]] = []
    for component_index in range(component_count):
        component_rng = _sample_rng(config, sample_index, salt=component_index + 1)
        if component_schedule:
            generator = component_schedule[(start_slot + component_index) % len(component_schedule)]
        else:
            generator = _choose_generator(config, sample_index + component_index, component_rng)
        family = _generator_family(generator)
        baseband = _generate_baseband(generator, sample_len, sample_rate, component_rng)
        baseband, burst_meta = _apply_burst_envelope(baseband, component_rng)

        component_center = float(
            np.clip(
                config.global_params.get("center_frequency_hz", 0.0)
                + component_rng.uniform(-0.32, 0.32) * sample_rate,
                -sample_rate / 2.0,
                sample_rate / 2.0,
            )
        )
        t = np.arange(sample_len, dtype=float) / sample_rate
        component_clean = baseband * np.exp(1j * 2.0 * np.pi * component_center * t)
        clean += component_clean.astype(np.complex64)
        components.append(
            {
                "generator": generator,
                "family": family,
                "center_frequency_hz": component_center,
                "burst": burst_meta,
            }
        )

    primary_generator = components[0]["generator"] if components else _choose_generator(config, sample_index, rng)
    primary_family = _generator_family(str(primary_generator))
    impaired, impairment_meta = _apply_channel_effects(clean, config, sample_index, rng)

    metadata = {
        "sample_index": sample_index,
        "generator": primary_generator,
        "family": primary_family,
        "sample_rate": sample_rate,
        "sample_len": sample_len,
        "class_distribution": config.global_params.get("class_distribution"),
        "num_components": component_count,
        "components": components,
        "impairments": impairment_meta,
    }

    return SynthSample(generator=str(primary_generator), clean=clean, impaired=impaired, metadata=metadata)


def synthesize_dataset_pair(config: AppConfig, sample_index: int) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    sample = synthesize_sample(config, sample_index)
    return sample.clean, sample.impaired, sample.metadata
