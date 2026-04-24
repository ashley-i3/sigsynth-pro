from __future__ import annotations

import re

from sigsynth.models import GeneratorMeta, TransformMeta


TORCHSIG_CONCRETE_GENERATORS: list[str] = [
    "tone",
    "ofdm-64",
    "ofdm-72",
    "ofdm-128",
    "ofdm-180",
    "ofdm-256",
    "ofdm-300",
    "ofdm-512",
    "ofdm-600",
    "ofdm-900",
    "ofdm-1024",
    "ofdm-1200",
    "ofdm-2048",
    "lfm-data",
    "lfm-radar",
    "2fsk",
    "4fsk",
    "8fsk",
    "16fsk",
    "2gfsk",
    "4gfsk",
    "8gfsk",
    "16gfsk",
    "2msk",
    "4msk",
    "8msk",
    "16msk",
    "2gmsk",
    "4gmsk",
    "8gmsk",
    "16gmsk",
    "fm",
    "ook",
    "bpsk",
    "4pam",
    "8pam",
    "16pam",
    "32pam",
    "64pam",
    "qpsk",
    "8psk",
    "16psk",
    "32psk",
    "64psk",
    "4ask",
    "8ask",
    "16ask",
    "32ask",
    "64ask",
    "16qam",
    "32qam",
    "64qam",
    "256qam",
    "1024qam",
    "32qam_cross",
    "128qam_cross",
    "512qam_cross",
    "chirpss",
    "am-dsb",
    "am-dsb-sc",
    "am-usb",
    "am-lsb",
]

TORCHSIG_CONCRETE_SET = set(TORCHSIG_CONCRETE_GENERATORS)


# The canonical 53 modulations from the original Sig53 dataset
# Reference: torchsig/datasets/modulations.py ModulationsDataset.default_classes
SIG53_MODULATIONS: list[str] = [
    "ook", "bpsk", "4pam", "4ask", "qpsk", "8pam", "8ask", "8psk",
    "16qam", "16pam", "16ask", "16psk", "32qam", "32qam_cross",
    "32pam", "32ask", "32psk", "64qam", "64pam", "64ask", "64psk",
    "128qam_cross", "256qam", "512qam_cross", "1024qam",
    "2fsk", "2gfsk", "2msk", "2gmsk", "4fsk", "4gfsk", "4msk", "4gmsk",
    "8fsk", "8gfsk", "8msk", "8gmsk", "16fsk", "16gfsk", "16msk", "16gmsk",
    "ofdm-64", "ofdm-72", "ofdm-128", "ofdm-180", "ofdm-256", "ofdm-300",
    "ofdm-512", "ofdm-600", "ofdm-900", "ofdm-1024", "ofdm-1200", "ofdm-2048",
]

SIG53_MODULATIONS_SET = set(SIG53_MODULATIONS)


GENERATOR_REGISTRY: dict[str, GeneratorMeta] = {
    "Tone": GeneratorMeta(
        name="Tone",
        produces=["complex_iq"],
        requires=["baseband"],
        tags=["narrowband"],
    ),
    "BPSK": GeneratorMeta(
        name="BPSK",
        produces=["complex_iq"],
        requires=["baseband"],
        tags=["narrowband"],
    ),
    "QPSK": GeneratorMeta(
        name="QPSK",
        produces=["complex_iq"],
        requires=["baseband"],
        tags=["narrowband"],
    ),
    "8PSK": GeneratorMeta(
        name="8PSK",
        produces=["complex_iq"],
        requires=["baseband"],
        tags=["narrowband"],
    ),
    "QAM16": GeneratorMeta(
        name="QAM16",
        produces=["complex_iq"],
        requires=["baseband"],
        tags=["narrowband"],
    ),
    "QAM64": GeneratorMeta(
        name="QAM64",
        produces=["complex_iq"],
        requires=["baseband"],
        tags=["narrowband"],
    ),
    "PSK": GeneratorMeta(
        name="PSK",
        produces=["complex_iq"],
        requires=["baseband"],
        tags=["narrowband"],
    ),
    "ASK": GeneratorMeta(
        name="ASK",
        produces=["complex_iq"],
        requires=["baseband"],
        tags=["narrowband"],
    ),
    "PAM": GeneratorMeta(
        name="PAM",
        produces=["complex_iq"],
        requires=["baseband"],
        tags=["narrowband"],
    ),
    "QAM": GeneratorMeta(
        name="QAM",
        produces=["complex_iq"],
        requires=["baseband"],
        tags=["narrowband"],
    ),
    "FSK": GeneratorMeta(
        name="FSK",
        produces=["complex_iq"],
        requires=["baseband"],
        tags=["narrowband"],
    ),
    "GFSK": GeneratorMeta(
        name="GFSK",
        produces=["complex_iq"],
        requires=["baseband"],
        tags=["narrowband"],
    ),
    "MSK": GeneratorMeta(
        name="MSK",
        produces=["complex_iq"],
        requires=["baseband"],
        tags=["narrowband"],
    ),
    "GMSK": GeneratorMeta(
        name="GMSK",
        produces=["complex_iq"],
        requires=["baseband"],
        tags=["narrowband"],
    ),
    "AM": GeneratorMeta(
        name="AM",
        produces=["complex_iq"],
        requires=["baseband"],
        tags=["narrowband"],
    ),
    "FM": GeneratorMeta(
        name="FM",
        produces=["complex_iq"],
        requires=["baseband"],
        tags=["narrowband"],
    ),
    "OOK": GeneratorMeta(
        name="OOK",
        produces=["complex_iq"],
        requires=["baseband"],
        tags=["narrowband"],
    ),
    "LFM": GeneratorMeta(
        name="LFM",
        produces=["complex_iq"],
        requires=["baseband"],
        tags=["chirp_preserving", "wideband"],
        parameter_groups=["chirp"],
    ),
    "OFDM": GeneratorMeta(
        name="OFDM",
        produces=["complex_iq"],
        requires=["baseband"],
        tags=["wideband"],
    ),
    "ChirpSS": GeneratorMeta(
        name="ChirpSS",
        produces=["complex_iq"],
        requires=["baseband"],
        tags=["wideband"],
    ),
}


TRANSFORM_REGISTRY: dict[str, TransformMeta] = {
    "AWGN": TransformMeta(
        name="AWGN",
        accepts=["complex_iq"],
        produces=["complex_iq"],
        modifies=["snr"],
    ),
    "FreqOffset": TransformMeta(
        name="FreqOffset",
        accepts=["complex_iq"],
        produces=["complex_iq"],
        modifies=["center_frequency"],
    ),
    "IQImbalance": TransformMeta(
        name="IQImbalance",
        accepts=["complex_iq"],
        produces=["complex_iq"],
        modifies=["iq_balance"],
    ),
    "ComplexToRealMagnitude": TransformMeta(
        name="ComplexToRealMagnitude",
        accepts=["complex_iq"],
        produces=["real"],
    ),
    "Spectrogram": TransformMeta(
        name="Spectrogram",
        accepts=["complex_iq", "real"],
        produces=["image"],
    ),
    "ChirpFlatten": TransformMeta(
        name="ChirpFlatten",
        accepts=["complex_iq"],
        produces=["complex_iq"],
        constraints={"incompatible_with": ["chirp_preserving"]},
    ),
    "RandomPhaseShift": TransformMeta(
        name="RandomPhaseShift",
        accepts=["complex_iq"],
        produces=["complex_iq"],
        modifies=["phase"],
    ),
    "RandomTimeShift": TransformMeta(
        name="RandomTimeShift",
        accepts=["complex_iq"],
        produces=["complex_iq"],
        modifies=["start_time"],
    ),
    "RayleighFadingChannel": TransformMeta(
        name="RayleighFadingChannel",
        accepts=["complex_iq"],
        produces=["complex_iq"],
        modifies=["multipath"],
    ),
    "RandomResample": TransformMeta(
        name="RandomResample",
        accepts=["complex_iq"],
        produces=["complex_iq"],
        modifies=["sample_rate"],
    ),
}


def _normalize_registry_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


GENERATOR_ALIASES: dict[str, str] = {
    _normalize_registry_key(name): name for name in GENERATOR_REGISTRY
}


TRANSFORM_ALIASES: dict[str, str] = {
    _normalize_registry_key(name): name for name in TRANSFORM_REGISTRY
}


def resolve_generator_name(name: str) -> str | None:
    """Resolve a generator name using case-insensitive and punctuation-insensitive matching."""
    if name in GENERATOR_REGISTRY:
        return name
    normalized = _normalize_registry_key(name)
    if normalized in GENERATOR_ALIASES:
        return GENERATOR_ALIASES[normalized]
    concrete_map = {
        "tone": "Tone",
        "bpsk": "BPSK",
        "qpsk": "QPSK",
        "8psk": "8PSK",
        "16qam": "QAM16",
        "64qam": "QAM64",
        "4pam": "PAM",
        "8pam": "PAM",
        "16pam": "PAM",
        "32pam": "PAM",
        "64pam": "PAM",
        "4ask": "ASK",
        "8ask": "ASK",
        "16ask": "ASK",
        "32ask": "ASK",
        "64ask": "ASK",
        "2fsk": "FSK",
        "4fsk": "FSK",
        "8fsk": "FSK",
        "16fsk": "FSK",
        "2gfsk": "GFSK",
        "4gfsk": "GFSK",
        "8gfsk": "GFSK",
        "16gfsk": "GFSK",
        "2msk": "MSK",
        "4msk": "MSK",
        "8msk": "MSK",
        "16msk": "MSK",
        "2gmsk": "GMSK",
        "4gmsk": "GMSK",
        "8gmsk": "GMSK",
        "16gmsk": "GMSK",
        "fm": "FM",
        "ook": "OOK",
        "lfmdata": "LFM",
        "lfmradar": "LFM",
        "chirpss": "ChirpSS",
        "amdsb": "AM",
        "amdsbsc": "AM",
        "amusb": "AM",
        "amlsb": "AM",
        "ofdm64": "OFDM",
        "ofdm72": "OFDM",
        "ofdm128": "OFDM",
        "ofdm180": "OFDM",
        "ofdm256": "OFDM",
        "ofdm300": "OFDM",
        "ofdm512": "OFDM",
        "ofdm600": "OFDM",
        "ofdm900": "OFDM",
        "ofdm1024": "OFDM",
        "ofdm1200": "OFDM",
        "ofdm2048": "OFDM",
    }
    if normalized in concrete_map:
        return concrete_map[normalized]
    if normalized.startswith("ofdm"):
        return "OFDM"
    if normalized.startswith("lfm"):
        return "LFM"
    if normalized.startswith("chirpss"):
        return "ChirpSS"
    if normalized.startswith("am"):
        return "AM"
    if normalized == "fm":
        return "FM"
    if normalized == "ook":
        return "OOK"
    if normalized.endswith("gmsk"):
        return "GMSK"
    if normalized.endswith("gfsk"):
        return "GFSK"
    if normalized.endswith("msk"):
        return "MSK"
    if normalized.endswith("fsk"):
        return "FSK"
    if normalized.endswith("psk"):
        return "PSK"
    if normalized.endswith("pam"):
        return "PAM"
    if normalized.endswith("ask"):
        return "ASK"
    if "qamcross" in normalized or normalized.endswith("qam"):
        return "QAM"
    if normalized == "tone":
        return "Tone"
    return None


def is_torchsig_concrete_generator(name: str) -> bool:
    return _normalize_registry_key(name) in TORCHSIG_CONCRETE_SET


def to_torchsig_generator_name(name: str) -> str | None:
    normalized = _normalize_registry_key(name)
    if normalized in TORCHSIG_CONCRETE_SET:
        return normalized
    canonical = resolve_generator_name(name)
    if canonical is None:
        return None
    mapping = {
        "Tone": "tone",
        "BPSK": "bpsk",
        "QPSK": "qpsk",
        "8PSK": "8psk",
        "PSK": "8psk",
        "ASK": "16ask",
        "QAM16": "16qam",
        "QAM64": "64qam",
        "QAM": "16qam",
        "FSK": "4fsk",
        "GFSK": "4gfsk",
        "MSK": "4msk",
        "GMSK": "4gmsk",
        "AM": "am-dsb",
        "FM": "fm",
        "OOK": "ook",
        "LFM": "lfm-data",
        "OFDM": "ofdm-128",
        "ChirpSS": "chirpss",
    }
    return mapping.get(canonical)


def resolve_transform_name(name: str) -> str | None:
    """Resolve a transform name using case-insensitive and punctuation-insensitive matching."""
    if name in TRANSFORM_REGISTRY:
        return name
    return TRANSFORM_ALIASES.get(_normalize_registry_key(name))
