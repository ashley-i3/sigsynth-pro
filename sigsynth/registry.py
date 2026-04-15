from __future__ import annotations

import re

from sigsynth.models import GeneratorMeta, TransformMeta


GENERATOR_REGISTRY: dict[str, GeneratorMeta] = {
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
    return GENERATOR_ALIASES.get(_normalize_registry_key(name))


def resolve_transform_name(name: str) -> str | None:
    """Resolve a transform name using case-insensitive and punctuation-insensitive matching."""
    if name in TRANSFORM_REGISTRY:
        return name
    return TRANSFORM_ALIASES.get(_normalize_registry_key(name))
