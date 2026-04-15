from __future__ import annotations

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
