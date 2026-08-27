# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Curated neuron model family taxonomy

"""Curated family and category taxonomy for the neuron model library.

Unlike a keyword heuristic, this is an explicit, reviewable mapping: every model
is deliberately assigned to one family with a stable category slug. It is the
declarative discovery backbone consumed by the descriptor corpus and the
catalogue. Adding a model and forgetting to classify it is caught by the
taxonomy completeness check, so the taxonomy cannot silently fall behind the
registry.
"""

from __future__ import annotations

# family display name -> (category slug, member class names)
_FAMILIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "Integrate-and-Fire": (
        "integrate-and-fire",
        (
            "AdExNeuron",
            "AdaptiveThresholdIFNeuron",
            "AdaptiveThresholdMoENeuron",
            "ComplementaryLIFNeuron",
            "EnergyLIFNeuron",
            "SCNormalizedEnergyLIFNeuron",
            "ExpIFNeuron",
            "FractionalLIFNeuron",
            "GLIFNeuron",
            "GatedLIFNeuron",
            "InhibitoryLIFNeuron",
            "IntegerQIFNeuron",
            "KLIFNeuron",
            "LapicqueNeuron",
            "MATNeuron",
            "MihalasNieburNeuron",
            "NonResettingLIFNeuron",
            "SCNonResettingAdaptiveLIFNeuron",
            "SCSigmaDeltaAccumulatorNeuron",
            "NonlinearLIFNeuron",
            "ParametricLIFNeuron",
            "PerfectIntegratorNeuron",
            "QuadraticIFNeuron",
            "SCResettingMATNeuron",
            "SFANeuron",
            "StochasticIFNeuron",
            "StochasticLIFNeuron",
            "ThetaNeuron",
        ),
    ),
    "Resonate-and-Fire": (
        "resonate-and-fire",
        (
            "BalancedResonateAndFireNeuron",
            "Izhikevich2007Neuron",
            "ResonateAndFireNeuron",
        ),
    ),
    "Rate adaptation": (
        "rate-adaptation",
        (
            "BendaHerzNeuron",
            "SCStochasticRateAdaptationNeuron",
        ),
    ),
    "Conductance-based": (
        "conductance-based",
        (
            "AvRonCardiacNeuron",
            "ConnorStevensNeuron",
            "DestexheThalamicNeuron",
            "GolombFSNeuron",
            "GutkinErmentroutNeuron",
            "HayL5PyramidalNeuron",
            "HillTononiNeuron",
            "SCSixStateThalamocorticalNeuron",
            "HodgkinHuxleyNeuron",
            "HuberBraunNeuron",
            "MainenSejnowskiNeuron",
            "PospischilNeuron",
            "PrescottNeuron",
            "TraubMilesNeuron",
            "WangBuzsakiNeuron",
        ),
    ),
    "Compartmental": (
        "compartmental",
        (
            "BoothRinzelNeuron",
            "DendrifyNeuron",
            "MulticompartmentMCNNeuron",
            "PinskyRinzelNeuron",
            "RallCableNeuron",
            "TwoCompartmentLIFNeuron",
            "SCLeakyTwoCompartmentLIFNeuron",
            "SCExponentialTwoCompartmentLIFNeuron",
        ),
    ),
    "Ion-channel": (
        "ion-channel",
        (
            "ATypeKNeuron",
            "BKNeuron",
            "DendriticNMDANeuron",
            "IhNeuron",
            "NMDANeuron",
            "SCWBNMDAMagnesiumBlockNeuron",
            "PersistentNaNeuron",
            "SKNeuron",
            "TTypeCaNeuron",
        ),
    ),
    "Cerebellar": (
        "cerebellar",
        (
            "CerebellarBasketNeuron",
            "DCNNeuron",
            "DeSchutterPurkinjeNeuron",
            "GolgiCell",
            "GranuleCell",
            "LugaroCell",
            "StellateCell",
            "UnipolarBrushCell",
        ),
    ),
    "Motor system": (
        "motor-system",
        (
            "AlphaMotorNeuron",
            "GammaMotorNeuron",
            "MotorUnit",
            "RenshawCell",
            "UpperMotorNeuron",
        ),
    ),
    "Cortical interneuron": (
        "cortical-interneuron",
        (
            "ChandelierNeuron",
            "MartinottiNeuron",
            "PVFastSpikingNeuron",
            "SSTNeuron",
            "VIPNeuron",
        ),
    ),
    "Bursting": (
        "bursting",
        (
            "BertramPhantomBurster",
            "SCThreeStatePhantomBurster",
            "ButeraRespiratoryNeuron",
            "SCUnitCapacitanceRespiratoryNeuron",
            "ChayKeizerMinimalNeuron",
            "ChayKeizerNeuron",
            "ChayNeuron",
            "FitzHughRinzelNeuron",
            "HindmarshRoseNeuron",
            "MarderSTGNeuron",
            "PernarowskiNeuron",
            "PlantR15Neuron",
            "ShermanRinzelKeizerNeuron",
            "WilsonHRNeuron",
            "YamadaNeuron",
        ),
    ),
    "Oscillator": (
        "oscillator",
        (
            "FitzHughNagumoNeuron",
            "McKeanNeuron",
            "SCTriangularMcKeanNeuron",
            "MorrisLecarNeuron",
            "TermanWangOscillator",
        ),
    ),
    "Map-based": (
        "map-based",
        (
            "AiharaMapNeuron",
            "CazellesMapNeuron",
            "ChialvoMapNeuron",
            "CourageNekorkinMapNeuron",
            "ErmentroutKopellMapNeuron",
            "IbarzTanakaMapNeuron",
            "MedvedevMapNeuron",
            "NagumoSatoMapNeuron",
            "RulkovMapNeuron",
            "SCAdaptiveThresholdMapNeuron",
            "SCChaoticMapNeuron",
        ),
    ),
    "Rate / mean-field": (
        "rate-mean-field",
        (
            "AmariNeuralField",
            "BrunelWangNeuron",
            "CompteWMNeuron",
            "ErmentroutKopellPopulation",
            "EscapeRateNeuron",
            "GIFPopulationNeuron",
            "JansenRitUnit",
            "LarterBreakspearNeuron",
            "SCDecoupledAdaptationIonMassNeuron",
            "LeakyCompeteFireNeuron",
            "SiegertTransferFunction",
            "SigmoidRateNeuron",
            "ThresholdLinearRateNeuron",
            "WendlingNeuron",
            "WilsonCowanUnit",
            "WongWangUnit",
        ),
    ),
    "Statistical": (
        "statistical",
        (
            "GLMNeuron",
            "GalvesLocherbachNeuron",
            "GammaRenewalNeuron",
            "InhomogeneousPoissonNeuron",
            "McCullochPittsNeuron",
            "PoissonNeuron",
            "SRM0Neuron",
            "SpikeResponseNeuron",
        ),
    ),
    "Hardware / neuromorphic": (
        "hardware-neuromorphic",
        (
            "AkidaNeuron",
            "BrainScaleSAdExNeuron",
            "COBALIFNeuron",
            "DPINeuron",
            "EPropALIFNeuron",
            "Loihi2Neuron",
            "LoihiCUBANeuron",
            "NeuroGridNeuron",
            "ParallelSpikingNeuron",
            "SCResettingParallelSpikingNeuron",
            "SigmaDeltaNeuron",
            "SpiNNaker2Neuron",
            "SpiNNakerLIFNeuron",
            "TrueNorthNeuron",
        ),
    ),
    "AI-optimized": (
        "ai-optimized",
        (
            "ArcaneNeuron",
            "AttentionGatedNeuron",
            "ClosedFormContinuousNeuron",
            "CompositionalBindingNeuron",
            "ContinuousAttractorNeuron",
            "DifferentiableSurrogateNeuron",
            "HybridFisherPosnerLIFNeuron",
            "HybridLinearAttentionNeuron",
            "LearnableNeuronModel",
            "LiquidTimeConstantNeuron",
            "MetaPlasticNeuron",
            "MultiTimescaleNeuron",
            "PredictiveCodingNeuron",
            "QuantumInspiredLIFNeuron",
            "SelfReferentialNeuron",
            "SuperSpikeNeuron",
        ),
    ),
    "Glia / astrocyte": (
        "glia-astrocyte",
        (
            "AstrocyteLIFNeuron",
            "AstrocyteModel",
            "AstrocyteNeuron",
        ),
    ),
    "Synaptic dynamics": (
        "synaptic-dynamics",
        (
            "AlphaNeuron",
            "TsodyksMarkramNeuron",
        ),
    ),
    "Sensory": (
        "sensory",
        (
            "CochlearHairCell",
            "DirectionSelectiveRGC",
        ),
    ),
    "Neuromodulatory": (
        "neuromodulatory",
        ("DurstewitzDopamineNeuron",),
    ),
}

_COMPATIBILITY_ALIASES = {
    "KilincBhattMapNeuron": "SCAdaptiveThresholdMapNeuron",
}

_CLASS_TO_FAMILY: dict[str, tuple[str, str]] = {
    class_name: (family, category)
    for family, (category, members) in _FAMILIES.items()
    for class_name in members
}


def canonical_model_name(class_name: str) -> str:
    """Return the catalogue identity for ``class_name``, resolving aliases."""

    return _COMPATIBILITY_ALIASES.get(class_name, class_name)


def model_family(class_name: str) -> tuple[str, str] | None:
    """Return ``(family, category_slug)`` for a model, or ``None`` if unclassified."""

    return _CLASS_TO_FAMILY.get(canonical_model_name(class_name))


def families() -> dict[str, str]:
    """Return the family display name to category slug mapping."""

    return {family: category for family, (category, _members) in _FAMILIES.items()}


def classified_models() -> frozenset[str]:
    """Return canonical catalogue model names.

    Historical aliases remain accepted by :func:`model_family`, but they are
    not independent catalogue identities and therefore must not inflate the
    model inventory.
    """

    return frozenset(_CLASS_TO_FAMILY)


__all__ = [
    "canonical_model_name",
    "classified_models",
    "families",
    "model_family",
]
