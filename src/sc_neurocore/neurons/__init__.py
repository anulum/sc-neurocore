# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — neuron library - every published model, one file each

"""SC-NeuroCore neuron library - every published model, one file each.

Auto-dispatch
-------------
Every class in ``sc_neurocore.neurons.models.__all__`` is importable here.
When the Rust engine (``sc_neurocore_engine``) is available,
``from sc_neurocore.neurons import X`` returns the Rust class only for the
names in ``_RUST_DISPATCH_NAMES``: those whose Rust class accepts every
Python constructor field and reproduces the Python trajectory, as
``tests/test_neuron_facade_completeness.py`` checks. Every other name
returns the Python class. ``from sc_neurocore.neurons.models import X``
always returns the Python class, and the Rust classes stay importable from
``sc_neurocore_engine`` directly.

Set the environment variable ``SC_NEUROCORE_NO_RUST=1`` to force the
pure-Python path even for the listed names.

Lazy loading
------------
The 109+ model classes from ``neurons.models`` are loaded on first access
to keep ``import sc_neurocore`` fast (<5s). Core neurons (StochasticLIFNeuron,
FixedPointLIFNeuron, etc.) are always available immediately.
"""

from .base import BaseNeuron as BaseNeuron
from .dendritic import StochasticDendriticNeuron as StochasticDendriticNeuron
from .fixed_point_lif import FixedPointBitstreamEncoder as FixedPointBitstreamEncoder
from .fixed_point_lif import FixedPointLFSR as FixedPointLFSR
from .fixed_point_lif import FixedPointLIFNeuron as FixedPointLIFNeuron
from .homeostatic_lif import HomeostaticLIFNeuron as HomeostaticLIFNeuron
from .reference_traces import ReferenceTraceSpec as ReferenceTraceSpec
from .reference_traces import TraceValidationReport as TraceValidationReport
from .reference_traces import validate_all_reference_traces as validate_all_reference_traces
from .reference_traces import validate_reference_trace as validate_reference_trace
from .sc_izhikevich import SCIzhikevichNeuron as SCIzhikevichNeuron
from .stochastic_lif import StochasticLIFNeuron as StochasticLIFNeuron
from .universal_dsl import UniversalNeuron as UniversalNeuron
from .universal_dsl import list_bundled_schemas as list_bundled_schemas

_CORE_NAMES = {
    "BaseNeuron",
    "StochasticLIFNeuron",
    "FixedPointLIFNeuron",
    "FixedPointLFSR",
    "FixedPointBitstreamEncoder",
    "HomeostaticLIFNeuron",
    "ReferenceTraceSpec",
    "StochasticDendriticNeuron",
    "SCIzhikevichNeuron",
    "UniversalNeuron",
    "TraceValidationReport",
    "list_bundled_schemas",
    "validate_all_reference_traces",
    "validate_reference_trace",
}

_MODEL_NAMES = {
    "ATypeKNeuron",
    "AdExNeuron",
    "AdaptiveThresholdIFNeuron",
    "AdaptiveThresholdMoENeuron",
    "AiharaMapNeuron",
    "AkidaNeuron",
    "AlphaMotorNeuron",
    "AlphaNeuron",
    "AmariNeuralField",
    "ArcaneNeuron",
    "AstrocyteLIFNeuron",
    "AstrocyteModel",
    "AstrocyteNeuron",
    "AttentionGatedNeuron",
    "AvRonCardiacNeuron",
    "BKNeuron",
    "BalancedResonateAndFireNeuron",
    "BendaHerzNeuron",
    "BertramPhantomBurster",
    "BoothRinzelNeuron",
    "BrainScaleSAdExNeuron",
    "BrunelWangNeuron",
    "ButeraRespiratoryNeuron",
    "COBALIFNeuron",
    "CazellesMapNeuron",
    "CerebellarBasketNeuron",
    "ChandelierNeuron",
    "ChayKeizerMinimalNeuron",
    "ChayKeizerNeuron",
    "ChayNeuron",
    "ChialvoMapNeuron",
    "ClosedFormContinuousNeuron",
    "CochlearHairCell",
    "ComplementaryLIFNeuron",
    "CompositionalBindingNeuron",
    "CompteWMNeuron",
    "ConnorStevensNeuron",
    "ContinuousAttractorNeuron",
    "CourageNekorkinMapNeuron",
    "DCNNeuron",
    "DPINeuron",
    "DeSchutterPurkinjeNeuron",
    "DendrifyNeuron",
    "DendriticNMDANeuron",
    "DestexheThalamicNeuron",
    "DifferentiableSurrogateNeuron",
    "DirectionSelectiveRGC",
    "DurstewitzDopamineNeuron",
    "EPropALIFNeuron",
    "EnergyLIFNeuron",
    "ErmentroutKopellMapNeuron",
    "ErmentroutKopellPopulation",
    "EscapeRateNeuron",
    "ExpIFNeuron",
    "FitzHughNagumoNeuron",
    "FitzHughRinzelNeuron",
    "FractionalLIFNeuron",
    "GIFPopulationNeuron",
    "GLIFNeuron",
    "GLMNeuron",
    "GalvesLocherbachNeuron",
    "GammaMotorNeuron",
    "GammaRenewalNeuron",
    "GatedLIFNeuron",
    "GolgiCell",
    "GolombFSNeuron",
    "GranuleCell",
    "GutkinErmentroutNeuron",
    "HayL5PyramidalNeuron",
    "HillTononiNeuron",
    "HindmarshRoseNeuron",
    "HodgkinHuxleyNeuron",
    "HuberBraunNeuron",
    "HybridFisherPosnerLIFNeuron",
    "HybridLinearAttentionNeuron",
    "IbarzTanakaMapNeuron",
    "IhNeuron",
    "InhibitoryLIFNeuron",
    "InhomogeneousPoissonNeuron",
    "IntegerQIFNeuron",
    "Izhikevich2007Neuron",
    "JansenRitUnit",
    "KLIFNeuron",
    "LapicqueNeuron",
    "LarterBreakspearNeuron",
    "LeakyCompeteFireNeuron",
    "LearnableNeuronModel",
    "LiquidTimeConstantNeuron",
    "Loihi2Neuron",
    "LoihiCUBANeuron",
    "LugaroCell",
    "MATNeuron",
    "MainenSejnowskiNeuron",
    "MarderSTGNeuron",
    "MartinottiNeuron",
    "McCullochPittsNeuron",
    "McKeanNeuron",
    "MedvedevMapNeuron",
    "MetaPlasticNeuron",
    "MihalasNieburNeuron",
    "MorrisLecarNeuron",
    "MotorUnit",
    "MultiTimescaleNeuron",
    "MulticompartmentMCNNeuron",
    "NMDANeuron",
    "NagumoSatoMapNeuron",
    "NeuroGridNeuron",
    "NonResettingLIFNeuron",
    "NonlinearLIFNeuron",
    "PVFastSpikingNeuron",
    "ParallelSpikingNeuron",
    "ParametricLIFNeuron",
    "PerfectIntegratorNeuron",
    "PernarowskiNeuron",
    "PersistentNaNeuron",
    "PinskyRinzelNeuron",
    "PlantR15Neuron",
    "PoissonNeuron",
    "PospischilNeuron",
    "PredictiveCodingNeuron",
    "PrescottNeuron",
    "QuadraticIFNeuron",
    "QuantumInspiredLIFNeuron",
    "RallCableNeuron",
    "RenshawCell",
    "ResonateAndFireNeuron",
    "RulkovMapNeuron",
    "SCAdaptiveThresholdMapNeuron",
    "SCChaoticMapNeuron",
    "SCClippedLogisticBurstingMapNeuron",
    "SCClippedRationalRecoveryMapNeuron",
    "SCDecoupledAdaptationIonMassNeuron",
    "SCExponentialTwoCompartmentLIFNeuron",
    "SCFourStateGLIFNeuron",
    "SCInclusivePerfectIntegratorNeuron",
    "SCLapicqueLIFNeuron",
    "SCLeakyTwoCompartmentLIFNeuron",
    "SCNonResettingAdaptiveLIFNeuron",
    "SCNormalizedEnergyLIFNeuron",
    "SCResettingMATNeuron",
    "SCResettingParallelSpikingNeuron",
    "SCResettingWilsonHRNeuron",
    "SCScaledResetAdaptiveIFNeuron",
    "SCSigmaDeltaAccumulatorNeuron",
    "SCSixStateThalamocorticalNeuron",
    "SCStochasticRateAdaptationNeuron",
    "SCSymmetricQuadraticIFNeuron",
    "SCThreeStatePhantomBurster",
    "SCTriangularMcKeanNeuron",
    "SCUnitCapacitanceRespiratoryNeuron",
    "SCUpwardCrossingRulkovMapNeuron",
    "SCWBNMDAMagnesiumBlockNeuron",
    "SFANeuron",
    "SKNeuron",
    "SRM0Neuron",
    "SSTNeuron",
    "SelfReferentialNeuron",
    "ShermanRinzelKeizerNeuron",
    "SiegertTransferFunction",
    "SigmaDeltaNeuron",
    "SigmoidRateNeuron",
    "SpiNNaker2Neuron",
    "SpiNNakerLIFNeuron",
    "SpikeResponseNeuron",
    "StellateCell",
    "StochasticIFNeuron",
    "SuperSpikeNeuron",
    "TTypeCaNeuron",
    "TermanWangOscillator",
    "ThetaNeuron",
    "ThresholdLinearRateNeuron",
    "TraubMilesNeuron",
    "TrueNorthNeuron",
    "TsodyksMarkramNeuron",
    "TwoCompartmentLIFNeuron",
    "UnipolarBrushCell",
    "UpperMotorNeuron",
    "VIPNeuron",
    "WangBuzsakiNeuron",
    "WendlingNeuron",
    "WilsonCowanUnit",
    "WilsonHRNeuron",
    "WongWangUnit",
    "YamadaNeuron",
}

__all__ = sorted(_CORE_NAMES | _MODEL_NAMES)

# Names whose Rust class accepts every Python constructor field and reproduces the Python
# trajectory (tests/test_neuron_facade_completeness.py). Only these auto-dispatch to Rust; a class
# joins in the same unit as its parity proof.
_RUST_DISPATCH_NAMES = frozenset(
    {
        "AmariNeuralField",
        "COBALIFNeuron",
        "CompteWMNeuron",
        "EnergyLIFNeuron",
        "IntegerQIFNeuron",
        "JansenRitUnit",
        "MATNeuron",
        "McCullochPittsNeuron",
        "McKeanNeuron",
        "NonResettingLIFNeuron",
        "SCNonResettingAdaptiveLIFNeuron",
        "SCNormalizedEnergyLIFNeuron",
        "SCResettingMATNeuron",
        "SCSigmaDeltaAccumulatorNeuron",
        "SigmaDeltaNeuron",
        "SigmoidRateNeuron",
        "ThresholdLinearRateNeuron",
    }
)

# -- Rust auto-dispatch cache --
_rust_map: dict[str, type] | None = None


def _load_rust_map() -> None:
    """Lazily populate the Rust-backed neuron model dispatch map."""
    global _rust_map
    if _rust_map is not None:
        return
    import os

    _rust_map = {}
    if not os.environ.get("SC_NEUROCORE_NO_RUST"):
        try:  # pragma: no cover
            from sc_neurocore_engine import sc_neurocore_engine as _eng

            _rust_map = {
                name: getattr(_eng, name)
                for name in sorted(_RUST_DISPATCH_NAMES)
                if hasattr(_eng, name)
            }
        except ImportError:
            pass


def __getattr__(name: str) -> type:
    """Lazy-load neuron models and apply Rust auto-dispatch on first access."""
    if name in _MODEL_NAMES:
        _load_rust_map()
        rust_map = _rust_map if _rust_map is not None else {}
        if name in rust_map:  # pragma: no cover
            obj = rust_map[name]
        else:
            from . import models

            obj = getattr(models, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
