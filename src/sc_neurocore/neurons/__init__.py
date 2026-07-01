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
When the Rust engine (``sc_neurocore_engine``) is available,
``from sc_neurocore.neurons import X`` transparently returns the Rust
implementation.  ``from sc_neurocore.neurons.models import X`` always
returns the pure-Python version.

Set the environment variable ``SC_NEUROCORE_NO_RUST=1`` to force the
pure-Python path even when the Rust engine is present.

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
    "StochasticDendriticNeuron",
    "SCIzhikevichNeuron",
    "UniversalNeuron",
    "list_bundled_schemas",
}

_MODEL_NAMES = {
    "ArcaneNeuron",
    "AdExNeuron",
    "AdaptiveThresholdIFNeuron",
    "AttentionGatedNeuron",
    "AkidaNeuron",
    "AlphaNeuron",
    "AmariNeuralField",
    "AstrocyteModel",
    "AvRonCardiacNeuron",
    "BalancedResonateAndFireNeuron",
    "BendaHerzNeuron",
    "BertramPhantomBurster",
    "BoothRinzelNeuron",
    "BrainScaleSAdExNeuron",
    "ButeraRespiratoryNeuron",
    "COBALIFNeuron",
    "CazellesMapNeuron",
    "ChayKeizerNeuron",
    "ChayNeuron",
    "ChialvoMapNeuron",
    "ClosedFormContinuousNeuron",
    "ComplementaryLIFNeuron",
    "CompositionalBindingNeuron",
    "CompteWMNeuron",
    "ContinuousAttractorNeuron",
    "ConnorStevensNeuron",
    "CourageNekorkinMapNeuron",
    "DPINeuron",
    "DeSchutterPurkinjeNeuron",
    "DendrifyNeuron",
    "DestexheThalamicNeuron",
    "DifferentiableSurrogateNeuron",
    "DurstewitzDopamineNeuron",
    "EPropALIFNeuron",
    "EnergyLIFNeuron",
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
    "GammaRenewalNeuron",
    "GatedLIFNeuron",
    "GolombFSNeuron",
    "GutkinErmentroutNeuron",
    "HayL5PyramidalNeuron",
    "HillTononiNeuron",
    "HindmarshRoseNeuron",
    "HodgkinHuxleyNeuron",
    "HuberBraunNeuron",
    "IbarzTanakaMapNeuron",
    "Izhikevich2007Neuron",
    "InhibitoryLIFNeuron",
    "InhomogeneousPoissonNeuron",
    "IntegerQIFNeuron",
    "JansenRitUnit",
    "KLIFNeuron",
    "LapicqueNeuron",
    "LarterBreakspearNeuron",
    "LeakyCompeteFireNeuron",
    "LearnableNeuronModel",
    "LiquidTimeConstantNeuron",
    "Loihi2Neuron",
    "LoihiCUBANeuron",
    "MATNeuron",
    "MainenSejnowskiNeuron",
    "MarderSTGNeuron",
    "McCullochPittsNeuron",
    "McKeanNeuron",
    "MedvedevMapNeuron",
    "MetaPlasticNeuron",
    "MihalasNieburNeuron",
    "MorrisLecarNeuron",
    "MultiTimescaleNeuron",
    "NeuroGridNeuron",
    "NonResettingLIFNeuron",
    "NonlinearLIFNeuron",
    "ParallelSpikingNeuron",
    "ParametricLIFNeuron",
    "PerfectIntegratorNeuron",
    "PernarowskiNeuron",
    "PinskyRinzelNeuron",
    "PlantR15Neuron",
    "PoissonNeuron",
    "PospischilNeuron",
    "PredictiveCodingNeuron",
    "PrescottNeuron",
    "QuadraticIFNeuron",
    "RallCableNeuron",
    "ResonateAndFireNeuron",
    "RulkovMapNeuron",
    "SFANeuron",
    "SelfReferentialNeuron",
    "ShermanRinzelKeizerNeuron",
    "SiegertTransferFunction",
    "SigmaDeltaNeuron",
    "SigmoidRateNeuron",
    "SpiNNaker2Neuron",
    "SpiNNakerLIFNeuron",
    "SpikeResponseNeuron",
    "StochasticIFNeuron",
    "SuperSpikeNeuron",
    "TermanWangOscillator",
    "ThetaNeuron",
    "ThresholdLinearRateNeuron",
    "TraubMilesNeuron",
    "TrueNorthNeuron",
    "TsodyksMarkramNeuron",
    "TwoCompartmentLIFNeuron",
    "WangBuzsakiNeuron",
    "WendlingNeuron",
    "WilsonCowanUnit",
    "WilsonHRNeuron",
    "WongWangUnit",
    "YamadaNeuron",
}

__all__ = sorted(_CORE_NAMES | _MODEL_NAMES)

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

            _rust_map = {name: getattr(_eng, name) for name in __all__ if hasattr(_eng, name)}
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
