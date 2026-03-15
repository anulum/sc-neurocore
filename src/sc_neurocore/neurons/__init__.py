# SPDX-License-Identifier: AGPL-3.0-or-later
"""SC-NeuroCore neuron library — 60+ models, one file per model.

Core models (original SC bitstream neurons):
    from sc_neurocore.neurons import StochasticLIFNeuron, FixedPointLIFNeuron

All models (including biophysical, bursting, hardware, etc.):
    from sc_neurocore.neurons import HodgkinHuxleyNeuron, AdExNeuron, ...

Individual imports:
    from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron
"""

# Core SC neurons (original, bitstream-capable)
from .base import BaseNeuron
from .stochastic_lif import StochasticLIFNeuron
from .fixed_point_lif import FixedPointLIFNeuron, FixedPointLFSR, FixedPointBitstreamEncoder
from .homeostatic_lif import HomeostaticLIFNeuron
from .dendritic import StochasticDendriticNeuron
from .sc_izhikevich import SCIzhikevichNeuron

# All extended models (one file per model in models/)
from .models import (
    AdExNeuron,
    AlphaNeuron,
    BrainScaleSAdExNeuron,
    ButeraRespiratoryNeuron,
    ChayNeuron,
    ChialvoMapNeuron,
    ConnorStevensNeuron,
    CourageNekorkinMapNeuron,
    DestexheThalamicNeuron,
    EnergyLIFNeuron,
    ErmentroutKopellPopulation,
    EscapeRateNeuron,
    ExpIFNeuron,
    FitzHughNagumoNeuron,
    FitzHughRinzelNeuron,
    FractionalLIFNeuron,
    GalvesLocherbachNeuron,
    GatedLIFNeuron,
    GLIFNeuron,
    GutkinErmentroutNeuron,
    HindmarshRoseNeuron,
    HodgkinHuxleyNeuron,
    HuberBraunNeuron,
    InhomogeneousPoissonNeuron,
    JansenRitUnit,
    LapicqueNeuron,
    LeakyCompeteFireNeuron,
    LoihiCUBANeuron,
    MATNeuron,
    MedvedevMapNeuron,
    MihalasNieburNeuron,
    MorrisLecarNeuron,
    PinskyRinzelNeuron,
    PoissonNeuron,
    PrescottNeuron,
    QuadraticIFNeuron,
    ResonateAndFireNeuron,
    RulkovMapNeuron,
    SFANeuron,
    ShermanRinzelKeizerNeuron,
    SigmaDeltaNeuron,
    SpikeResponseNeuron,
    SpiNNakerLIFNeuron,
    StochasticIFNeuron,
    ThetaNeuron,
    TrueNorthNeuron,
    WangBuzsakiNeuron,
    WilsonCowanUnit,
    WongWangUnit,
)

__all__ = [
    # Core SC neurons
    "BaseNeuron",
    "StochasticLIFNeuron",
    "FixedPointLIFNeuron",
    "FixedPointLFSR",
    "FixedPointBitstreamEncoder",
    "HomeostaticLIFNeuron",
    "StochasticDendriticNeuron",
    "SCIzhikevichNeuron",
    # Extended models (one file per model)
    "AdExNeuron",
    "AlphaNeuron",
    "BrainScaleSAdExNeuron",
    "ButeraRespiratoryNeuron",
    "ChayNeuron",
    "ChialvoMapNeuron",
    "ConnorStevensNeuron",
    "CourageNekorkinMapNeuron",
    "DestexheThalamicNeuron",
    "EnergyLIFNeuron",
    "ErmentroutKopellPopulation",
    "EscapeRateNeuron",
    "ExpIFNeuron",
    "FitzHughNagumoNeuron",
    "FitzHughRinzelNeuron",
    "FractionalLIFNeuron",
    "GalvesLocherbachNeuron",
    "GatedLIFNeuron",
    "GLIFNeuron",
    "GutkinErmentroutNeuron",
    "HindmarshRoseNeuron",
    "HodgkinHuxleyNeuron",
    "HuberBraunNeuron",
    "InhomogeneousPoissonNeuron",
    "JansenRitUnit",
    "LapicqueNeuron",
    "LeakyCompeteFireNeuron",
    "LoihiCUBANeuron",
    "MATNeuron",
    "MedvedevMapNeuron",
    "MihalasNieburNeuron",
    "MorrisLecarNeuron",
    "PinskyRinzelNeuron",
    "PoissonNeuron",
    "PrescottNeuron",
    "QuadraticIFNeuron",
    "ResonateAndFireNeuron",
    "RulkovMapNeuron",
    "SFANeuron",
    "ShermanRinzelKeizerNeuron",
    "SigmaDeltaNeuron",
    "SpikeResponseNeuron",
    "SpiNNakerLIFNeuron",
    "StochasticIFNeuron",
    "ThetaNeuron",
    "TrueNorthNeuron",
    "WangBuzsakiNeuron",
    "WilsonCowanUnit",
    "WongWangUnit",
]
