# SPDX-License-Identifier: AGPL-3.0-or-later

"""Individual neuron model files — one class per file."""

from .adex import AdExNeuron
from .alpha import AlphaNeuron
from .brainscales_adex import BrainScaleSAdExNeuron
from .butera_respiratory import ButeraRespiratoryNeuron
from .chay import ChayNeuron
from .chialvo_map import ChialvoMapNeuron
from .connor_stevens import ConnorStevensNeuron
from .courage_nekorkin_map import CourageNekorkinMapNeuron
from .destexhe_thalamic import DestexheThalamicNeuron
from .energy_lif import EnergyLIFNeuron
from .ermentrout_kopell_pop import ErmentroutKopellPopulation
from .escape_rate import EscapeRateNeuron
from .expif import ExpIFNeuron
from .fitzhugh_nagumo import FitzHughNagumoNeuron
from .fitzhugh_rinzel import FitzHughRinzelNeuron
from .fractional_lif import FractionalLIFNeuron
from .galves_locherbach import GalvesLocherbachNeuron
from .gated_lif import GatedLIFNeuron
from .glif import GLIFNeuron
from .gutkin_ermentrout import GutkinErmentroutNeuron
from .hindmarsh_rose import HindmarshRoseNeuron
from .hodgkin_huxley import HodgkinHuxleyNeuron
from .huber_braun import HuberBraunNeuron
from .inhomogeneous_poisson import InhomogeneousPoissonNeuron
from .jansen_rit import JansenRitUnit
from .lapicque import LapicqueNeuron
from .leaky_compete_fire import LeakyCompeteFireNeuron
from .loihi_cuba import LoihiCUBANeuron
from .mat import MATNeuron
from .medvedev_map import MedvedevMapNeuron
from .mihalas_niebur import MihalasNieburNeuron
from .morris_lecar import MorrisLecarNeuron
from .pinsky_rinzel import PinskyRinzelNeuron
from .poisson import PoissonNeuron
from .prescott import PrescottNeuron
from .quadratic_if import QuadraticIFNeuron
from .resonate_and_fire import ResonateAndFireNeuron
from .rulkov_map import RulkovMapNeuron
from .sfa import SFANeuron
from .sherman_rinzel_keizer import ShermanRinzelKeizerNeuron
from .sigma_delta import SigmaDeltaNeuron
from .spike_response import SpikeResponseNeuron
from .spinnaker_lif import SpiNNakerLIFNeuron
from .stochastic_if import StochasticIFNeuron
from .theta import ThetaNeuron
from .truenorth import TrueNorthNeuron
from .wang_buzsaki import WangBuzsakiNeuron
from .wilson_cowan import WilsonCowanUnit
from .wong_wang import WongWangUnit

__all__ = [
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
    "GLIFNeuron",
    "GalvesLocherbachNeuron",
    "GatedLIFNeuron",
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
    "SpiNNakerLIFNeuron",
    "SpikeResponseNeuron",
    "StochasticIFNeuron",
    "ThetaNeuron",
    "TrueNorthNeuron",
    "WangBuzsakiNeuron",
    "WilsonCowanUnit",
    "WongWangUnit",
]
