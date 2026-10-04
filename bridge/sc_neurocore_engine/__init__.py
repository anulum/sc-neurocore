# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Engine package re-exports, SIMD primitives, and IR

"""SC-NeuroCore Engine package re-exports, SIMD primitives, and IR compiler."""

from pkgutil import extend_path as _extend_path


# Pytest loads the checkout bridge before the extension installed by maturin develop.
__path__ = _extend_path(__path__, __name__)

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        __version__ as __version__,
        simd_tier as simd_tier,
        set_num_threads as set_num_threads,
        pack_bitstream as pack_bitstream,
        unpack_bitstream as unpack_bitstream,
        popcount as popcount,
        pack_bitstream_numpy as pack_bitstream_numpy,
        popcount_numpy as popcount_numpy,
        unpack_bitstream_numpy as unpack_bitstream_numpy,
        batch_lif_run as batch_lif_run,
        batch_lif_run_multi as batch_lif_run_multi,
        batch_lif_run_varying as batch_lif_run_varying,
        batch_encode as batch_encode,
        batch_encode_numpy as batch_encode_numpy,
        Lfsr16 as Lfsr16,
        BitstreamEncoder as BitstreamEncoder,
        FixedPointLif as FixedPointLif,
        DenseLayer as DenseLayer,
        StdpSynapse as StdpSynapse,
        SCPNMetrics as SCPNMetrics,
        BitStreamTensor as BitStreamTensor,
        BrunelNetwork as BrunelNetwork,
        FixedPointBrunelNetwork as FixedPointBrunelNetwork,
        PySpikingControllerPool as PySpikingControllerPool,
    )
except ImportError:
    _core_available = False
else:
    _core_available = True

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        QuadraticIFNeuron as QuadraticIFNeuron,
        quadratic_if_simulate_complete as quadratic_if_simulate_complete,
        ThetaNeuron as ThetaNeuron,
        theta_simulate_complete as theta_simulate_complete,
        PerfectIntegratorNeuron as PerfectIntegratorNeuron,
        perfect_integrator_simulate_complete as perfect_integrator_simulate_complete,
        GatedLIFNeuron as GatedLIFNeuron,
        NonlinearLIFNeuron as NonlinearLIFNeuron,
        SFANeuron as SFANeuron,
        MATNeuron as MATNeuron,
        SCResettingMATNeuron as SCResettingMATNeuron,
        EscapeRateNeuron as EscapeRateNeuron,
        KLIFNeuron as KLIFNeuron,
        InhibitoryLIFNeuron as InhibitoryLIFNeuron,
        ComplementaryLIFNeuron as ComplementaryLIFNeuron,
        ParametricLIFNeuron as ParametricLIFNeuron,
        NonResettingLIFNeuron as NonResettingLIFNeuron,
        SCNonResettingAdaptiveLIFNeuron as SCNonResettingAdaptiveLIFNeuron,
        AdaptiveThresholdIFNeuron as AdaptiveThresholdIFNeuron,
        SigmaDeltaNeuron as SigmaDeltaNeuron,
        SCSigmaDeltaAccumulatorNeuron as SCSigmaDeltaAccumulatorNeuron,
        EnergyLIFNeuron as EnergyLIFNeuron,
        SCNormalizedEnergyLIFNeuron as SCNormalizedEnergyLIFNeuron,
        IntegerQIFNeuron as IntegerQIFNeuron,
        ClosedFormContinuousNeuron as ClosedFormContinuousNeuron,
        FitzHughNagumoNeuron as FitzHughNagumoNeuron,
        MorrisLecarNeuron as MorrisLecarNeuron,
        HindmarshRoseNeuron as HindmarshRoseNeuron,
        ResonateAndFireNeuron as ResonateAndFireNeuron,
        BalancedResonateAndFireNeuron as BalancedResonateAndFireNeuron,
        FitzHughRinzelNeuron as FitzHughRinzelNeuron,
        McKeanNeuron as McKeanNeuron,
        SCTriangularMcKeanNeuron as SCTriangularMcKeanNeuron,
        TermanWangOscillator as TermanWangOscillator,
        BendaHerzNeuron as BendaHerzNeuron,
        SCStochasticRateAdaptationNeuron as SCStochasticRateAdaptationNeuron,
        AlphaNeuron as AlphaNeuron,
        COBALIFNeuron as COBALIFNeuron,
        GutkinErmentroutNeuron as GutkinErmentroutNeuron,
        WilsonHRNeuron as WilsonHRNeuron,
        SCResettingWilsonHRNeuron as SCResettingWilsonHRNeuron,
        ChayNeuron as ChayNeuron,
        ChayKeizerNeuron as ChayKeizerNeuron,
        ShermanRinzelKeizerNeuron as ShermanRinzelKeizerNeuron,
        ButeraRespiratoryNeuron as ButeraRespiratoryNeuron,
        EPropALIFNeuron as EPropALIFNeuron,
        SuperSpikeNeuron as SuperSpikeNeuron,
        LearnableNeuronModel as LearnableNeuronModel,
        PernarowskiNeuron as PernarowskiNeuron,
        ChialvoMapNeuron as ChialvoMapNeuron,
        RulkovMapNeuron as RulkovMapNeuron,
        SCUpwardCrossingRulkovMapNeuron as SCUpwardCrossingRulkovMapNeuron,
        IbarzTanakaMapNeuron as IbarzTanakaMapNeuron,
        MedvedevMapNeuron as MedvedevMapNeuron,
        CazellesMapNeuron as CazellesMapNeuron,
        SCClippedLogisticBurstingMapNeuron as SCClippedLogisticBurstingMapNeuron,
        SCClippedRationalRecoveryMapNeuron as SCClippedRationalRecoveryMapNeuron,
        CourageNekorkinMapNeuron as CourageNekorkinMapNeuron,
        AiharaMapNeuron as AiharaMapNeuron,
        NagumoSatoMapNeuron as NagumoSatoMapNeuron,
        SCAdaptiveThresholdMapNeuron as SCAdaptiveThresholdMapNeuron,
        SCChaoticMapNeuron as SCChaoticMapNeuron,
        KilincBhattMapNeuron as KilincBhattMapNeuron,
        HodgkinHuxleyNeuron as HodgkinHuxleyNeuron,
        TraubMilesNeuron as TraubMilesNeuron,
        WangBuzsakiNeuron as WangBuzsakiNeuron,
        NMDANeuron as NMDANeuron,
        SCWBNMDAMagnesiumBlockNeuron as SCWBNMDAMagnesiumBlockNeuron,
        ConnorStevensNeuron as ConnorStevensNeuron,
        DestexheThalamicNeuron as DestexheThalamicNeuron,
        HuberBraunNeuron as HuberBraunNeuron,
        GolombFSNeuron as GolombFSNeuron,
        PospischilNeuron as PospischilNeuron,
        MainenSejnowskiNeuron as MainenSejnowskiNeuron,
        DeSchutterPurkinjeNeuron as DeSchutterPurkinjeNeuron,
        PlantR15Neuron as PlantR15Neuron,
        PrescottNeuron as PrescottNeuron,
        MihalasNieburNeuron as MihalasNieburNeuron,
        SCScaledResetAdaptiveIFNeuron as SCScaledResetAdaptiveIFNeuron,
        GLIFNeuron as GLIFNeuron,
        SCFourStateGLIFNeuron as SCFourStateGLIFNeuron,
        GIFPopulationNeuron as GIFPopulationNeuron,
        AvRonCardiacNeuron as AvRonCardiacNeuron,
        DurstewitzDopamineNeuron as DurstewitzDopamineNeuron,
        HillTononiNeuron as HillTononiNeuron,
        BertramPhantomBurster as BertramPhantomBurster,
        YamadaNeuron as YamadaNeuron,
        PinskyRinzelNeuron as PinskyRinzelNeuron,
        HayL5PyramidalNeuron as HayL5PyramidalNeuron,
        MarderSTGNeuron as MarderSTGNeuron,
        RallCableNeuron as RallCableNeuron,
        BoothRinzelNeuron as BoothRinzelNeuron,
        DendrifyNeuron as DendrifyNeuron,
        TwoCompartmentLIFNeuron as TwoCompartmentLIFNeuron,
        PoissonNeuron as PoissonNeuron,
        InhomogeneousPoissonNeuron as InhomogeneousPoissonNeuron,
        GammaRenewalNeuron as GammaRenewalNeuron,
        StochasticIFNeuron as StochasticIFNeuron,
        StochasticLIFNeuron as StochasticLIFNeuron,
        GalvesLocherbachNeuron as GalvesLocherbachNeuron,
        SpikeResponseNeuron as SpikeResponseNeuron,
        GLMNeuron as GLMNeuron,
        WilsonCowanUnit as WilsonCowanUnit,
        JansenRitUnit as JansenRitUnit,
        WongWangUnit as WongWangUnit,
        ErmentroutKopellPopulation as ErmentroutKopellPopulation,
        WendlingNeuron as WendlingNeuron,
        LarterBreakspearNeuron as LarterBreakspearNeuron,
        LoihiCUBANeuron as LoihiCUBANeuron,
        Loihi2Neuron as Loihi2Neuron,
        TrueNorthNeuron as TrueNorthNeuron,
        BrainScaleSAdExNeuron as BrainScaleSAdExNeuron,
        SpiNNakerLIFNeuron as SpiNNakerLIFNeuron,
        SpiNNaker2Neuron as SpiNNaker2Neuron,
        DPINeuron as DPINeuron,
        dpi_neuron_simulate_complete as dpi_neuron_simulate_complete,
        AkidaNeuron as AkidaNeuron,
        NeuroGridNeuron as NeuroGridNeuron,
        McCullochPittsNeuron as McCullochPittsNeuron,
        SigmoidRateNeuron as SigmoidRateNeuron,
        ThresholdLinearRateNeuron as ThresholdLinearRateNeuron,
        AstrocyteModel as AstrocyteModel,
        TsodyksMarkramNeuron as TsodyksMarkramNeuron,
        LiquidTimeConstantNeuron as LiquidTimeConstantNeuron,
        CompteWMNeuron as CompteWMNeuron,
        SiegertTransferFunction as SiegertTransferFunction,
        FractionalLIFNeuron as FractionalLIFNeuron,
        ParallelSpikingNeuron as ParallelSpikingNeuron,
        AmariNeuralField as AmariNeuralField,
        BrunelWangNeuron as BrunelWangNeuron,
        LeakyCompeteFireNeuron as LeakyCompeteFireNeuron,
        AdExNeuron as AdExNeuron,
        adex_simulate_complete as adex_simulate_complete,
        ExpIFNeuron as ExpIFNeuron,
        expif_simulate_complete as expif_simulate_complete,
        LapicqueNeuron as LapicqueNeuron,
        lapicque_simulate_complete as lapicque_simulate_complete,
    )

    _neurons_available = True
except ImportError:
    _neurons_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        ArcaneNeuron as ArcaneNeuron,
        AttentionGatedNeuron as AttentionGatedNeuron,
        CompositionalBindingNeuron as CompositionalBindingNeuron,
        DifferentiableSurrogateNeuron as DifferentiableSurrogateNeuron,
        MetaPlasticNeuron as MetaPlasticNeuron,
        MultiTimescaleNeuron as MultiTimescaleNeuron,
        PredictiveCodingNeuron as PredictiveCodingNeuron,
        SelfReferentialNeuron as SelfReferentialNeuron,
    )
    from sc_neurocore_engine.sc_neurocore_engine import (
        RustContinuousAttractorNeuron as _RustContinuousAttractorNeuron,
    )

    ContinuousAttractorNeuron = _RustContinuousAttractorNeuron
    from sc_neurocore_engine.sc_neurocore_engine import (
        Izhikevich as Izhikevich,
        BitstreamAverager as BitstreamAverager,
        NetworkRunner as NetworkRunner,
    )

    _ai_available = True
except ImportError:
    _ai_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_simulate_ei_network as py_simulate_ei_network,
        py_batch_simulate as py_batch_simulate,
        py_rk4_neuron_simulate as py_rk4_neuron_simulate,
    )

    _studio_rust_available = True
except (ImportError, ModuleNotFoundError):
    # Bridge can't find the .pyd submodule — try loading from site-packages directly
    try:
        import glob as _glob
        import importlib.util as _ilu
        import sysconfig as _sc

        _site = _sc.get_path("purelib")
        _pyds = _glob.glob(f"{_site}/sc_neurocore_engine/sc_neurocore_engine*.pyd")
        _pyds += _glob.glob(f"{_site}/sc_neurocore_engine/sc_neurocore_engine*.so")
        if _pyds:
            _spec = _ilu.spec_from_file_location("sc_neurocore_engine", _pyds[0])
            if _spec is not None and _spec.loader is not None:
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                py_simulate_ei_network = _mod.py_simulate_ei_network
                py_batch_simulate = _mod.py_batch_simulate
                py_rk4_neuron_simulate = _mod.py_rk4_neuron_simulate
                _studio_rust_available = True
            else:
                _studio_rust_available = False
        else:
            _studio_rust_available = False
    except Exception:
        _studio_rust_available = False

_ENGINE_AVAILABLE = _core_available and _neurons_available
if not _ENGINE_AVAILABLE:
    raise ImportError(
        "sc_neurocore_engine not found. Build with:\ncd engine && maturin develop --release"
    )

from .layers import VectorizedSCLayer
from .neurons import FixedPointLIFNeuron
from .grad import SurrogateLif, DifferentiableDenseLayer
from .attention import StochasticAttention
from .graphs import StochasticGraphLayer
from .scpn import KuramotoSolver
from .ir import ScGraph, ScGraphBuilder, parse_ir
from .hdc import HDCVector
from .petri_net import PetriNetEngine

_bridge_available = True

_NEURON_MODELS = [
    "QuadraticIFNeuron",
    "ThetaNeuron",
    "PerfectIntegratorNeuron",
    "GatedLIFNeuron",
    "NonlinearLIFNeuron",
    "SFANeuron",
    "MATNeuron",
    "SCResettingMATNeuron",
    "EscapeRateNeuron",
    "KLIFNeuron",
    "InhibitoryLIFNeuron",
    "ComplementaryLIFNeuron",
    "ParametricLIFNeuron",
    "NonResettingLIFNeuron",
    "SCNonResettingAdaptiveLIFNeuron",
    "AdaptiveThresholdIFNeuron",
    "SigmaDeltaNeuron",
    "SCSigmaDeltaAccumulatorNeuron",
    "EnergyLIFNeuron",
    "SCNormalizedEnergyLIFNeuron",
    "IntegerQIFNeuron",
    "ClosedFormContinuousNeuron",
    "FitzHughNagumoNeuron",
    "MorrisLecarNeuron",
    "HindmarshRoseNeuron",
    "ResonateAndFireNeuron",
    "FitzHughRinzelNeuron",
    "McKeanNeuron",
    "SCTriangularMcKeanNeuron",
    "TermanWangOscillator",
    "BendaHerzNeuron",
    "AlphaNeuron",
    "COBALIFNeuron",
    "GutkinErmentroutNeuron",
    "WilsonHRNeuron",
    "SCResettingWilsonHRNeuron",
    "ChayNeuron",
    "ChayKeizerNeuron",
    "ShermanRinzelKeizerNeuron",
    "ButeraRespiratoryNeuron",
    "EPropALIFNeuron",
    "SuperSpikeNeuron",
    "LearnableNeuronModel",
    "PernarowskiNeuron",
    "ChialvoMapNeuron",
    "RulkovMapNeuron",
    "SCUpwardCrossingRulkovMapNeuron",
    "IbarzTanakaMapNeuron",
    "MedvedevMapNeuron",
    "CazellesMapNeuron",
    "SCClippedLogisticBurstingMapNeuron",
    "SCClippedRationalRecoveryMapNeuron",
    "CourageNekorkinMapNeuron",
    "HodgkinHuxleyNeuron",
    "TraubMilesNeuron",
    "WangBuzsakiNeuron",
    "NMDANeuron",
    "SCWBNMDAMagnesiumBlockNeuron",
    "ConnorStevensNeuron",
    "DestexheThalamicNeuron",
    "HuberBraunNeuron",
    "GolombFSNeuron",
    "PospischilNeuron",
    "MainenSejnowskiNeuron",
    "DeSchutterPurkinjeNeuron",
    "PlantR15Neuron",
    "PrescottNeuron",
    "MihalasNieburNeuron",
    "SCScaledResetAdaptiveIFNeuron",
    "GLIFNeuron",
    "SCFourStateGLIFNeuron",
    "GIFPopulationNeuron",
    "AvRonCardiacNeuron",
    "DurstewitzDopamineNeuron",
    "HillTononiNeuron",
    "BertramPhantomBurster",
    "YamadaNeuron",
    "PinskyRinzelNeuron",
    "HayL5PyramidalNeuron",
    "MarderSTGNeuron",
    "RallCableNeuron",
    "BoothRinzelNeuron",
    "DendrifyNeuron",
    "TwoCompartmentLIFNeuron",
    "PoissonNeuron",
    "InhomogeneousPoissonNeuron",
    "GammaRenewalNeuron",
    "StochasticIFNeuron",
    "StochasticLIFNeuron",
    "GalvesLocherbachNeuron",
    "SpikeResponseNeuron",
    "GLMNeuron",
    "WilsonCowanUnit",
    "JansenRitUnit",
    "WongWangUnit",
    "ErmentroutKopellPopulation",
    "WendlingNeuron",
    "LarterBreakspearNeuron",
    "LoihiCUBANeuron",
    "Loihi2Neuron",
    "TrueNorthNeuron",
    "BrainScaleSAdExNeuron",
    "SpiNNakerLIFNeuron",
    "SpiNNaker2Neuron",
    "DPINeuron",
    "AkidaNeuron",
    "NeuroGridNeuron",
    "McCullochPittsNeuron",
    "SigmoidRateNeuron",
    "ThresholdLinearRateNeuron",
    "AstrocyteModel",
    "TsodyksMarkramNeuron",
    "LiquidTimeConstantNeuron",
    "CompteWMNeuron",
    "SiegertTransferFunction",
    "FractionalLIFNeuron",
    "ParallelSpikingNeuron",
    "AmariNeuralField",
    "BrunelWangNeuron",
    "LeakyCompeteFireNeuron",
    "AdExNeuron",
    "ExpIFNeuron",
    "LapicqueNeuron",
]

_AI_MODELS = [
    "ArcaneNeuron",
    "AttentionGatedNeuron",
    "CompositionalBindingNeuron",
    "ContinuousAttractorNeuron",
    "DifferentiableSurrogateNeuron",
    "MetaPlasticNeuron",
    "MultiTimescaleNeuron",
    "PredictiveCodingNeuron",
    "SelfReferentialNeuron",
]

__all__ = [
    "__version__",
    "simd_tier",
    "set_num_threads",
    "pack_bitstream",
    "unpack_bitstream",
    "popcount",
    "pack_bitstream_numpy",
    "popcount_numpy",
    "unpack_bitstream_numpy",
    "batch_lif_run",
    "batch_lif_run_multi",
    "batch_lif_run_varying",
    "batch_encode",
    "batch_encode_numpy",
    "Lfsr16",
    "BitstreamEncoder",
    "FixedPointLif",
    "DenseLayer",
    "StdpSynapse",
    "SCPNMetrics",
    "BitStreamTensor",
    "VectorizedSCLayer",
    "FixedPointLIFNeuron",
    "SurrogateLif",
    "DifferentiableDenseLayer",
    "StochasticAttention",
    "StochasticGraphLayer",
    "KuramotoSolver",
    "ScGraph",
    "ScGraphBuilder",
    "parse_ir",
    "HDCVector",
    "PetriNetEngine",
    "BrunelNetwork",
    "FixedPointBrunelNetwork",
    "Izhikevich",
    "BitstreamAverager",
    "NetworkRunner",
    "PySpikingControllerPool",
    *(_NEURON_MODELS if _neurons_available else []),
    *(
        [
            "expif_simulate_complete",
            "dpi_neuron_simulate_complete",
            "lapicque_simulate_complete",
            "perfect_integrator_simulate_complete",
            "quadratic_if_simulate_complete",
            "theta_simulate_complete",
        ]
        if _neurons_available
        else []
    ),
    *(_AI_MODELS if _ai_available else []),
    *(
        ["py_simulate_ei_network", "py_batch_simulate", "py_rk4_neuron_simulate"]
        if _studio_rust_available
        else []
    ),
]


# ─── Extracted engine-domain re-exports ───────────────────────────────
try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_evo_batch_crossover as py_evo_batch_crossover,
        py_evo_batch_fitness as py_evo_batch_fitness,
        py_evo_batch_mutate as py_evo_batch_mutate,
        py_evo_diversity as py_evo_diversity,
        py_evo_novelty as py_evo_novelty,
        py_evo_tournament as py_evo_tournament,
    )

    __all__ += [
        "py_evo_batch_crossover",
        "py_evo_batch_fitness",
        "py_evo_batch_mutate",
        "py_evo_diversity",
        "py_evo_novelty",
        "py_evo_tournament",
    ]
    _evolution_rust_available = True
except ImportError:
    _evolution_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_opt_extract_pareto as py_opt_extract_pareto,
        py_opt_sa_search as py_opt_sa_search,
    )

    __all__ += ["py_opt_extract_pareto", "py_opt_sa_search"]
    _optimizer_rust_available = True
except ImportError:
    _optimizer_rust_available = False


# ─── Bridges Rust acceleration paths ──────────────────────────────────
# The bridges/ Python modules (quantum_annealing, dna_mapper, photonic_noc)
# probe these names with `try: from sc_neurocore_engine import py_qa_*`.
# Re-exporting them at this top-level lets the bridges' _HAS_RUST_*
# flags resolve to True when the engine wheel is installed.
try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_qa_batch_ising_energy as py_qa_batch_ising_energy,
        py_qa_gauge_transform as py_qa_gauge_transform,
        py_qa_generate_gauges as py_qa_generate_gauges,
        py_qa_greedy_partition as py_qa_greedy_partition,
        py_qa_ising_energy as py_qa_ising_energy,
        py_qa_simulated_annealing as py_qa_simulated_annealing,
    )

    __all__ += [
        "py_qa_batch_ising_energy",
        "py_qa_gauge_transform",
        "py_qa_generate_gauges",
        "py_qa_greedy_partition",
        "py_qa_ising_energy",
        "py_qa_simulated_annealing",
    ]
    _qa_rust_available = True
except ImportError:
    _qa_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_dna_check_cross_hybridization as py_dna_check_cross_hybridization,
        py_dna_design_orthogonal_set as py_dna_design_orthogonal_set,
        py_dna_design_sequence as py_dna_design_sequence,
        py_dna_detect_hairpins as py_dna_detect_hairpins,
        py_dna_simulate_kinetics as py_dna_simulate_kinetics,
    )

    __all__ += [
        "py_dna_check_cross_hybridization",
        "py_dna_design_orthogonal_set",
        "py_dna_design_sequence",
        "py_dna_detect_hairpins",
        "py_dna_simulate_kinetics",
    ]
    _dna_rust_available = True
except ImportError:
    _dna_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_lgssm_kalman_filter as py_lgssm_kalman_filter,
    )

    __all__ += ["py_lgssm_kalman_filter"]
    _lgssm_rust_available = True
except ImportError:
    _lgssm_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import py_gpfa_em as py_gpfa_em

    __all__ += ["py_gpfa_em"]
    _gpfa_em_rust_available = True
except ImportError:
    _gpfa_em_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import py_phi_star as py_phi_star

    __all__ += ["py_phi_star"]
    _phi_star_rust_available = True
except ImportError:
    _phi_star_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_isolation_distance as py_isolation_distance,
        py_l_ratio as py_l_ratio,
    )

    __all__ += ["py_isolation_distance", "py_l_ratio"]
    _sorting_quality_rust_available = True
except ImportError:
    _sorting_quality_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_pca_components as py_pca_components,
        py_demixed_components as py_demixed_components,
        py_factor_loadings as py_factor_loadings,
    )

    __all__ += ["py_pca_components", "py_demixed_components", "py_factor_loadings"]
    _dimensionality_rust_available = True
except ImportError:
    _dimensionality_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_ollivier_ricci_curvature as py_ollivier_ricci_curvature,
    )

    __all__ += ["py_ollivier_ricci_curvature"]
    _topology_rust_available = True
except ImportError:
    _topology_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_chialvo_map_simulate as py_chialvo_map_simulate,
    )

    __all__ += ["py_chialvo_map_simulate"]
    _chialvo_rust_available = True
except ImportError:
    _chialvo_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_aihara_map_simulate as py_aihara_map_simulate,
    )

    __all__ += ["py_aihara_map_simulate"]
    _aihara_rust_available = True
except ImportError:
    _aihara_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_sc_chaotic_map_simulate as py_sc_chaotic_map_simulate,
    )

    __all__ += ["py_sc_chaotic_map_simulate"]
    _sc_chaotic_map_rust_available = True
except ImportError:
    _sc_chaotic_map_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_nagumo_sato_map_simulate as py_nagumo_sato_map_simulate,
    )

    __all__ += ["py_nagumo_sato_map_simulate"]
    _nagumo_sato_rust_available = True
except ImportError:
    _nagumo_sato_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_sc_adaptive_threshold_map_simulate as py_sc_adaptive_threshold_map_simulate,
    )

    __all__ += ["py_sc_adaptive_threshold_map_simulate"]
    _sc_adaptive_threshold_map_rust_available = True
except ImportError:
    _sc_adaptive_threshold_map_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_cazelles_map_simulate as py_cazelles_map_simulate,
    )

    __all__ += ["py_cazelles_map_simulate"]
    _cazelles_rust_available = True
except ImportError:
    _cazelles_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_sc_clipped_logistic_bursting_map_simulate as py_sc_clipped_logistic_bursting_map_simulate,
    )

    __all__ += ["py_sc_clipped_logistic_bursting_map_simulate"]
    _sc_clipped_logistic_bursting_map_rust_available = True
except ImportError:
    _sc_clipped_logistic_bursting_map_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_sc_clipped_rational_recovery_map_simulate as py_sc_clipped_rational_recovery_map_simulate,
    )

    __all__ += ["py_sc_clipped_rational_recovery_map_simulate"]
    _sc_clipped_rational_recovery_map_rust_available = True
except ImportError:
    _sc_clipped_rational_recovery_map_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_courage_nekorkin_map_simulate as py_courage_nekorkin_map_simulate,
    )

    __all__ += ["py_courage_nekorkin_map_simulate"]
    _courage_nekorkin_rust_available = True
except ImportError:
    _courage_nekorkin_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_mckean_simulate as py_mckean_simulate,
        py_sc_triangular_mckean_simulate as py_sc_triangular_mckean_simulate,
    )

    __all__ += ["py_mckean_simulate", "py_sc_triangular_mckean_simulate"]
    _mckean_rust_available = True
except ImportError:
    _mckean_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_sc_resetting_wilson_hr_simulate as py_sc_resetting_wilson_hr_simulate,
        py_wilson_hr_simulate as py_wilson_hr_simulate,
    )

    __all__ += ["py_wilson_hr_simulate", "py_sc_resetting_wilson_hr_simulate"]
    _wilson_hr_rust_available = True
except ImportError:
    _wilson_hr_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_pernarowski_simulate as py_pernarowski_simulate,
    )

    __all__ += ["py_pernarowski_simulate"]
    _pernarowski_rust_available = True
except ImportError:
    _pernarowski_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_terman_wang_simulate as py_terman_wang_simulate,
    )

    __all__ += ["py_terman_wang_simulate"]
    _terman_wang_rust_available = True
except ImportError:
    _terman_wang_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_coba_lif_simulate as py_coba_lif_simulate,
    )

    __all__ += ["py_coba_lif_simulate"]
    _coba_lif_rust_available = True
except ImportError:
    _coba_lif_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_escape_rate_simulate as py_escape_rate_simulate,
    )

    __all__ += ["py_escape_rate_simulate"]
    _escape_rate_rust_available = True
except ImportError:
    _escape_rate_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_poisson_simulate as py_poisson_simulate,
    )

    __all__ += ["py_poisson_simulate"]
    _poisson_rust_available = True
except ImportError:
    _poisson_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_mcculloch_pitts_evaluate_batch as py_mcculloch_pitts_evaluate_batch,
    )

    __all__ += ["py_mcculloch_pitts_evaluate_batch"]
    _mcculloch_pitts_rust_available = True
except ImportError:
    _mcculloch_pitts_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_iqif_simulate as py_iqif_simulate,
    )

    __all__ += ["py_iqif_simulate"]
    _iqif_rust_available = True
except ImportError:
    _iqif_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_sigmoid_rate_simulate as py_sigmoid_rate_simulate,
    )

    __all__ += ["py_sigmoid_rate_simulate"]
    _sigmoid_rate_rust_available = True
except ImportError:
    _sigmoid_rate_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_threshold_linear_rate_simulate as py_threshold_linear_rate_simulate,
    )

    __all__ += ["py_threshold_linear_rate_simulate"]
    _threshold_linear_rate_rust_available = True
except ImportError:
    _threshold_linear_rate_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_mihalas_niebur_simulate as py_mihalas_niebur_simulate,
    )

    __all__ += ["py_mihalas_niebur_simulate"]
    _mihalas_niebur_rust_available = True
except ImportError:
    _mihalas_niebur_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_sc_scaled_reset_adaptive_if_simulate as py_sc_scaled_reset_adaptive_if_simulate,
    )

    __all__ += ["py_sc_scaled_reset_adaptive_if_simulate"]
    _sc_scaled_reset_adaptive_if_rust_available = True
except ImportError:
    _sc_scaled_reset_adaptive_if_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_glif_simulate as py_glif_simulate,
        py_sc_four_state_glif_simulate as py_sc_four_state_glif_simulate,
    )

    __all__ += ["py_glif_simulate", "py_sc_four_state_glif_simulate"]
    _glif_rust_available = True
except ImportError:
    _glif_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_rulkov_map_simulate as py_rulkov_map_simulate,
    )

    __all__ += ["py_rulkov_map_simulate"]
    _rulkov_rust_available = True
except ImportError:
    _rulkov_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_sc_upward_crossing_rulkov_map_simulate as py_sc_upward_crossing_rulkov_map_simulate,
    )

    __all__ += ["py_sc_upward_crossing_rulkov_map_simulate"]
    _sc_upward_crossing_rulkov_rust_available = True
except ImportError:
    _sc_upward_crossing_rulkov_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_ibarz_tanaka_map_simulate as py_ibarz_tanaka_map_simulate,
    )

    __all__ += ["py_ibarz_tanaka_map_simulate"]
    _ibarz_tanaka_rust_available = True
except ImportError:
    _ibarz_tanaka_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_medvedev_map_simulate as py_medvedev_map_simulate,
    )

    __all__ += ["py_medvedev_map_simulate"]
    _medvedev_rust_available = True
except ImportError:
    _medvedev_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_ermentrout_kopell_map_simulate as py_ermentrout_kopell_map_simulate,
    )

    __all__ += ["py_ermentrout_kopell_map_simulate"]
    _ermentrout_kopell_rust_available = True
except ImportError:
    _ermentrout_kopell_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_fitzhugh_nagumo_simulate as py_fitzhugh_nagumo_simulate,
    )

    __all__ += ["py_fitzhugh_nagumo_simulate"]
    _fitzhugh_nagumo_rust_available = True
except ImportError:
    _fitzhugh_nagumo_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_hindmarsh_rose_simulate as py_hindmarsh_rose_simulate,
    )

    __all__ += ["py_hindmarsh_rose_simulate"]
    _hindmarsh_rose_rust_available = True
except ImportError:
    _hindmarsh_rose_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_fitzhugh_rinzel_simulate as py_fitzhugh_rinzel_simulate,
    )

    __all__ += ["py_fitzhugh_rinzel_simulate"]
    _fitzhugh_rinzel_rust_available = True
except ImportError:
    _fitzhugh_rinzel_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_izhikevich2007_simulate as py_izhikevich2007_simulate,
    )

    __all__ += ["py_izhikevich2007_simulate"]
    _izhikevich2007_rust_available = True
except ImportError:
    _izhikevich2007_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_ph_route_waveguides as py_ph_route_waveguides,
        py_ph_mzi_transfer_matrix as py_ph_mzi_transfer_matrix,
        py_ph_cascade_mzi as py_ph_cascade_mzi,
        py_ph_analyze_crosstalk as py_ph_analyze_crosstalk,
        py_ph_analyze_power_budget as py_ph_analyze_power_budget,
        py_ph_analyze_crosstalk_bank as py_ph_analyze_crosstalk_bank,
        py_ph_analyze_crosstalk_pairs as py_ph_analyze_crosstalk_pairs,
    )

    __all__ += [
        "py_ph_route_waveguides",
        "py_ph_mzi_transfer_matrix",
        "py_ph_cascade_mzi",
        "py_ph_analyze_crosstalk",
        "py_ph_analyze_power_budget",
        "py_ph_analyze_crosstalk_bank",
        "py_ph_analyze_crosstalk_pairs",
    ]
    _ph_rust_available = True
except ImportError:
    _ph_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_inject_bitflip_u8 as py_inject_bitflip_u8,
        py_inject_stuck_at_0_u8 as py_inject_stuck_at_0_u8,
        py_inject_stuck_at_1_u8 as py_inject_stuck_at_1_u8,
        py_inject_dropout_u8 as py_inject_dropout_u8,
        py_inject_gaussian_u8 as py_inject_gaussian_u8,
    )

    __all__ += [
        "py_inject_bitflip_u8",
        "py_inject_stuck_at_0_u8",
        "py_inject_stuck_at_1_u8",
        "py_inject_dropout_u8",
        "py_inject_gaussian_u8",
    ]
    _fault_inject_rust_available = True
except ImportError:
    _fault_inject_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import py_kl_refine as py_kl_refine

    __all__ += ["py_kl_refine"]
    _kl_refine_rust_available = True
except ImportError:
    _kl_refine_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import py_ping_step as py_ping_step

    __all__ += ["py_ping_step"]
    _ping_step_rust_available = True
except ImportError:
    _ping_step_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_parallel_csr_multi_spmv_add as py_parallel_csr_multi_spmv_add,
        py_parallel_csr_spmv_add as py_parallel_csr_spmv_add,
    )

    __all__ += [
        "py_parallel_csr_spmv_add",
        "py_parallel_csr_multi_spmv_add",
    ]
    _parallel_csr_spmv_rust_available = True
except ImportError:
    _parallel_csr_spmv_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_jansen_rit_simulate as py_jansen_rit_simulate,
    )

    __all__ += ["py_jansen_rit_simulate"]
    _jansen_rit_rust_available = True
except ImportError:
    _jansen_rit_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_ermentrout_kopell_pop_simulate as py_ermentrout_kopell_pop_simulate,
    )

    __all__ += ["py_ermentrout_kopell_pop_simulate"]
    _ermentrout_kopell_pop_rust_available = True
except ImportError:
    _ermentrout_kopell_pop_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_resonate_and_fire_simulate as py_resonate_and_fire_simulate,
    )

    __all__ += ["py_resonate_and_fire_simulate"]
    _resonate_and_fire_rust_available = True
except ImportError:
    _resonate_and_fire_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_mat_simulate as py_mat_simulate,
        py_sc_resetting_mat_simulate as py_sc_resetting_mat_simulate,
    )

    __all__ += ["py_mat_simulate", "py_sc_resetting_mat_simulate"]
    _mat_rust_available = True
except ImportError:
    _mat_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_adaptive_threshold_if_simulate as py_adaptive_threshold_if_simulate,
    )

    __all__ += ["py_adaptive_threshold_if_simulate"]
    _adaptive_threshold_if_rust_available = True
except ImportError:
    _adaptive_threshold_if_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_wong_wang_simulate as py_wong_wang_simulate,
    )

    __all__ += ["py_wong_wang_simulate"]
    _wong_wang_rust_available = True
except ImportError:
    _wong_wang_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import py_alpha_simulate as py_alpha_simulate

    __all__ += ["py_alpha_simulate"]
    _alpha_rust_available = True
except ImportError:
    _alpha_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_wilson_cowan_simulate as py_wilson_cowan_simulate,
    )

    __all__ += ["py_wilson_cowan_simulate"]
    _wilson_cowan_rust_available = True
except ImportError:
    _wilson_cowan_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_dcls_max_forward_batch_q88 as py_dcls_max_forward_batch_q88,
    )

    __all__ += ["py_dcls_max_forward_batch_q88"]
    _dcls_tent_rust_available = True
except ImportError:
    _dcls_tent_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_mixed_dense_forward_batch_q88_q1616 as py_mixed_dense_forward_batch_q88_q1616,
    )

    __all__ += ["py_mixed_dense_forward_batch_q88_q1616"]
    _mixed_dense_rust_available = True
except ImportError:
    _mixed_dense_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_adc_to_spike_windows as py_adc_to_spike_windows,
    )

    __all__ += ["py_adc_to_spike_windows"]
    _adc_to_spike_rust_available = True
except ImportError:
    _adc_to_spike_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import py_sc_forward_packed as py_sc_forward_packed

    __all__ += ["py_sc_forward_packed"]
    _sc_forward_rust_available = True
except ImportError:
    _sc_forward_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_predict_xor_ema as py_predict_xor_ema,
        py_predict_xor_lfsr as py_predict_xor_lfsr,
        py_recover_xor_ema as py_recover_xor_ema,
        py_recover_xor_lfsr as py_recover_xor_lfsr,
    )

    __all__ += [
        "py_predict_xor_ema",
        "py_predict_xor_lfsr",
        "py_recover_xor_ema",
        "py_recover_xor_lfsr",
    ]
    _predictive_codec_rust_available = True
except ImportError:
    _predictive_codec_rust_available = False
