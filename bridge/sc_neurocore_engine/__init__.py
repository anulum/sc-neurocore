# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Engine — 111 Rust neuron models + SIMD primitives + IR

"""SC-NeuroCore Engine — 111 Rust neuron models + SIMD primitives + IR compiler."""

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        __version__,
        simd_tier,
        set_num_threads,
        pack_bitstream,
        unpack_bitstream,
        popcount,
        pack_bitstream_numpy,
        popcount_numpy,
        unpack_bitstream_numpy,
        batch_lif_run,
        batch_lif_run_multi,
        batch_lif_run_varying,
        batch_encode,
        batch_encode_numpy,
        Lfsr16,
        BitstreamEncoder,
        FixedPointLif,
        DenseLayer,
        StdpSynapse,
        SCPNMetrics,
        BitStreamTensor,
        BrunelNetwork,
    )
except ImportError:
    _core_available = False
else:
    _core_available = True

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        QuadraticIFNeuron,
        ThetaNeuron,
        PerfectIntegratorNeuron,
        GatedLIFNeuron,
        NonlinearLIFNeuron,
        SFANeuron,
        MATNeuron,
        EscapeRateNeuron,
        KLIFNeuron,
        InhibitoryLIFNeuron,
        ComplementaryLIFNeuron,
        ParametricLIFNeuron,
        NonResettingLIFNeuron,
        AdaptiveThresholdIFNeuron,
        SigmaDeltaNeuron,
        EnergyLIFNeuron,
        IntegerQIFNeuron,
        ClosedFormContinuousNeuron,
        FitzHughNagumoNeuron,
        MorrisLecarNeuron,
        HindmarshRoseNeuron,
        ResonateAndFireNeuron,
        FitzHughRinzelNeuron,
        McKeanNeuron,
        TermanWangOscillator,
        BendaHerzNeuron,
        AlphaNeuron,
        COBALIFNeuron,
        GutkinErmentroutNeuron,
        WilsonHRNeuron,
        ChayNeuron,
        ChayKeizerNeuron,
        ShermanRinzelKeizerNeuron,
        ButeraRespiratoryNeuron,
        EPropALIFNeuron,
        SuperSpikeNeuron,
        LearnableNeuronModel,
        PernarowskiNeuron,
        ChialvoMapNeuron,
        RulkovMapNeuron,
        IbarzTanakaMapNeuron,
        MedvedevMapNeuron,
        CazellesMapNeuron,
        CourageNekorkinMapNeuron,
        HodgkinHuxleyNeuron,
        TraubMilesNeuron,
        WangBuzsakiNeuron,
        ConnorStevensNeuron,
        DestexheThalamicNeuron,
        HuberBraunNeuron,
        GolombFSNeuron,
        PospischilNeuron,
        MainenSejnowskiNeuron,
        DeSchutterPurkinjeNeuron,
        PlantR15Neuron,
        PrescottNeuron,
        MihalasNieburNeuron,
        GLIFNeuron,
        GIFPopulationNeuron,
        AvRonCardiacNeuron,
        DurstewitzDopamineNeuron,
        HillTononiNeuron,
        BertramPhantomBurster,
        YamadaNeuron,
        PinskyRinzelNeuron,
        HayL5PyramidalNeuron,
        MarderSTGNeuron,
        RallCableNeuron,
        BoothRinzelNeuron,
        DendrifyNeuron,
        TwoCompartmentLIFNeuron,
        PoissonNeuron,
        InhomogeneousPoissonNeuron,
        GammaRenewalNeuron,
        StochasticIFNeuron,
        GalvesLocherbachNeuron,
        SpikeResponseNeuron,
        GLMNeuron,
        WilsonCowanUnit,
        JansenRitUnit,
        WongWangUnit,
        ErmentroutKopellPopulation,
        WendlingNeuron,
        LarterBreakspearNeuron,
        LoihiCUBANeuron,
        Loihi2Neuron,
        TrueNorthNeuron,
        BrainScaleSAdExNeuron,
        SpiNNakerLIFNeuron,
        SpiNNaker2Neuron,
        DPINeuron,
        AkidaNeuron,
        NeuroGridNeuron,
        McCullochPittsNeuron,
        SigmoidRateNeuron,
        ThresholdLinearRateNeuron,
        AstrocyteModel,
        TsodyksMarkramNeuron,
        LiquidTimeConstantNeuron,
        CompteWMNeuron,
        SiegertTransferFunction,
        FractionalLIFNeuron,
        ParallelSpikingNeuron,
        AmariNeuralField,
        LeakyCompeteFireNeuron,
        AdExNeuron,
        ExpIFNeuron,
        LapicqueNeuron,
    )

    _neurons_available = True
except ImportError:
    _neurons_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        ArcaneNeuron,
        AttentionGatedNeuron,
        CompositionalBindingNeuron,
        DifferentiableSurrogateNeuron,
        MetaPlasticNeuron,
        MultiTimescaleNeuron,
        PredictiveCodingNeuron,
        SelfReferentialNeuron,
        RustContinuousAttractorNeuron as ContinuousAttractorNeuron,
        Izhikevich,
        BitstreamAverager,
        NetworkRunner,
    )

    _ai_available = True
except ImportError:
    _ai_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_simulate_ei_network,
        py_batch_simulate,
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
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            py_simulate_ei_network = _mod.py_simulate_ei_network
            py_batch_simulate = _mod.py_batch_simulate
            _studio_rust_available = True
        else:
            _studio_rust_available = False
    except Exception:
        _studio_rust_available = False

if _core_available:
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
else:
    # Rust engine not built — raise ImportError so pytest.importorskip works
    _bridge_available = False
    raise ImportError(
        "sc_neurocore_engine native module not found. "
        "Build with: cd engine && maturin develop --release"
    )

_NEURON_MODELS = [
    "QuadraticIFNeuron",
    "ThetaNeuron",
    "PerfectIntegratorNeuron",
    "GatedLIFNeuron",
    "NonlinearLIFNeuron",
    "SFANeuron",
    "MATNeuron",
    "EscapeRateNeuron",
    "KLIFNeuron",
    "InhibitoryLIFNeuron",
    "ComplementaryLIFNeuron",
    "ParametricLIFNeuron",
    "NonResettingLIFNeuron",
    "AdaptiveThresholdIFNeuron",
    "SigmaDeltaNeuron",
    "EnergyLIFNeuron",
    "IntegerQIFNeuron",
    "ClosedFormContinuousNeuron",
    "FitzHughNagumoNeuron",
    "MorrisLecarNeuron",
    "HindmarshRoseNeuron",
    "ResonateAndFireNeuron",
    "FitzHughRinzelNeuron",
    "McKeanNeuron",
    "TermanWangOscillator",
    "BendaHerzNeuron",
    "AlphaNeuron",
    "COBALIFNeuron",
    "GutkinErmentroutNeuron",
    "WilsonHRNeuron",
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
    "IbarzTanakaMapNeuron",
    "MedvedevMapNeuron",
    "CazellesMapNeuron",
    "CourageNekorkinMapNeuron",
    "HodgkinHuxleyNeuron",
    "TraubMilesNeuron",
    "WangBuzsakiNeuron",
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
    "GLIFNeuron",
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
    "Izhikevich",
    "BitstreamAverager",
    "NetworkRunner",
    *(_NEURON_MODELS if _neurons_available else []),
    *(_AI_MODELS if _ai_available else []),
    *(["py_simulate_ei_network", "py_batch_simulate"] if _studio_rust_available else []),
]


# ─── Bridges Rust acceleration paths ──────────────────────────────────
# The bridges/ Python modules (quantum_annealing, dna_mapper, photonic_noc)
# probe these names with `try: from sc_neurocore_engine import py_qa_*`.
# Re-exporting them at this top-level lets the bridges' _HAS_RUST_*
# flags resolve to True when the engine wheel is installed.
try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_qa_batch_ising_energy,
        py_qa_gauge_transform,
        py_qa_generate_gauges,
        py_qa_greedy_partition,
        py_qa_ising_energy,
        py_qa_simulated_annealing,
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
        py_dna_check_cross_hybridization,
        py_dna_design_orthogonal_set,
        py_dna_design_sequence,
        py_dna_detect_hairpins,
        py_dna_simulate_kinetics,
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
        py_lgssm_kalman_filter,
    )

    __all__ += ["py_lgssm_kalman_filter"]
    _lgssm_rust_available = True
except ImportError:
    _lgssm_rust_available = False

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        py_inject_bitflip_u8,
        py_inject_stuck_at_0_u8,
        py_inject_stuck_at_1_u8,
        py_inject_dropout_u8,
        py_inject_gaussian_u8,
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
    from sc_neurocore_engine.sc_neurocore_engine import py_kl_refine
    __all__ += ["py_kl_refine"]
    _kl_refine_rust_available = True
except ImportError:
    _kl_refine_rust_available = False
