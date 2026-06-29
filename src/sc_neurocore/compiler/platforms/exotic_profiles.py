# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

"""Register non-CMOS and research hardware-profile targets.

Every profile here is registered exactly once through :func:`registry._reg`,
which rejects a duplicate name at import time. The targets are grouped by
physical substrate: superconducting/cryogenic, wetware (living neurons),
electrochemical/memristive, wafer-scale, quantum-neuromorphic, optical I/O,
acoustic/phononic, fluidic, space-qualified, magnonic, organic-bioelectronic,
RISC-V sovereign, thermodynamic, probabilistic (p-bit), polariton, metamaterial,
molecular/chemical, reversible/adiabatic, and microfluidic/mechanical.

Where a substrate has no meaningful binary word (continuous-time wetware,
single-molecule chemistry), ``data_width``/``fraction`` encode the host-side
stimulation/encoding word used to drive the device, not an on-device register.
"""

from __future__ import annotations

from .registry import HardwareProfile, _reg

# ── Superconducting / Cryogenic ──────────────────────────────────────

_reg(
    HardwareProfile(
        name="nist_sfq",
        vendor="NIST",
        family="SFQ",
        platform_class="superconducting",
        data_width=8,
        fraction=4,
        overflow="wrap",
        rounding="truncate",
        max_freq_mhz=100000,  # 100 GHz SFQ
        notes="NIST SFQ: Single Flux Quantum logic. 100 GHz, µW at 4K.",
    )
)
_reg(
    HardwareProfile(
        name="northrop_aqfp",
        vendor="Northrop Grumman",
        family="AQFP",
        platform_class="superconducting",
        data_width=8,
        fraction=4,
        overflow="wrap",
        rounding="truncate",
        max_freq_mhz=5000,  # 5 GHz AQFP
        notes="Northrop Grumman AQFP: Adiabatic QFP. 5 GHz, near-zero power.",
    )
)
_reg(
    HardwareProfile(
        name="josephson_jj",
        vendor="Research",
        family="Josephson",
        platform_class="superconducting",
        data_width=8,
        fraction=4,
        overflow="wrap",
        rounding="truncate",
        max_freq_mhz=50000,
        notes="Josephson Junction neurons: superconducting neuron analogue.",
    )
)

# ── Wetware / Biological ─────────────────────────────────────────────
# Living neural substrates: the host word encodes the electrode stimulation
# protocol, not an on-device register. Continuous-time, asynchronous.

_reg(
    HardwareProfile(
        name="finalspark_neuroplatform",
        vendor="FinalSpark",
        family="Neuroplatform",
        platform_class="wetware",
        data_width=8,
        fraction=0,
        max_freq_mhz=0,
        notes="Organoid biocomputing platform with electrophysiological API.",
    )
)
_reg(
    HardwareProfile(
        name="cortical_labs_dishbrain",
        vendor="Cortical Labs",
        family="DishBrain",
        platform_class="wetware",
        data_width=8,
        fraction=0,
        max_freq_mhz=0,
        notes="Biological neuronal network in closed-loop MEA array. Asynchronous continuous time.",
    )
)

# ── Electrochemical / Memristive ─────────────────────────────────────

_reg(
    HardwareProfile(
        name="ibm_ecram",
        vendor="IBM",
        family="ECRAM-AnalogAI",
        platform_class="electrochemical",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="IBM ECRAM: electrochemical RAM. High linearity for on-chip "
        "learning. Multi-level analog weights.",
    )
)
_reg(
    HardwareProfile(
        name="samsung_pcram",
        vendor="Samsung",
        family="PCRAM",
        platform_class="electrochemical",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Samsung PCRAM: phase-change memory compute. Non-volatile, "
        "multi-level cell analog synapses.",
    )
)
_reg(
    HardwareProfile(
        name="stanford_ecram",
        vendor="Stanford",
        family="ECRAM-Research",
        platform_class="electrochemical",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Stanford ECRAM: research-grade electrochemical synapse array. "
        "WO₃-based, 10⁶ endurance cycles.",
    )
)

# ── Wafer-Scale ──────────────────────────────────────────────────────

_reg(
    HardwareProfile(
        name="cerebras_wse3_ws",
        vendor="Cerebras",
        family="WSE-3-WS",
        platform_class="wafer_scale",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        dsp_block="PE",
        dsp_mult_a=16,
        dsp_mult_b=16,
        max_freq_mhz=1000,
        notes="Cerebras WSE-3 wafer-scale: 900K cores, 44GB SRAM, 4 Tflop.",
    )
)
_reg(
    HardwareProfile(
        name="tesla_dojo3",
        vendor="Tesla",
        family="Dojo-3",
        platform_class="wafer_scale",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        dsp_block="TU",
        dsp_mult_a=16,
        dsp_mult_b=16,
        max_freq_mhz=2000,
        notes="Tesla Dojo 3: in-house wafer-scale AI supercomputer tile.",
    )
)
_reg(
    HardwareProfile(
        name="tachyum_prodigy",
        vendor="Tachyum",
        family="Prodigy-2nm",
        platform_class="wafer_scale",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        max_freq_mhz=5500,
        notes="Tachyum Prodigy: universal processor. CPU+GPU+AI in one die.",
    )
)

# ── Quantum Neuromorphic ─────────────────────────────────────────────

_reg(
    HardwareProfile(
        name="ibm_qnn",
        vendor="IBM",
        family="Quantum-NN",
        platform_class="quantum_neuro",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        notes="IBM Quantum Neural Network: superconducting transmon qubits "
        "for quantum reservoir computing.",
    )
)
_reg(
    HardwareProfile(
        name="ionq_trapped_ion",
        vendor="IonQ",
        family="Trapped-Ion-QNN",
        platform_class="quantum_neuro",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        notes="IonQ Trapped-Ion QNN: all-to-all connectivity for quantum SNN simulation.",
    )
)

# ── Optical Interconnect / CPO ───────────────────────────────────────

_reg(
    HardwareProfile(
        name="ayar_teraphy",
        vendor="Ayar Labs",
        family="TeraPHY",
        platform_class="optical_io",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        max_freq_mhz=25000,
        notes="Ayar Labs TeraPHY: silicon photonic I/O chiplet. "
        "8 Tbps bidirectional, UCIe-compatible.",
    )
)
_reg(
    HardwareProfile(
        name="intel_cpo",
        vendor="Intel",
        family="CPO",
        platform_class="optical_io",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        max_freq_mhz=20000,
        notes="Intel co-packaged optics: silicon photonic I/O for "
        "die-to-die and rack-scale optical links.",
    )
)

# ── Acoustic / Phononic ──────────────────────────────────────────────

_reg(
    HardwareProfile(
        name="mit_phononic",
        vendor="MIT",
        family="Phononic-NN",
        platform_class="acoustic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="MIT phononic neural network: acoustic wave reservoir "
        "computing in MEMS resonator arrays.",
    )
)
_reg(
    HardwareProfile(
        name="caltech_mems_nn",
        vendor="Caltech",
        family="MEMS-NN",
        platform_class="acoustic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Caltech MEMS neural processor: mechanical resonator array for edge inference.",
    )
)

# ── Fluidic / Microfluidic ───────────────────────────────────────────

_reg(
    HardwareProfile(
        name="stanford_microfluidic",
        vendor="Stanford",
        family="µFluidic-NN",
        platform_class="fluidic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Stanford microfluidic neural network: droplet-based "
        "logic gates for lab-on-chip compute.",
    )
)
_reg(
    HardwareProfile(
        name="eth_fluidic_logic",
        vendor="ETH Zurich",
        family="Fluidic-Logic",
        platform_class="fluidic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="ETH Zurich fluidic logic: pressure-driven bistable "
        "valves for chemical neural computation.",
    )
)

# ── Space-Qualified ──────────────────────────────────────────────────

_reg(
    HardwareProfile(
        name="bae_rad750_sq",
        vendor="BAE Systems",
        family="RAD750",
        platform_class="space_qualified",
        data_width=32,
        fraction=16,
        overflow="saturate",
        rounding="nearest",
        max_freq_mhz=200,
        notes="BAE RAD750: radiation-hardened processor. Mars rovers, ISS, deep-space missions.",
    )
)
_reg(
    HardwareProfile(
        name="seakr_sbc",
        vendor="SEAKR",
        family="SBC-SpaceAI",
        platform_class="space_qualified",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        max_freq_mhz=400,
        notes="SEAKR SpaceAI SBC: radiation-tolerant single-board "
        "computer for on-orbit neural inference.",
    )
)
_reg(
    HardwareProfile(
        name="vorago_va10820",
        vendor="Vorago",
        family="VA10820",
        platform_class="space_qualified",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        max_freq_mhz=100,
        notes="Vorago VA10820: Arm Cortex-M0 rad-hard MCU for space-grade edge neural processing.",
    )
)
_reg(
    HardwareProfile(
        name="frontgrade_leon5",
        vendor="Frontgrade",
        family="LEON5-FT",
        platform_class="space_qualified",
        data_width=32,
        fraction=16,
        overflow="saturate",
        rounding="nearest",
        max_freq_mhz=250,
        notes="Frontgrade LEON5-FT: SPARC V8 rad-hard for ESA/NASA "
        "mission-critical neural control.",
    )
)

# ── Magnonic / Skyrmion ──────────────────────────────────────────────

_reg(
    HardwareProfile(
        name="tum_skyrmion",
        vendor="TU Munich",
        family="SkyANN-v1",
        platform_class="magnonic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Skyrmion-based reservoir computing. EU SkyANN project. "
        "Topological stability enables ultra-low-power edge AI.",
    )
)
_reg(
    HardwareProfile(
        name="kaist_spinwave",
        vendor="KAIST",
        family="SpinWave-RC",
        platform_class="magnonic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Spin-wave interference reservoir. Field-free operation via "
        "SOT bilayer nanostructures.",
    )
)
_reg(
    HardwareProfile(
        name="imec_mtj_reservoir",
        vendor="imec",
        family="MTJ-Reservoir",
        platform_class="magnonic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Magnetic tunnel junction reservoir computing array. "
        "Sub-fJ switching energy per MAC operation.",
    )
)

# ── Organic Bioelectronic ────────────────────────────────────────────

_reg(
    HardwareProfile(
        name="cambridge_oect",
        vendor="Cambridge",
        family="OECT-Synapse",
        platform_class="organic_bioelectronic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Organic Electrochemical Transistor synapse. PEDOT:PSS "
        "channel for in-vivo bioelectronic neural interfaces.",
    )
)
_reg(
    HardwareProfile(
        name="linkoping_organic",
        vendor="Linköping",
        family="Organic-NN",
        platform_class="organic_bioelectronic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Printed organic transistor array. Biodegradable substrate "
        "for disposable sensor-neural-interface.",
    )
)

# ── RISC-V Sovereign AI ──────────────────────────────────────────────

_reg(
    HardwareProfile(
        name="sifive_x280_ai",
        vendor="SiFive",
        family="X280-AI",
        platform_class="risc_v_sovereign",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        max_freq_mhz=2000,
        notes="SiFive Intelligence X280: RISC-V vector AI core. "
        "Open ISA, no ITAR restrictions, sovereign compute.",
    )
)
_reg(
    HardwareProfile(
        name="esperanto_et_soc",
        vendor="Esperanto",
        family="ET-SoC-1",
        platform_class="risc_v_sovereign",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="nearest",
        max_freq_mhz=1000,
        notes="Esperanto ET-SoC-1: 1000+ RISC-V cores for sovereign "
        "AI inference. No export control dependencies.",
    )
)
_reg(
    HardwareProfile(
        name="ventana_veyron_ai",
        vendor="Ventana",
        family="Veyron-V2",
        platform_class="risc_v_sovereign",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        max_freq_mhz=3600,
        notes="Ventana Veyron V2: high-perf RISC-V with AI extensions. "
        "Chiplet-based, UCIe-compatible.",
    )
)
_reg(
    HardwareProfile(
        name="tenstorrent_ascalon",
        vendor="Tenstorrent",
        family="Ascalon",
        platform_class="risc_v_sovereign",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        max_freq_mhz=4000,
        notes="Tenstorrent Ascalon: RISC-V AI server processor. "
        "Open-source ISA for data-sovereign deployments.",
    )
)
_reg(
    HardwareProfile(
        name="andes_ax45mpv",
        vendor="Andes",
        family="AX45MPV",
        platform_class="risc_v_sovereign",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        max_freq_mhz=1500,
        notes="Andes AX45MPV: multiprocessor RISC-V with vector extension. "
        "Targets automotive and edge AI sovereignty.",
    )
)

# ── Thermodynamic Computing ──────────────────────────────────────────

_reg(
    HardwareProfile(
        name="extropic_epu",
        vendor="Extropic",
        family="EPU-v1",
        platform_class="thermodynamic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Extropic Energy-Based Processor: probabilistic generative AI "
        "via controlled thermal fluctuations. Room-temperature.",
    )
)
_reg(
    HardwareProfile(
        name="normal_cn101",
        vendor="Normal Computing",
        family="CN101",
        platform_class="thermodynamic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Normal Computing CN101: thermodynamic AI chip. Stochastic "
        "sampling via thermal noise exploitation.",
    )
)

# ── Probabilistic / p-Bit ────────────────────────────────────────────

_reg(
    HardwareProfile(
        name="purdue_pbit",
        vendor="Purdue",
        family="p-Bit-Array",
        platform_class="probabilistic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Purdue MRAM p-bit array: room-temperature probabilistic "
        "computing. Boltzmann machine substrate.",
    )
)
_reg(
    HardwareProfile(
        name="tohoku_sot_pbit",
        vendor="Tohoku",
        family="SOT-pBit",
        platform_class="probabilistic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Tohoku SOT-MRAM probabilistic computing: tuneable "
        "fluctuation rate via spin-orbit torque bias.",
    )
)

# ── Polariton / Exciton ──────────────────────────────────────────────

_reg(
    HardwareProfile(
        name="marvell_polariton",
        vendor="Marvell",
        family="Polariton-PIC",
        platform_class="polariton",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Marvell/Polariton Technologies: silicon photonic + plasmonic "
        "active devices for ultrafast optical neural compute.",
    )
)
_reg(
    HardwareProfile(
        name="stanford_polariton",
        vendor="Stanford",
        family="Perovskite-RC",
        platform_class="polariton",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Stanford perovskite microcavity: exciton-polariton "
        "condensate reservoir computing at room temperature.",
    )
)

# ── Metamaterial / Programmable Matter ───────────────────────────────

_reg(
    HardwareProfile(
        name="mit_metamaterial",
        vendor="MIT",
        family="RF-Metasurface",
        platform_class="metamaterial",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="MIT RF metasurface neural network: programmable "
        "unit-cell phases perform analog matrix-vector multiply.",
    )
)
_reg(
    HardwareProfile(
        name="penn_acoustic_meta",
        vendor="UPenn",
        family="Acoustic-Meta",
        platform_class="metamaterial",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="UPenn acoustic metamaterial classifier: mechanical "
        "wave propagation implements inference at zero digital power.",
    )
)

# ── Molecular / Chemical ─────────────────────────────────────────────

_reg(
    HardwareProfile(
        name="catalog_dna_compute",
        vendor="Catalog",
        family="Shannon",
        platform_class="molecular",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Catalog DNA computation platform. Translates neural inference "
        "into parallel molecular search operations and enzymatic reactions.",
    )
)
_reg(
    HardwareProfile(
        name="biomemory_dna",
        vendor="Biomemory",
        family="DNA-Drive",
        platform_class="molecular",
        data_width=2,
        fraction=0,
        max_freq_mhz=0,
        notes="DNA storage mapped to compute-in-memory sequences. "
        "Ultra-high latency, exabyte capacity.",
    )
)
_reg(
    HardwareProfile(
        name="belousov_zhabotinsky",
        vendor="Academic",
        family="BZ Reaction",
        platform_class="molecular",
        data_width=1,
        fraction=0,
        max_freq_mhz=0,
        notes="Chemical oscillator networks for reaction-diffusion computation.",
    )
)

# ── Reversible / Adiabatic ───────────────────────────────────────────

_reg(
    HardwareProfile(
        name="superconducting_aqfp",
        vendor="Yokohama Univ",
        family="AQFP",
        platform_class="reversible",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        notes="Adiabatic Quantum-Flux-Parametron. Operates at the Landauer limit "
        "via multi-phase resonant clocking. Zero static power dissipation.",
    )
)
_reg(
    HardwareProfile(
        name="scrl_logic",
        vendor="Generic",
        family="SCRL",
        platform_class="reversible",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        notes="Split-Level Charge Recovery Logic. Implements Fredkin and Toffoli "
        "gates for thermodynamically reversible CMOS neural computation.",
    )
)
_reg(
    HardwareProfile(
        name="fujitsu_digital_annealer",
        vendor="Fujitsu",
        family="Digital Annealer",
        platform_class="reversible",
        data_width=64,
        fraction=0,
        max_freq_mhz=1000,
        notes="CMOS-based reversible logic for combinatorial optimization. "
        "Landauer limit approximation.",
    )
)
_reg(
    HardwareProfile(
        name="rl_toffoli_asic",
        vendor="Custom",
        family="Reversible ASIC",
        platform_class="reversible",
        data_width=32,
        fraction=16,
        max_freq_mhz=500,
        notes="Standard cell library mapped strictly to Toffoli and Fredkin gates "
        "for zero dissipation.",
    )
)

# ── Microfluidic / Mechanical ────────────────────────────────────────

_reg(
    HardwareProfile(
        name="nanofluidic_logic",
        vendor="EPFL",
        family="Ion-Channel",
        platform_class="microfluidic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Nanofluidic computing using 2D material ionic channels. "
        "Simulates biological ion dynamics directly using physical water/ion flow.",
    )
)
_reg(
    HardwareProfile(
        name="mems_neuromorphic",
        vendor="Generic",
        family="MEMS-Resonator",
        platform_class="microfluidic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Micro-electromechanical systems (MEMS) coupled oscillators. "
        "Nonlinear mechanical resonance implements spiking neuron membranes.",
    )
)
_reg(
    HardwareProfile(
        name="ibm_microfluidic",
        vendor="IBM",
        family="Electronic Blood",
        platform_class="microfluidic",
        data_width=4,
        fraction=0,
        max_freq_mhz=0,
        notes="Fluidic redox batteries driving micro-scale continuous flow logic.",
    )
)
_reg(
    HardwareProfile(
        name="mems_resonator",
        vendor="SiTime",
        family="MEMS Logic",
        platform_class="microfluidic",
        data_width=12,
        fraction=0,
        max_freq_mhz=50,
        notes="Coupled mechanical oscillators simulating non-linear dynamic states.",
    )
)
