# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — UVM generation configuration and simulator targets

"""Define UVM stimulus, coverage, scoreboard, formal, and simulator contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class StimulusConfig:
    """Configuration for randomised SC bitstream stimulus."""

    num_transactions: int = 1000
    bitstream_density_range: Tuple[float, float] = (0.1, 0.9)
    lfsr_seed_range: Tuple[int, int] = (1, 65535)
    enable_corner_cases: bool = True
    max_consecutive_ones: int = 32
    max_consecutive_zeros: int = 32


@dataclass
class CoverageSpec:
    """Functional coverage specification."""

    bitstream_density_bins: int = 10
    spike_rate_bins: int = 5
    scc_bins: int = 8
    cross_coverage: bool = True
    toggle_coverage: bool = True
    target_percent: float = 95.0
    formal_property_map: Dict[str, str] = field(default_factory=dict)


@dataclass
class ScoreboardConfig:
    """Scoreboard configuration for golden model comparison."""

    tolerance_bits: int = 0
    check_popcount: bool = True
    check_probability: bool = True
    check_spike_timing: bool = True
    check_golden_comparison: bool = False
    golden_model_type: str = "bit_true"
    golden_expressions: Dict[str, str] = field(default_factory=dict)


@dataclass
class FormalLink:
    """Link between UVM assertions and SymbiYosys formal proofs."""

    property_name: str
    sby_module: str
    assertion_sv: str
    cover_sv: str


@dataclass
class SimTarget:
    """Simulation tool target for Makefile generation."""

    name: str
    compile_cmd: str
    run_cmd: str
    coverage_cmd: str


SIM_TARGETS = {
    "vcs": SimTarget(
        "vcs",
        "vcs -sverilog -ntb_opts uvm -f {flist} -o simv",
        "./simv +UVM_TESTNAME={test}",
        "urg -dir simv.vdb -format both",
    ),
    "questa": SimTarget(
        "questa",
        "vlog -sv +incdir+$UVM_HOME/src -f {flist}",
        'vsim -c -do "run -all" +UVM_TESTNAME={test} work.tb_{module}_top',
        "vcover merge -out merged.ucdb *.ucdb",
    ),
    "xcelium": SimTarget(
        "xcelium",
        "xrun -sv -uvm -f {flist} -elaborate",
        "xrun -R +UVM_TESTNAME={test}",
        'imc -load cov_work/scope/test -exec "report -metrics -out cov.rpt"',
    ),
}


SPDX_HEADER = """\
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li"""
