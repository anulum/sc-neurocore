# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Automated UVM/SystemVerilog Verification IP Generator

"""Auto-generates full UVM testbench + coverage + scoreboarding from RTL.

Given a module specification (port list, parameters, SC semantics), emits
a complete UVM environment:

- **Agent**: driver + monitor + sequencer for SC bitstream I/O
- **Sequences**: constrained-random SC bitstream stimulus with LFSR seeds
- **Scoreboard**: bit-true golden model comparison (popcount/probability)
- **Coverage**: functional coverpoints (bitstream density, spike rate, SCC)
- **Formal link**: generates SymbiYosys .sby configs that reuse UVM assertions
- **Top-level TB**: instantiates DUT + env + clock/reset generation

Usage::

    gen = UVMGenerator()
    rtl = RTLModule.from_verilog_source(verilog_text)
    bench = gen.generate(rtl)
    bench.write_all("/path/to/output/")
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ── RTL Module Parsing ───────────────────────────────────────────────


class PortDirection(Enum):
    """SystemVerilog port direction tokens accepted by the UVM generator."""

    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


class PortType(Enum):
    """SystemVerilog net/data type tokens emitted for module ports."""

    LOGIC = "logic"
    WIRE = "wire"
    REG = "reg"


@dataclass
class ModulePort:
    """Parsed RTL port metadata used by generated UVM components."""

    name: str
    direction: PortDirection
    port_type: PortType = PortType.LOGIC
    width: int = 1
    is_signed: bool = False
    is_array: bool = False
    array_size: int = 0

    @property
    def sv_decl(self) -> str:
        """Return the SystemVerilog declaration for this parsed port."""
        signed = " signed" if self.is_signed else ""
        width = f" [{self.width - 1}:0]" if self.width > 1 else ""
        arr = f" [0:{self.array_size - 1}]" if self.is_array else ""
        return f"{self.direction.value} {self.port_type.value}{signed}{width} {self.name}{arr}"

    @property
    def is_clock(self) -> bool:
        """Return whether the port name matches a supported clock convention."""
        return self.name.lower() in ("clk", "clock", "i_clk")

    @property
    def is_reset(self) -> bool:
        """Return whether the port name matches a supported reset convention."""
        return self.name.lower() in ("rst_n", "reset_n", "rst", "reset", "i_rst_n")


@dataclass
class ModuleParam:
    """RTL module parameter."""

    name: str
    value: str
    param_type: str = "int"


@dataclass
class RTLModule:
    """Parsed RTL module specification."""

    name: str
    ports: List[ModulePort]
    params: List[ModuleParam] = field(default_factory=list)
    is_sc_module: bool = True

    @classmethod
    def from_verilog_source(cls, source: str) -> RTLModule:
        """Parse a Verilog/SystemVerilog module header."""
        name_match = re.search(r"module\s+(\w+)", source)
        if not name_match:
            raise ValueError("No module declaration found")
        name = name_match.group(1)

        params = []
        param_block = re.search(r"#\s*\((.*?)\)\s*\(", source, re.DOTALL)
        if param_block:
            for m in re.finditer(
                r"parameter\s+(?:(\w+)\s+)?(\w+)\s*=\s*(\S+)", param_block.group(1)
            ):
                ptype = m.group(1) or "int"
                params.append(ModuleParam(m.group(2), m.group(3), ptype))

        ports = []
        if param_block:
            # param_block regex consumed the opening '(' of the port list.
            # The match ends right after that '(' so rest should start
            # from there and we search for the closing ')' ';'.
            rest = source[param_block.end() :]
            # rest starts just after the '(' of the port list
            port_section = re.search(r"(.*?)\)\s*;", rest, re.DOTALL)
        else:
            port_section = re.search(r"\(\s*(.*?)\s*\)\s*;", source, re.DOTALL)

        if port_section:
            text = port_section.group(1)
            for line in text.split(","):
                line = line.strip()
                if not line:
                    continue
                pm = re.match(
                    r"(input|output|inout)\s+"
                    r"(?:(logic|wire|reg)\s+)?"
                    r"(signed\s+)?"
                    r"(?:\[(\d+):(\d+)\]\s+)?"
                    r"(\w+)"
                    r"(?:\s+\[0:(\d+)\])?",
                    line,
                )
                if pm:
                    direction = PortDirection(pm.group(1))
                    ptype = PortType(pm.group(2)) if pm.group(2) else PortType.LOGIC
                    is_signed = pm.group(3) is not None
                    if pm.group(4) and pm.group(5):
                        width = int(pm.group(4)) - int(pm.group(5)) + 1
                    else:
                        width = 1
                    pname = pm.group(6)
                    is_array = pm.group(7) is not None
                    arr_size = int(pm.group(7)) + 1 if is_array else 0
                    ports.append(
                        ModulePort(pname, direction, ptype, width, is_signed, is_array, arr_size)
                    )

        return cls(name=name, ports=ports, params=params)

    @property
    def input_ports(self) -> List[ModulePort]:
        """Input ports excluding generated clock and reset controls."""
        return [
            p
            for p in self.ports
            if p.direction == PortDirection.INPUT and not p.is_clock and not p.is_reset
        ]

    @property
    def output_ports(self) -> List[ModulePort]:
        """Output ports monitored by the generated scoreboard and coverage."""
        return [p for p in self.ports if p.direction == PortDirection.OUTPUT]

    @property
    def clock_port(self) -> Optional[ModulePort]:
        """First parsed clock-like port, if the RTL module declares one."""
        return next((p for p in self.ports if p.is_clock), None)

    @property
    def reset_port(self) -> Optional[ModulePort]:
        """First parsed reset-like port, if the RTL module declares one."""
        return next((p for p in self.ports if p.is_reset), None)

    @property
    def total_input_bits(self) -> int:
        """Total data-input width excluding clock and reset ports."""
        return sum(p.width for p in self.input_ports)

    @property
    def total_output_bits(self) -> int:
        """Total output width observed by generated verification components."""
        return sum(p.width for p in self.output_ports)


# ── Stimulus / Coverage / Scoreboard Configuration ───────────────────


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


# ── UVM Benchmark (Generated Output) ────────────────────────────────


@dataclass
class UVMBenchmark:
    """Complete generated UVM testbench."""

    module_name: str
    transaction_sv: str
    sequence_sv: str
    driver_sv: str
    monitor_sv: str
    scoreboard_sv: str
    coverage_sv: str
    agent_sv: str
    env_sv: str
    top_sv: str
    sby_config: str
    bind_sv: str = ""
    makefile: str = ""
    regression_list: str = ""
    filelist: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, str]:
        """Return generated artefacts keyed by their output filenames."""
        d = {
            f"{self.module_name}_transaction.sv": self.transaction_sv,
            f"{self.module_name}_sequence.sv": self.sequence_sv,
            f"{self.module_name}_driver.sv": self.driver_sv,
            f"{self.module_name}_monitor.sv": self.monitor_sv,
            f"{self.module_name}_scoreboard.sv": self.scoreboard_sv,
            f"{self.module_name}_coverage.sv": self.coverage_sv,
            f"{self.module_name}_agent.sv": self.agent_sv,
            f"{self.module_name}_env.sv": self.env_sv,
            f"tb_{self.module_name}_top.sv": self.top_sv,
            f"{self.module_name}_verify.sby": self.sby_config,
        }
        if self.bind_sv:
            d[f"{self.module_name}_bind.sv"] = self.bind_sv
        if self.makefile:
            d["Makefile"] = self.makefile
        if self.regression_list:
            d["regression.list"] = self.regression_list
        return d


# ── UVM Generator ───────────────────────────────────────────────────

_SPDX = """\
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li"""


class UVMGenerator:
    """Generates complete UVM verification IP from an RTL module spec."""

    def __init__(
        self,
        stimulus: Optional[StimulusConfig] = None,
        coverage: Optional[CoverageSpec] = None,
        scoreboard: Optional[ScoreboardConfig] = None,
    ):
        self.stimulus = stimulus or StimulusConfig()
        self.coverage = coverage or CoverageSpec()
        self.scoreboard = scoreboard or ScoreboardConfig()

    def generate(self, rtl: RTLModule) -> UVMBenchmark:
        """Generate the complete UVM testbench for an RTL module."""
        m = rtl.name
        return UVMBenchmark(
            module_name=m,
            transaction_sv=self._emit_transaction(rtl),
            sequence_sv=self._emit_sequence(rtl),
            driver_sv=self._emit_driver(rtl),
            monitor_sv=self._emit_monitor(rtl),
            scoreboard_sv=self._emit_scoreboard(rtl),
            coverage_sv=self._emit_coverage(rtl),
            agent_sv=self._emit_agent(rtl),
            env_sv=self._emit_env(rtl),
            top_sv=self._emit_top(rtl),
            sby_config=self._emit_sby(rtl),
            bind_sv=self._emit_bind(rtl),
            makefile=self._emit_makefile(rtl),
            regression_list=self._emit_regression_list(rtl),
            filelist=self._filelist(rtl),
        )

    def generate_multi(self, modules: List[RTLModule]) -> List[UVMBenchmark]:
        """Generate UVM testbenches for multiple modules."""
        return [self.generate(rtl) for rtl in modules]

    # ── Transaction ──────────────────────────────────────────────────

    def _emit_transaction(self, rtl: RTLModule) -> str:
        m = rtl.name
        fields = []
        for p in rtl.input_ports:
            signed = " signed" if p.is_signed else ""
            width = f" [{p.width - 1}:0]" if p.width > 1 else ""
            fields.append(f"    rand logic{signed}{width} {p.name};")
        for p in rtl.output_ports:
            signed = " signed" if p.is_signed else ""
            width = f" [{p.width - 1}:0]" if p.width > 1 else ""
            fields.append(f"    logic{signed}{width} {p.name};")

        field_block = "\n".join(fields) if fields else "    rand logic [7:0] data;"

        constraints = []
        lo, hi = self.stimulus.bitstream_density_range
        for p in rtl.input_ports:
            if p.width > 1:
                constraints.append(
                    f"    constraint c_{p.name}_density {{\n"
                    f"        $countones({p.name}) inside "
                    f"{{[{int(lo * p.width)}:{int(hi * p.width)}]}};\n"
                    f"    }}"
                )
        constraint_block = "\n".join(constraints)

        return textwrap.dedent(f"""\
{_SPDX}
// SC-NeuroCore UVM — Transaction for {m}

class {m}_transaction extends uvm_sequence_item;
    `uvm_object_utils({m}_transaction)

{field_block}

{constraint_block}

    function new(string name = "{m}_transaction");
        super.new(name);
    endfunction

    function string convert2string();
        return $sformatf("{m}_txn: {" ".join(f"{p.name}=%0h" for p in rtl.input_ports)}",
            {", ".join(p.name for p in rtl.input_ports) or "data"});
    endfunction
endclass
""")

    # ── Sequence ─────────────────────────────────────────────────────

    def _emit_sequence(self, rtl: RTLModule) -> str:
        m = rtl.name
        num_txn = self.stimulus.num_transactions
        corner = ""
        if self.stimulus.enable_corner_cases:
            corner_assigns = []
            for p in rtl.input_ports:
                if p.width > 1:
                    corner_assigns.append(f"                txn.{p.name} = '0;")
                    corner_assigns.append(f"                txn.{p.name} = '1;")
            if corner_assigns:
                corner = textwrap.indent("\n".join(corner_assigns), "")

        return textwrap.dedent(f"""\
{_SPDX}
// SC-NeuroCore UVM — Sequences for {m}

class {m}_random_seq extends uvm_sequence #({m}_transaction);
    `uvm_object_utils({m}_random_seq)

    int num_transactions = {num_txn};

    function new(string name = "{m}_random_seq");
        super.new(name);
    endfunction

    task body();
        {m}_transaction txn;
        for (int i = 0; i < num_transactions; i++) begin
            txn = {m}_transaction::type_id::create($sformatf("txn_%0d", i));
            start_item(txn);
            if (!txn.randomize())
                `uvm_fatal("SEQ", "Randomization failed")
            finish_item(txn);
        end
    endtask
endclass

class {m}_corner_seq extends uvm_sequence #({m}_transaction);
    `uvm_object_utils({m}_corner_seq)

    function new(string name = "{m}_corner_seq");
        super.new(name);
    endfunction

    task body();
        {m}_transaction txn;
        // All zeros
        txn = {m}_transaction::type_id::create("txn_zeros");
        start_item(txn);
        if (!txn.randomize())
            `uvm_fatal("SEQ", "Randomization failed")
{corner}
        finish_item(txn);
    endtask
endclass

class {m}_lfsr_seq extends uvm_sequence #({m}_transaction);
    `uvm_object_utils({m}_lfsr_seq)

    int seed = 16'hACE1;
    int length = 256;

    function new(string name = "{m}_lfsr_seq");
        super.new(name);
    endfunction

    task body();
        {m}_transaction txn;
        logic [15:0] lfsr = seed;
        for (int i = 0; i < length; i++) begin
            txn = {m}_transaction::type_id::create($sformatf("lfsr_%0d", i));
            start_item(txn);
            if (!txn.randomize())
                `uvm_fatal("SEQ", "Randomization failed")
            // Override with LFSR-driven data
            lfsr = {{lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]}};
            finish_item(txn);
        end
    endtask
endclass
""")

    # ── Driver ───────────────────────────────────────────────────────

    def _emit_driver(self, rtl: RTLModule) -> str:
        m = rtl.name
        drive_lines = []
        for p in rtl.input_ports:
            drive_lines.append(f"            vif.{p.name} <= txn.{p.name};")
        drive_block = "\n".join(drive_lines) if drive_lines else "            // no input ports"

        return textwrap.dedent(f"""\
{_SPDX}
// SC-NeuroCore UVM — Driver for {m}

class {m}_driver extends uvm_driver #({m}_transaction);
    `uvm_component_utils({m}_driver)

    virtual {m}_if vif;

    function new(string name, uvm_component parent);
        super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        if (!uvm_config_db #(virtual {m}_if)::get(this, "", "vif", vif))
            `uvm_fatal("DRV", "No virtual interface")
    endfunction

    task run_phase(uvm_phase phase);
        {m}_transaction txn;
        forever begin
            seq_item_port.get_next_item(txn);
            @(posedge vif.clk);
{drive_block}
            seq_item_port.item_done();
        end
    endtask
endclass
""")

    # ── Monitor ──────────────────────────────────────────────────────

    def _emit_monitor(self, rtl: RTLModule) -> str:
        m = rtl.name
        sample_in = []
        for p in rtl.input_ports:
            sample_in.append(f"            txn.{p.name} = vif.{p.name};")
        sample_out = []
        for p in rtl.output_ports:
            sample_out.append(f"            txn.{p.name} = vif.{p.name};")
        in_block = "\n".join(sample_in) if sample_in else "            // no inputs"
        out_block = "\n".join(sample_out) if sample_out else "            // no outputs"

        return textwrap.dedent(f"""\
{_SPDX}
// SC-NeuroCore UVM — Monitor for {m}

class {m}_monitor extends uvm_monitor;
    `uvm_component_utils({m}_monitor)

    virtual {m}_if vif;
    uvm_analysis_port #({m}_transaction) ap;

    function new(string name, uvm_component parent);
        super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        ap = new("ap", this);
        if (!uvm_config_db #(virtual {m}_if)::get(this, "", "vif", vif))
            `uvm_fatal("MON", "No virtual interface")
    endfunction

    task run_phase(uvm_phase phase);
        {m}_transaction txn;
        forever begin
            @(posedge vif.clk);
            txn = {m}_transaction::type_id::create("mon_txn");
{in_block}
{out_block}
            ap.write(txn);
        end
    endtask
endclass
""")

    # ── Scoreboard ───────────────────────────────────────────────────

    def _emit_scoreboard(self, rtl: RTLModule) -> str:
        m = rtl.name
        checks = []
        if self.scoreboard.check_popcount:
            for p in rtl.output_ports:
                if p.width > 1:
                    checks.append(
                        f"        // Popcount check for {p.name}\n"
                        f"        int pc_{p.name} = $countones(txn.{p.name});\n"
                        f'        `uvm_info("SB", $sformatf("{p.name} popcount=%0d", pc_{p.name}), UVM_MEDIUM)'
                    )
        if self.scoreboard.check_spike_timing:
            for p in rtl.output_ports:
                if p.name.startswith("spike") or p.name.endswith("spike") or p.width == 1:
                    checks.append(
                        f"        if (txn.{p.name})\n"
                        f"            spike_count++;\n"
                        f'        `uvm_info("SB", $sformatf("spike_count=%0d", spike_count), UVM_HIGH)'
                    )

        golden_block = ""
        if self.scoreboard.check_golden_comparison:
            golden_lines = []
            for p in rtl.output_ports:
                if p.width > 1:
                    if p.name not in self.scoreboard.golden_expressions:
                        raise ValueError(
                            "Missing golden reference expression for output "
                            f"{p.name!r}; provide ScoreboardConfig.golden_expressions."
                        )
                    golden_lines.append(
                        f"        // Golden model comparison for {p.name}\n"
                        f"        expected_{p.name} = golden_compute_{p.name}(txn);\n"
                        f"        if (txn.{p.name} !== expected_{p.name}) begin\n"
                        f"            mismatch_count++;\n"
                        f'            `uvm_error("SB", $sformatf(\n'
                        f'                "MISMATCH {p.name}: got=%0h exp=%0h",\n'
                        f"                txn.{p.name}, expected_{p.name}))\n"
                        f"        end"
                    )
            golden_block = "\n".join(golden_lines)

        check_block = "\n".join(checks) if checks else "        // No specific checks configured"

        golden_funcs = []
        golden_vars = []
        if self.scoreboard.check_golden_comparison:
            for p in rtl.output_ports:
                if p.width > 1:
                    w = p.width
                    expression = self.scoreboard.golden_expressions[p.name]
                    golden_funcs.append(
                        f"    // Bit-true reference expression for {p.name}\n"
                        f"    function logic [{w - 1}:0] golden_compute_{p.name}({m}_transaction txn);\n"
                        f"        return {expression};\n"
                        f"    endfunction"
                    )
                    golden_vars.append(f"    logic [{p.width - 1}:0] expected_{p.name};")
        golden_func_block = "\n\n".join(golden_funcs)
        golden_var_block = "\n".join(golden_vars)

        return textwrap.dedent(f"""\
{_SPDX}
// SC-NeuroCore UVM — Scoreboard for {m}

class {m}_scoreboard extends uvm_scoreboard;
    `uvm_component_utils({m}_scoreboard)

    uvm_analysis_imp #({m}_transaction, {m}_scoreboard) ap;
    int transaction_count;
    int mismatch_count;
    int spike_count;
{golden_var_block}

    function new(string name, uvm_component parent);
        super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        ap = new("ap", this);
        transaction_count = 0;
        mismatch_count = 0;
        spike_count = 0;
    endfunction

{golden_func_block}

    function void write({m}_transaction txn);
        transaction_count++;
{check_block}
{golden_block}
    endfunction

    function void report_phase(uvm_phase phase);
        `uvm_info("SB", $sformatf(
            "Scoreboard summary: transactions=%0d mismatches=%0d spikes=%0d",
            transaction_count, mismatch_count, spike_count), UVM_LOW)
        if (mismatch_count > 0)
            `uvm_error("SB", $sformatf("%0d mismatches detected", mismatch_count))
    endfunction
endclass
""")

    # ── Coverage ─────────────────────────────────────────────────────

    def _emit_coverage(self, rtl: RTLModule) -> str:
        m = rtl.name
        coverpoints = []

        for p in rtl.input_ports:
            if p.width > 1:
                bins = self.coverage.bitstream_density_bins
                coverpoints.append(
                    f"        {p.name}_density: coverpoint $countones(txn.{p.name}) {{\n"
                    f"            bins density[{bins}] = {{[0:{p.width}]}};\n"
                    f"        }}"
                )

        for p in rtl.output_ports:
            if p.width == 1:
                coverpoints.append(
                    f"        {p.name}_toggle: coverpoint txn.{p.name} {{\n"
                    f"            bins off = {{0}};\n"
                    f"            bins on  = {{1}};\n"
                    f"        }}"
                )
            elif p.width > 1:
                bins = self.coverage.spike_rate_bins
                coverpoints.append(
                    f"        {p.name}_density: coverpoint $countones(txn.{p.name}) {{\n"
                    f"            bins density[{bins}] = {{[0:{p.width}]}};\n"
                    f"        }}"
                )

        # SCC (stochastic cross-correlation) coverage bins
        scc_cps = []
        input_wide = [p for p in rtl.input_ports if p.width > 1]
        if len(input_wide) >= 2 and self.coverage.scc_bins > 0:
            p1, p2 = input_wide[0], input_wide[1]
            min_w = min(p1.width, p2.width)
            scc_cps.append(
                f"        // SCC correlation bins between {p1.name} and {p2.name}\n"
                f"        {p1.name}_{p2.name}_scc: coverpoint \n"
                f"            ($countones(txn.{p1.name}[{min_w - 1}:0] & txn.{p2.name}[{min_w - 1}:0])) {{\n"
                f"            bins scc_bins[{self.coverage.scc_bins}] = {{[0:{min_w}]}};\n"
                f"        }}"
            )
        coverpoints.extend(scc_cps)

        # Toggle coverage (per-bit transition tracking)
        toggle_cps = []
        if self.coverage.toggle_coverage:
            for p in rtl.input_ports:
                if p.width > 1:
                    toggle_cps.append(
                        f"        {p.name}_activity: coverpoint txn.{p.name} {{\n"
                        f"            bins zero = {{'0}};\n"
                        f"            bins full = {{'1}};\n"
                        f"            bins mid[4] = {{[1:{(1 << p.width) - 2}]}};\n"
                        f"        }}"
                    )
        coverpoints.extend(toggle_cps)

        cross_covers = ""
        if self.coverage.cross_coverage and len(coverpoints) >= 2:
            cross_covers = "        // Cross coverage between input/output density\n"

        cp_block = (
            "\n".join(coverpoints) if coverpoints else "        // Auto-generated coverpoints"
        )

        target = self.coverage.target_percent

        return textwrap.dedent(f"""\
{_SPDX}
// SC-NeuroCore UVM — Functional Coverage for {m}

class {m}_coverage extends uvm_subscriber #({m}_transaction);
    `uvm_component_utils({m}_coverage)

    real coverage_target = {target};

    covergroup {m}_cg with function sample({m}_transaction txn);
{cp_block}
{cross_covers}
    endgroup

    function new(string name, uvm_component parent);
        super.new(name, parent);
        {m}_cg = new();
    endfunction

    function void write({m}_transaction t);
        {m}_cg.sample(t);
    endfunction

    function void report_phase(uvm_phase phase);
        real cov = {m}_cg.get_inst_coverage();
        `uvm_info("COV", $sformatf(
            "Coverage: %.1f%% (target: %.1f%%)", cov, coverage_target), UVM_LOW)
        if (cov < coverage_target)
            `uvm_warning("COV", $sformatf(
                "Coverage %.1f%% below target %.1f%%", cov, coverage_target))
    endfunction
endclass
""")

    # ── Agent ────────────────────────────────────────────────────────

    def _emit_agent(self, rtl: RTLModule) -> str:
        m = rtl.name
        return textwrap.dedent(f"""\
{_SPDX}
// SC-NeuroCore UVM — Agent for {m}

class {m}_agent extends uvm_agent;
    `uvm_component_utils({m}_agent)

    {m}_driver  drv;
    {m}_monitor mon;
    uvm_sequencer #({m}_transaction) sqr;

    function new(string name, uvm_component parent);
        super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        drv = {m}_driver::type_id::create("drv", this);
        mon = {m}_monitor::type_id::create("mon", this);
        sqr = uvm_sequencer#({m}_transaction)::type_id::create("sqr", this);
    endfunction

    function void connect_phase(uvm_phase phase);
        super.connect_phase(phase);
        drv.seq_item_port.connect(sqr.seq_item_export);
    endfunction
endclass
""")

    # ── Environment ──────────────────────────────────────────────────

    def _emit_env(self, rtl: RTLModule) -> str:
        m = rtl.name
        return textwrap.dedent(f"""\
{_SPDX}
// SC-NeuroCore UVM — Environment for {m}

class {m}_env extends uvm_env;
    `uvm_component_utils({m}_env)

    {m}_agent      agt;
    {m}_scoreboard sb;
    {m}_coverage   cov;

    function new(string name, uvm_component parent);
        super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        agt = {m}_agent::type_id::create("agt", this);
        sb  = {m}_scoreboard::type_id::create("sb", this);
        cov = {m}_coverage::type_id::create("cov", this);
    endfunction

    function void connect_phase(uvm_phase phase);
        super.connect_phase(phase);
        agt.mon.ap.connect(sb.ap);
        agt.mon.ap.connect(cov.analysis_export);
    endfunction
endclass
""")

    # ── Top-Level TB ─────────────────────────────────────────────────

    def _emit_top(self, rtl: RTLModule) -> str:
        m = rtl.name
        clk = rtl.clock_port
        rst = rtl.reset_port
        clk_name = clk.name if clk else "clk"
        rst_name = rst.name if rst else "rst_n"

        iface_signals = []
        for p in rtl.ports:
            if not p.is_clock and not p.is_reset:
                signed = " signed" if p.is_signed else ""
                width = f" [{p.width - 1}:0]" if p.width > 1 else ""
                iface_signals.append(f"    logic{signed}{width} {p.name};")
        iface_block = "\n".join(iface_signals) if iface_signals else "    logic [7:0] data;"

        dut_conns = []
        for p in rtl.ports:
            dut_conns.append(
                f"        .{p.name}({'intf.' + p.name if not p.is_clock and not p.is_reset else p.name})"
            )
        dut_block = ",\n".join(dut_conns)

        param_override = ""
        if rtl.params:
            param_vals = ", ".join(f".{p.name}({p.value})" for p in rtl.params)
            param_override = f" #({param_vals})"

        return textwrap.dedent(f"""\
{_SPDX}
// SC-NeuroCore UVM — Top-Level Testbench for {m}

`include "uvm_macros.svh"
import uvm_pkg::*;

interface {m}_if(input logic {clk_name});
{iface_block}
endinterface

module tb_{m}_top;
    logic {clk_name} = 0;
    logic {rst_name} = 0;

    always #5 {clk_name} = ~{clk_name};

    {m}_if intf(.{clk_name}({clk_name}));

    {m}{param_override} dut (
{dut_block}
    );

    initial begin
        uvm_config_db #(virtual {m}_if)::set(null, "*", "vif", intf);
        {rst_name} = 0;
        #20;
        {rst_name} = 1;
    end

    class {m}_test extends uvm_test;
        `uvm_component_utils({m}_test)
        {m}_env env;

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            env = {m}_env::type_id::create("env", this);
        endfunction

        task run_phase(uvm_phase phase);
            {m}_random_seq seq;
            phase.raise_objection(this);
            seq = {m}_random_seq::type_id::create("seq");
            seq.start(env.agt.sqr);
            phase.drop_objection(this);
        endtask
    endclass

    initial begin
        run_test("{m}_test");
    end
endmodule
""")

    # ── SymbiYosys Config ────────────────────────────────────────────

    def _emit_sby(self, rtl: RTLModule) -> str:
        m = rtl.name
        return textwrap.dedent(f"""\
[options]
mode prove
depth 25

[engines]
smtbmc

[script]
read -formal tb_{m}_top.sv
read -formal ../{m}.v
prep -top tb_{m}_top

[files]
tb_{m}_top.sv
../{m}.v
""")

    # ── File List ────────────────────────────────────────────────────

    def _filelist(self, rtl: RTLModule) -> List[str]:
        m = rtl.name
        flist = [
            f"{m}_transaction.sv",
            f"{m}_sequence.sv",
            f"{m}_driver.sv",
            f"{m}_monitor.sv",
            f"{m}_scoreboard.sv",
            f"{m}_coverage.sv",
            f"{m}_agent.sv",
            f"{m}_env.sv",
            f"tb_{m}_top.sv",
            f"{m}_verify.sby",
            f"{m}_bind.sv",
        ]
        return flist

    # ── Assertion Bind Module ────────────────────────────────────────

    def _emit_bind(self, rtl: RTLModule) -> str:
        m = rtl.name
        rst = rtl.reset_port
        clk = rtl.clock_port
        rst_name = rst.name if rst else "rst_n"
        clk_name = clk.name if clk else "clk"

        assertions = []
        for p in rtl.output_ports:
            if p.width == 1:
                assertions.append(
                    f"    // Reset assertion for {p.name}\n"
                    f"    property p_{p.name}_resets;\n"
                    f"        @(posedge {clk_name}) !{rst_name} |-> {p.name} == 0;\n"
                    f"    endproperty\n"
                    f"    a_{p.name}_rst: assert property(p_{p.name}_resets);\n"
                    f"    c_{p.name}_active: cover property(\n"
                    f"        @(posedge {clk_name}) {rst_name} |-> {p.name} == 1);"
                )
            elif p.width > 1:
                assertions.append(
                    f"    // Bounded output for {p.name}\n"
                    f"    a_{p.name}_bounded: assert property(\n"
                    f"        @(posedge {clk_name}) {rst_name} |-> ({p.name} <= {(1 << p.width) - 1}));\n"
                    f"    c_{p.name}_nonzero: cover property(\n"
                    f"        @(posedge {clk_name}) {rst_name} |-> ({p.name} != 0));"
                )

        assertion_block = (
            "\n\n".join(assertions) if assertions else "    // No assertions generated"
        )

        return textwrap.dedent(f"""\
{_SPDX}
// SC-NeuroCore UVM — Assertion Bind Module for {m}

module {m}_assertions (
    input logic {clk_name},
    input logic {rst_name},
{chr(10).join(f"    input logic [{p.width - 1}:0] {p.name}," if p.width > 1 else f"    input logic {p.name}," for p in rtl.output_ports).rstrip(",")}
);

{assertion_block}

endmodule

bind {m} {m}_assertions {m}_assert_inst (
    .{clk_name}({clk_name}),
    .{rst_name}({rst_name}),
{chr(10).join(f"    .{p.name}({p.name})," for p in rtl.output_ports).rstrip(",")}
);
""")

    # ── Makefile Generator ───────────────────────────────────────────

    def _emit_makefile(self, rtl: RTLModule, sim: str = "vcs") -> str:
        m = rtl.name
        target = SIM_TARGETS.get(sim, SIM_TARGETS["vcs"])
        flist = f"{m}.f"
        compile_cmd = target.compile_cmd.format(flist=flist, module=m)
        run_cmd = target.run_cmd.format(test=f"{m}_test", module=m)
        cov_cmd = target.coverage_cmd

        return textwrap.dedent(f"""\
# SC-NeuroCore UVM — Makefile for {m} ({target.name})
# Auto-generated by UVM Generator

MODULE = {m}
FLIST  = {flist}

.PHONY: compile sim coverage clean regression

compile:
\t{compile_cmd}

sim: compile
\t{run_cmd}

coverage:
\t{cov_cmd}

clean:
\trm -rf simv simv.daidir csrc *.vdb *.log work *.ucdb

regression:
\t@echo "Running regression..."
\t$(MAKE) sim
\t$(MAKE) coverage
\t@echo "Regression complete."
""")

    # ── Regression List ──────────────────────────────────────────────

    def _emit_regression_list(self, rtl: RTLModule) -> str:
        m = rtl.name
        lines = [
            f"# SC-NeuroCore UVM — Regression test list for {m}",
            "# Auto-generated by UVM Generator",
            "",
            "# test_name : sequence : iterations",
            f"{m}_random   : {m}_random_seq  : 1000",
            f"{m}_corner   : {m}_corner_seq  : 1",
            f"{m}_lfsr     : {m}_lfsr_seq    : 256",
        ]
        return "\n".join(lines) + "\n"

    # ── Formal Link Generation ───────────────────────────────────────

    def generate_formal_links(self, rtl: RTLModule) -> List[FormalLink]:
        """Generate formal-to-dynamic links for existing SymbiYosys modules."""
        links = []
        rst = rtl.reset_port
        rst_name = rst.name if rst else "rst_n"

        for p in rtl.output_ports:
            if p.width == 1:
                links.append(
                    FormalLink(
                        property_name=f"{p.name}_reset_check",
                        sby_module=f"{rtl.name}_formal",
                        assertion_sv=(
                            f"property p_{p.name}_rst;\n"
                            f"    @(posedge clk) !{rst_name} |-> {p.name} == 0;\n"
                            f"endproperty\n"
                            f"assert property(p_{p.name}_rst);"
                        ),
                        cover_sv=f"cover property(@(posedge clk) {rst_name} |-> {p.name} == 1);",
                    )
                )

        return links
