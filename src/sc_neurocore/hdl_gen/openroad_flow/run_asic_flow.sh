#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ASIC Synthesis Flow (Yosys + OpenROAD)
#
# Synthesizes the safety_monitor.sv module from gate-level netlist
# through to area/timing reports. Requires Yosys (synthesis) and
# optionally OpenROAD (place & route).
#
# Usage:
#   ./run_asic_flow.sh [--target safety_monitor|custom.sv]
#   ./run_asic_flow.sh --docker   # Use OpenROAD Docker image
#
# Output:
#   build/synth/   — Yosys synthesis results
#   build/reports/ — Area, timing, cell utilization reports

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
SYNTH_DIR="$BUILD_DIR/synth"
REPORT_DIR="$BUILD_DIR/reports"

# Default target
TARGET_SV="${1:-$REPO_ROOT/VISION2030/phase6_unification/neuro_safe_monitor/safety_monitor.sv}"
TOP_MODULE="neuro_safe_monitor"
CLOCK_PERIOD_NS=10  # 100 MHz target

# Technology library (use generic CMOS for open-source flow)
LIBERTY_FILE="$SCRIPT_DIR/tech/sky130_fd_sc_hd__tt_025C_1v80.lib"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[asic-flow] $*"; }
die() { echo "[asic-flow] ERROR: $*" >&2; exit 1; }

check_tools() {
    command -v yosys >/dev/null 2>&1 || die "yosys not found. Install: sudo apt install yosys"
    log "Yosys: $(yosys -V 2>&1 | head -1)"
}

mkdir_safe() {
    mkdir -p "$SYNTH_DIR" "$REPORT_DIR" "$SCRIPT_DIR/tech"
}

# ---------------------------------------------------------------------------
# Step 1: Generate minimal Liberty stub (if real PDK not available)
# ---------------------------------------------------------------------------
generate_liberty_stub() {
    if [ -f "$LIBERTY_FILE" ]; then
        log "Using existing Liberty: $LIBERTY_FILE"
        return
    fi
    log "Generating minimal Liberty stub for generic synthesis..."
    cat > "$LIBERTY_FILE" << 'LIBERTY_EOF'
/* Minimal Liberty for Yosys generic synthesis */
library(generic_cmos) {
  delay_model : table_lookup;
  time_unit : "1ns";
  voltage_unit : "1V";
  current_unit : "1mA";
  capacitive_load_unit(1, pf);

  cell(BUF_X1) {
    area : 1.0;
    pin(A) { direction: input; capacitance: 0.01; }
    pin(Y) { direction: output; function: "A";
      timing() { related_pin: "A"; cell_rise(scalar) { values("0.1"); }
        cell_fall(scalar) { values("0.1"); }
        rise_transition(scalar) { values("0.05"); }
        fall_transition(scalar) { values("0.05"); } } }
  }
  cell(INV_X1) {
    area : 0.8;
    pin(A) { direction: input; capacitance: 0.01; }
    pin(Y) { direction: output; function: "!A";
      timing() { related_pin: "A"; cell_rise(scalar) { values("0.08"); }
        cell_fall(scalar) { values("0.08"); }
        rise_transition(scalar) { values("0.04"); }
        fall_transition(scalar) { values("0.04"); } } }
  }
  cell(AND2_X1) {
    area : 1.5;
    pin(A) { direction: input; capacitance: 0.01; }
    pin(B) { direction: input; capacitance: 0.01; }
    pin(Y) { direction: output; function: "(A B)";
      timing() { related_pin: "A B"; cell_rise(scalar) { values("0.15"); }
        cell_fall(scalar) { values("0.15"); }
        rise_transition(scalar) { values("0.06"); }
        fall_transition(scalar) { values("0.06"); } } }
  }
  cell(OR2_X1) {
    area : 1.5;
    pin(A) { direction: input; capacitance: 0.01; }
    pin(B) { direction: input; capacitance: 0.01; }
    pin(Y) { direction: output; function: "(A+B)";
      timing() { related_pin: "A B"; cell_rise(scalar) { values("0.15"); }
        cell_fall(scalar) { values("0.15"); }
        rise_transition(scalar) { values("0.06"); }
        fall_transition(scalar) { values("0.06"); } } }
  }
  cell(DFF_X1) {
    area : 4.0;
    ff(IQ, IQN) { clocked_on: "CK"; next_state: "D"; }
    pin(CK) { direction: input; clock: true; capacitance: 0.02; }
    pin(D)  { direction: input; capacitance: 0.01;
      timing() { related_pin: "CK"; timing_type: setup_rising;
        rise_constraint(scalar) { values("0.1"); }
        fall_constraint(scalar) { values("0.1"); } } }
    pin(Q)  { direction: output; function: "IQ";
      timing() { related_pin: "CK"; timing_type: rising_edge;
        cell_rise(scalar) { values("0.2"); }
        cell_fall(scalar) { values("0.2"); }
        rise_transition(scalar) { values("0.08"); }
        fall_transition(scalar) { values("0.08"); } } }
  }
}
LIBERTY_EOF
    log "Liberty stub written to $LIBERTY_FILE"
}

# ---------------------------------------------------------------------------
# Step 2: Yosys Synthesis
# ---------------------------------------------------------------------------
run_synthesis() {
    log "Running Yosys synthesis: $TARGET_SV → $TOP_MODULE"

    YOSYS_SCRIPT="$SYNTH_DIR/synth.ys"
    cat > "$YOSYS_SCRIPT" << YOSYS_EOF
# SC-NeuroCore ASIC Synthesis Script
# Target: $TOP_MODULE @ ${CLOCK_PERIOD_NS}ns clock period

# Read design
read_verilog -sv $TARGET_SV

# Elaborate
hierarchy -top $TOP_MODULE

# Technology-independent optimization
proc
opt
fsm
opt
memory
opt
techmap
opt

# Map to generic cells
abc -liberty $LIBERTY_FILE

# Clean up
opt_clean
clean

# Reports
stat -liberty $LIBERTY_FILE
tee -o $REPORT_DIR/cell_stats.txt stat -liberty $LIBERTY_FILE

# Write outputs
write_verilog $SYNTH_DIR/${TOP_MODULE}_synth.v
write_json    $SYNTH_DIR/${TOP_MODULE}_synth.json

YOSYS_EOF

    yosys -s "$YOSYS_SCRIPT" > "$REPORT_DIR/yosys.log" 2>&1
    YOSYS_EXIT=$?

    if [ $YOSYS_EXIT -ne 0 ]; then
        log "Yosys synthesis FAILED (exit=$YOSYS_EXIT)"
        tail -20 "$REPORT_DIR/yosys.log"
        return 1
    fi

    log "Synthesis completed successfully"

    # Extract key metrics from log
    log "--- Synthesis Report ---"
    grep -E "Number of cells|Number of wires|Chip area|ABC:" "$REPORT_DIR/yosys.log" | tail -10
    log "Full report: $REPORT_DIR/cell_stats.txt"
    log "Gate netlist: $SYNTH_DIR/${TOP_MODULE}_synth.v"
}

# ---------------------------------------------------------------------------
# Step 3: Generate SDC timing constraints
# ---------------------------------------------------------------------------
generate_sdc() {
    SDC_FILE="$SYNTH_DIR/${TOP_MODULE}.sdc"
    cat > "$SDC_FILE" << SDC_EOF
# SC-NeuroCore Timing Constraints
# Target: $TOP_MODULE @ ${CLOCK_PERIOD_NS}ns ($(echo "1000/$CLOCK_PERIOD_NS" | bc) MHz)

create_clock -name clk -period $CLOCK_PERIOD_NS [get_ports clk]
set_input_delay  -clock clk -max [expr {$CLOCK_PERIOD_NS * 0.2}] [all_inputs]
set_output_delay -clock clk -max [expr {$CLOCK_PERIOD_NS * 0.2}] [all_outputs]
SDC_EOF
    log "SDC constraints written to $SDC_FILE"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    log "SC-NeuroCore ASIC Synthesis Flow"
    log "Target: $TARGET_SV"
    log "Module: $TOP_MODULE"
    log "Clock:  ${CLOCK_PERIOD_NS}ns ($(echo "1000/$CLOCK_PERIOD_NS" | bc) MHz)"
    echo "=========================================="

    check_tools
    mkdir_safe
    generate_liberty_stub
    generate_sdc
    run_synthesis

    echo "=========================================="
    log "Flow complete. Outputs in $BUILD_DIR/"
    ls -lh "$SYNTH_DIR/${TOP_MODULE}_synth.v" 2>/dev/null
    ls -lh "$REPORT_DIR/cell_stats.txt" 2>/dev/null
}

main "$@"
