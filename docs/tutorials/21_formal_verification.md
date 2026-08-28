<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 21: Formal Verification with SymbiYosys

Formal verification proves that hardware properties hold for **all
possible inputs**, not just a finite test set. SC-NeuroCore currently ships
82 SymbiYosys proof jobs and 372 formal statements (252 assert, 88 assume,
32 cover) across the HDL formal tree.

**Prerequisites**: [SymbiYosys](https://symbiyosys.readthedocs.io/),
[Yosys](https://yosyshq.net/yosys/), an SMT solver (Z3 or Boolector)

```bash
# Install on Ubuntu/Debian
sudo apt install yosys symbiyosys z3
# Or via conda
conda install -c litex-hub yosys symbiyosys
```

## 1. What's formally verified

SC-NeuroCore proves properties through the SymbiYosys jobs under `hdl/formal/`.
The current inventory is:

| Inventory | Count |
|-----------|------:|
| SymbiYosys `.sby` proof jobs | 82 |
| `assert(...)` statements | 252 |
| `assume(...)` statements | 88 |
| `cover(...)` statements | 32 |
| Total formal statements | 372 |

**Total: 82 SymbiYosys proof jobs and 372 formal statements (252 assert, 88 assume, 32 cover).**

The larger proof-job set covers the original stochastic-computing RTL blocks
plus timing, masking, controller, queue, and sensor wrappers added after the
older seven-module tutorial inventory.

## 2. Running a single proof

Each module has a `.sby` task file and a `_formal.v` wrapper:

```bash
cd hdl/formal

# Prove all LIF neuron properties (depth 25 cycles)
sby -f sc_lif_neuron.sby

# Expected output:
# SBY  0:00:01  sc_lif_neuron  engine_0: Status returned by engine: pass
# SBY  0:00:01  sc_lif_neuron  summary: Elapsed clock time [H:MM:SS]: ...
# SBY  0:00:01  sc_lif_neuron  DONE (PASS, rc=0)
```

Generated catalogue jobs live one level below the legacy cores. For example,
the Medvedev job is deliberately Q16.16 because its calibrated
`d=2271.1927977404063` parameter cannot fit a signed Q8.8 word:

```bash
cd hdl/formal/catalogue
sby -f sc_medvedev_map.sby
```

The Ibarz-Tanaka job is also Q16.16: the source timescale `mu=0.001`
quantises to zero in Q8.8. Its depth-4 Z3 job is:

```bash
sby -f sc_ibarz_tanaka_rulkov_map.sby
```

The IQIF job uses signed Q32.0 because its enrolled contract is exact integer
arithmetic rather than a fractional fixed-point approximation. It bounds reset,
spike, and signed-state safety at depth 4:

```bash
sby -f sc_integerqifneuron.sby
```

The stateless McCulloch-Pitts job also uses signed Q32.0. Non-negative words
carry active excitatory-afferent counts and `-1` is the enrolled absolute-
inhibition sentinel; its depth-4 job proves reset/output safety without
inventing membrane state:

```bash
sby -f sc_mccullochpittsneuron.sby
```

## 3. Anatomy of a formal wrapper

The formal wrapper instantiates the DUT and adds `assert` / `cover`
statements inside `` `ifdef FORMAL `` guards:

```verilog
// hdl/formal/sc_lif_neuron_formal.v (simplified)
module sc_lif_neuron_formal(input wire clk, rst_n, ...);

    sc_lif_neuron #(.DATA_WIDTH(16), .FRACTION(8), ...) uut (...);

`ifdef FORMAL
    reg past_valid = 0;
    always @(posedge clk) past_valid <= 1;

    // Property 1: After reset, V equals V_REST
    always @(posedge clk)
        if (past_valid && !rst_n) assert(v_out == 16'sd0);

    // Property 2: Spike → output resets to V_RESET
    always @(posedge clk)
        if (past_valid && rst_n && spike_out) assert(v_out == 16'sd0);

    // Property 3: During refractory, no spike and V clamped
    always @(posedge clk)
        if (past_valid && rst_n && uut.refractory_counter > 0) begin
            assert(spike_out == 1'b0);
            assert(v_out == 16'sd0);
        end

    // Property 4: Refractory counter bounded
    always @(posedge clk)
        if (past_valid && rst_n) assert(uut.refractory_counter <= 3);

    // Cover: spike is reachable
    always @(posedge clk)
        if (past_valid && rst_n) cover(spike_out == 1'b1);
`endif
endmodule
```

## 4. The .sby task file

```ini
# hdl/formal/sc_lif_neuron.sby
[options]
mode prove          # bounded model checking
depth 25            # check up to 25 clock cycles

[engines]
smtbmc              # use SMT-based BMC (Z3/Boolector)

[script]
read -formal sc_lif_neuron_formal.v
read -formal sc_lif_neuron.v
prep -top sc_lif_neuron_formal

[files]
sc_lif_neuron_formal.v
../sc_lif_neuron.v
```

## 5. Running all proofs

```bash
cd hdl/formal
for sby_file in *.sby; do
    echo "=== Proving: $sby_file ==="
    sby -f "$sby_file"
done
```

Or use the CI workflow — `v3-engine.yml` runs formal proofs
automatically on every push to `hdl/`.

## 6. Writing your own formal properties

To add properties to an existing module:

1. Open `hdl/formal/<module>_formal.v`
2. Add assertions inside the `` `ifdef FORMAL `` block
3. Run `sby -f <module>.sby` to verify

To verify a new module:

1. Create `hdl/formal/my_module_formal.v` with DUT instantiation
2. Create `hdl/formal/my_module.sby` task file (copy an existing one)
3. Add `assert()` for invariants, `cover()` for reachability

**Property categories:**

| Type | Keyword | Purpose |
|------|---------|---------|
| Safety | `assert(cond)` | Must always hold |
| Liveness | `cover(cond)` | Must be reachable |
| Assume | `assume(cond)` | Constrain input space |

## 7. Interpreting failures

If a proof fails, SymbiYosys generates a counterexample trace:

```
SBY  0:00:02  sc_lif_neuron  engine_0: Status returned by engine: FAIL
SBY  0:00:02  sc_lif_neuron  writing trace to ...sc_lif_neuron/engine_0/trace.vcd
```

Open `trace.vcd` in GTKWave to see the exact cycle where the
assertion violated:

```bash
gtkwave hdl/formal/sc_lif_neuron/engine_0/trace.vcd
```

## 8. Formal vs simulation testing

| Aspect | Formal (SymbiYosys) | Simulation (Icarus/Verilator) |
|--------|-------------------|-------------------------------|
| Coverage | All inputs up to depth N | Only test vectors you write |
| Speed | Slower per proof | Faster per run |
| Bugs found | Corner cases, edge conditions | Functional correctness |
| Guarantee | Mathematical proof | Statistical confidence |

SC-NeuroCore uses both: formal proofs for safety-critical properties,
simulation testbenches (`hdl/tb_*.v`) for functional validation.

## Next steps

- [Tutorial 09: Hardware Co-simulation](09_hardware_cosimulation.md) — Python↔Verilog bit-exact checking
- [FPGA Deployment in 20 Minutes](fpga_in_20_minutes.md) — synthesis flow
- [Hardware Guide](../hardware/HARDWARE_GUIDE.md) — full hardware reference
