<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Python ↔ Verilog Co-simulation

Verify that the Python neuron model and the Verilog RTL produce
identical spike trains. This is the critical validation step before
FPGA deployment: any divergence means the hardware will not match
the software simulation.

**Prerequisites**: `pip install sc-neurocore`, Verilator installed
(see [FPGA Toolchain Guide](../hardware/FPGA_TOOLCHAIN_GUIDE.md))

## 1. The two models

SC-NeuroCore ships both a Python and a Verilog implementation of the
LIF neuron:

| Model | File | Arithmetic |
|-------|------|-----------|
| Python | `sc_neurocore.neurons.fixed_point_lif` | Signed Q8.8 fixed-point |
| Verilog | `hdl/sc_lif_neuron.v` | Signed Q8.8 fixed-point |

Both use identical bit-width (16), fraction (8), leak/gain
multipliers, and threshold comparison. The Python `FixedPointLIFNeuron`
is a **bit-true model** — every intermediate value is masked to 16-bit
signed range, exactly matching the Verilog.

## 2. Generate a test vector

Create a stimulus file with random currents, leak, and gain values:

```python
import numpy as np
from sc_neurocore import FixedPointLIFNeuron

np.random.seed(42)
N_STEPS = 200

# Random parameters within valid range
leak_k = 240   # close to 1.0 in Q8 (~0.94)
gain_k = 16    # ~0.0625 in Q8
currents = np.random.randint(-50, 100, size=N_STEPS)

# Run Python model
neuron = FixedPointLIFNeuron()
py_spikes = []
py_voltages = []
for I in currents:
    spike, v = neuron.step(leak_k=leak_k, gain_k=gain_k, I_t=int(I))
    py_spikes.append(spike)
    py_voltages.append(v)

print(f"Python: {sum(py_spikes)} spikes in {N_STEPS} steps")

# Write stimulus file for Verilator
with open("stimulus.txt", "w") as f:
    f.write(f"# leak_k={leak_k} gain_k={gain_k}\n")
    for I in currents:
        f.write(f"{int(I)}\n")
```

## 3. Verilator testbench

The testbench reads `stimulus.txt`, drives the Verilog `sc_lif_neuron`
module, and writes `rtl_output.txt`:

```cpp
// tb_cosim.cpp
#include <verilated.h>
#include "Vsc_lif_neuron.h"
#include <fstream>
#include <cstdint>

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* top = new Vsc_lif_neuron;

    std::ifstream stim("stimulus.txt");
    std::ofstream out("rtl_output.txt");
    std::string line;

    int leak_k = 240, gain_k = 16;
    top->clk = 0;
    top->rst_n = 0;
    top->ALPHA_LEAK = leak_k;
    top->GAIN_IN = gain_k;
    top->I_in = 0;
    top->noise_in = 0;

    // Reset for 2 cycles
    for (int i = 0; i < 4; i++) {
        top->clk = !top->clk;
        top->eval();
    }
    top->rst_n = 1;

    while (std::getline(stim, line)) {
        if (line[0] == '#') continue;
        int16_t I = std::stoi(line);
        top->I_in = I;

        // Posedge
        top->clk = 1;
        top->eval();
        out << (int)top->spike_out << " " << (int16_t)top->V_out << "\n";

        // Negedge
        top->clk = 0;
        top->eval();
    }

    delete top;
    return 0;
}
```

Build and run:

```bash
verilator -Wall --cc hdl/sc_lif_neuron.v --exe tb_cosim.cpp --build
./obj_dir/Vsc_lif_neuron
```

## 4. Compare outputs

```python
import numpy as np

# Read RTL output
rtl_spikes = []
rtl_voltages = []
with open("rtl_output.txt") as f:
    for line in f:
        parts = line.strip().split()
        rtl_spikes.append(int(parts[0]))
        rtl_voltages.append(int(parts[1]))

# Compare
py_spikes_arr = np.array(py_spikes)
rtl_spikes_arr = np.array(rtl_spikes[:len(py_spikes)])
py_voltages_arr = np.array(py_voltages)
rtl_voltages_arr = np.array(rtl_voltages[:len(py_voltages)])

spike_match = np.array_equal(py_spikes_arr, rtl_spikes_arr)
voltage_match = np.array_equal(py_voltages_arr, rtl_voltages_arr)

print(f"Spike trains match: {spike_match}")
print(f"Voltage traces match: {voltage_match}")

if not spike_match:
    diffs = np.where(py_spikes_arr != rtl_spikes_arr)[0]
    print(f"  First mismatch at step {diffs[0]}: "
          f"Python={py_spikes_arr[diffs[0]]}, RTL={rtl_spikes_arr[diffs[0]]}")

if not voltage_match:
    diffs = np.where(py_voltages_arr != rtl_voltages_arr)[0]
    print(f"  First voltage mismatch at step {diffs[0]}: "
          f"Python={py_voltages_arr[diffs[0]]}, RTL={rtl_voltages_arr[diffs[0]]}")

assert spike_match and voltage_match, "MISMATCH — do not deploy to hardware"
print("Co-simulation PASSED — Python and RTL are bit-exact")
```

## 5. Visualise the traces

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(py_voltages_arr, label="Python", linewidth=0.8)
axes[0].plot(rtl_voltages_arr, label="RTL", linewidth=0.8, linestyle="--")
axes[0].set_ylabel("Membrane voltage (Q8)")
axes[0].legend()

spike_times_py = np.where(py_spikes_arr == 1)[0]
spike_times_rtl = np.where(rtl_spikes_arr == 1)[0]
axes[1].eventplot([spike_times_py, spike_times_rtl],
                  lineoffsets=[1.0, 0.5], linelengths=0.3,
                  colors=["tab:blue", "tab:orange"])
axes[1].set_yticks([0.5, 1.0])
axes[1].set_yticklabels(["RTL", "Python"])
axes[1].set_xlabel("Time step")

plt.suptitle("Python ↔ Verilog Co-simulation")
plt.tight_layout()
plt.savefig("cosim_traces.png", dpi=150)
```

## 6. LFSR co-simulation

The bitstream encoder uses a 16-bit LFSR. Verify it matches too:

```python
from sc_neurocore import FixedPointLFSR, FixedPointBitstreamEncoder

lfsr_py = FixedPointLFSR(seed=0xACE1)
py_vals = [lfsr_py.step() for _ in range(100)]

# Compare against RTL LFSR output (from Verilator or logic analyser)
# rtl_vals = read_rtl_lfsr_output("lfsr_output.txt")
# assert py_vals == rtl_vals

enc_py = FixedPointBitstreamEncoder(seed_init=0xACE1)
bits = [enc_py.step(x_value=128) for _ in range(100)]
print(f"Encoder: {sum(bits)} ones in 100 steps (expected ~50)")
```

## 7. Automated test in CI

The test suite includes `tests/cosim/` tests that run when Verilator
is available. The CI workflow detects Verilator and enables co-sim:

```yaml
# .github/workflows/ci.yml (excerpt)
- name: Co-simulation tests
  if: steps.verilator.outputs.available == 'true'
  run: pytest tests/cosim/ -v
```

## What you learned

- `FixedPointLIFNeuron` is a bit-true Python model of the Verilog RTL
- Co-simulation compares spike trains and voltage traces cycle-by-cycle
- Any mismatch means the hardware will not reproduce the software results
- The LFSR and encoder also have Python bit-true models for co-sim
- Automate co-sim in CI to prevent regression

## Next steps

- Run co-simulation on the full `sc_dense_layer_top.v` (multi-neuron)
- Generate synthesis timing constraints from co-sim waveforms
- Export trained weights as Verilog `$readmemh` files for FPGA
