# Optics — Photonic Stochastic Computing

End-to-end photonic stack: truly-random bitstream generation via laser
interference, compilation of SC IR onto Mach-Zehnder cascades, FDTD
co-simulation (1D absorbing boundary + 2D split-field Berenger PML),
coupled-mode crosstalk analysis for parallel waveguide banks, and GDSII
export via ``gdsfactory``.

Install the Rust acceleration + layout tooling with

```
pip install "sc-neurocore[optics]"
```

The Rust path (``libsc_neurocore_engine``) exposes parallel Rayon kernels
for crosstalk analysis and waveguide routing; a pure-Python fallback
mirrors the same math when the engine wheel is absent.

---

## 1. PhotonicBitstreamLayer — laser-interference bitstream source

Physical randomness instead of LFSR pseudo-randomness — two coherent laser
beams with phase noise ``φ`` produce interference intensity

```
I = I₁ + I₂ + 2√(I₁ I₂) cos(φ)
```

Normalised: ``I_norm = 0.5 + 0.5 · cos(φ)`` with ``φ ~ Uniform(0, 2π)``.
A bit is ``1`` if the intensity falls below the per-channel input
probability, matching a photodetector + comparator in hardware.

```python
from sc_neurocore.optics.photonic_layer import PhotonicBitstreamLayer
import numpy as np

layer = PhotonicBitstreamLayer(n_channels=4)
probs = np.array([0.2, 0.4, 0.6, 0.8])
bits = layer.forward(probs, length=10_000)
print(bits.mean(axis=1))  # ≈ [0.2, 0.4, 0.6, 0.8]
```

---

## 2. BitstreamToOptical + PhotonicCompiler

``BitstreamToOptical`` encodes an SC bitstream onto an optical carrier —
amplitude, phase, or IQ modulation — keyed by
:class:`sc_neurocore.optics.photonic_emitter.OpticalModulation`.

``PhotonicCompiler`` is the full SC-to-silicon-photonics pipeline. It
takes a bitstream, maps it onto an MZI cascade (one modulator per
quantised phase step) for a chosen :class:`PhotonicTarget`, and returns
a :class:`CompilationResult` with:

| Field                   | Meaning                                               |
| ----------------------- | ----------------------------------------------------- |
| ``target``              | Target PDK name (e.g. ``silicon_photonics``)          |
| ``num_modulators``      | MZI cells required by the compiled bitstream          |
| ``optical_power_mean_mw`` | Mean optical power across the cascade              |
| ``phase_coverage_rad``  | Total phase the cascade is configured to cover        |
| ``netlist``             | Textual netlist (Verilog-style) of the MZI cascade    |
| ``fdtd_energy``         | Optional scalar from a co-run FDTD sanity simulation  |

```python
from sc_neurocore.optics.photonic_emitter import PhotonicCompiler

compiler = PhotonicCompiler()
result = compiler.compile_bitstream(bitstream)
print(result.netlist)
```

### 2.1 GDSII export

``CompilationResult.to_gdsii(filename, mzi_length_um=10.0, pitch_um=100.0)``
writes a physical GDSII layout of the compiled MZI cascade via
``gdsfactory``. Each compiled MZI maps to a ``gf.components.mzi`` cell,
placed at uniform pitch along x. An SC-NeuroCore header label and the
compiled netlist string are written to the GDS TEXT layer (63/0) so the
layout file carries the logical build alongside the physical geometry.

An empty layout (``num_modulators == 0``) is rejected with
``NotImplementedError`` — silent no-op export would mask a compiler
misconfiguration.

Returns a summary dict: ``{filename, n_modulators, mzi_length_um,
pitch_um, total_length_um, target}`` — useful for verification and
regression tests.

```python
info = result.to_gdsii("cascade.gds", mzi_length_um=12.5, pitch_um=80.0)
print(info["total_length_um"])  # len(cells) × pitch_um
```

---

## 3. FDTD co-simulation

Two solvers cover 1D pulse propagation and 2D cross-section crosstalk
studies on a Yee grid.

### 3.1 FDTDSolver — 1D

Classical 1D Yee leapfrog with a **quadratic-ramp multiplicative absorbing
boundary** at each end — not a Berenger split-field PML: 1D does not
require the σ-matched split formulation because there is no transverse
dimension into which energy could scatter. The default taper reaches
``−30 dB`` reflection for wavelengths much smaller than the boundary
depth.

```python
from sc_neurocore.optics.photonic_emitter import FDTDSolver

sim = FDTDSolver(grid_size=512, dx_um=0.02, refractive_index=3.48, boundary_cells=20)
sim.inject_pulse(position=256, wavelength_nm=1550.0, amplitude=1.0)
sim.step(400)
print(sim.field_energy())
```

### 3.2 FDTD2DSolver — 2D split-field Berenger PML

TE-mode 2D Yee grid with a full Berenger split-field PML at all four
edges. ``Ez`` is split into ``Ezx + Ezy`` with independent per-cell
conductivities ``σ_x(x)``, ``σ_y(y)``; the magnetic conductivity
``σ*_x = σ_x · μ₀/ε₀`` enforces the matched-impedance condition that
suppresses reflection. Conductivity grades cubically from 0 (interior)
to ``σ_max`` (outermost PML cell).

Material is set cell-by-cell on ``n_map``; :meth:`set_waveguide` places a
horizontal refractive-index stripe.

```python
from sc_neurocore.optics.photonic_emitter import FDTD2DSolver

sim = FDTD2DSolver(nx=200, ny=100, pml_layers=12)
sim.set_waveguide(y_center=50, width_cells=10, refractive_index=3.48)
sim.inject_source(x=50, y=50, wavelength_nm=1550.0, amplitude=1.0, sigma_cells=8)
sim.step(500)
ez_cs = sim.cross_section(x=150)
```

References:

- Yee, *IEEE Trans. Antennas Propag.* 14(3):302-307, 1966.
- Berenger, *J. Comp. Phys.* 114(2):185-200, 1994.

### 3.3 MeepAdapter — drop-in to a full-featured FDTD engine

``MeepAdapter`` is an optional wrapper that translates the
``FDTD2DSolver`` geometry/source setup into a ``meep.Simulation`` when
the ``pymeep`` package is installed, for higher-order accuracy or
dispersive material studies.

---

## 4. Crosstalk — coupled-mode theory

``CrosstalkModel`` + :class:`WaveguidePair` model evanescent crosstalk
between parallel waveguides using coupled-mode theory with a Marcatili
transverse-decay form.

For waveguides with core index ``n_c`` and cladding index ``n_s`` at
wavelength ``λ`` (all units consistent) the mode-overlap decay length is

```
L_decay = λ / (2π √(n_c² − n_s²))
```

The even/odd effective-index split at gap ``g`` follows the standard
empirical exponential,

```
Δn_eff(g) = 0.1 · exp(−g / L_decay)
```

The coupling coefficient and the power coupling ratio after a uniform
coupler of length ``L`` are

```
κ(g)   = π · Δn_eff(g) / λ
ratio  = sin²(κ · L)
iso_dB = −10 log₁₀(ratio)
```

### 4.1 analyze_bank — uniform parallel bank

``CrosstalkModel.analyze_bank(waveguides, gap_nm, coupling_length_um)``
reports the full statistics of a uniform bank of ``N`` parallel
waveguides — both adjacent pairs (gap = ``gap_nm``) and the largest
secondary term (next-nearest pairs at gap = ``2·gap_nm``). All further
pairs decay at least as ``exp(−2·g/L_decay)`` smaller and are dropped.

The ``crosstalk_safe`` flag is ``True`` when the worst-case isolation
exceeds 20 dB — the standard industry threshold for logical independence.

### 4.2 analyze_pairs — arbitrary geometry, O(N²)

When the bank isn't uniform, pass pair-wise geometry explicitly:
``analyze_pairs(pair_indices, gaps_nm, coupling_lengths_um)``. Each pair
is evaluated in parallel via Rayon when the Rust engine is present; the
Python fallback reproduces the same math serially.

```python
from sc_neurocore.optics.photonic_emitter import CrosstalkModel

model = CrosstalkModel()
bank = model.analyze_bank(waveguides=8, gap_nm=250.0, coupling_length_um=10.0)
print(bank["worst_isolation_db"], bank["crosstalk_safe"])

pairs = model.analyze_pairs(
    pair_indices=[(0, 1), (1, 2), (0, 2)],
    gaps_nm=[200.0, 400.0, 800.0],
    coupling_lengths_um=[10.0, 10.0, 10.0],
)
print(pairs["isolation_db"])  # sorted ascending with gap
```

References:

- Marcatili, *Bell Syst. Tech. J.* 48(7):2071-2102, 1969.
- Okamoto, *Fundamentals of Optical Waveguides*, 2006, ch. 4.

---

## 5. Rust acceleration path

The Rust engine (``libsc_neurocore_engine``) exposes four photonic FFI
entry points re-exported from ``sc_neurocore_engine``:

| Function                             | Purpose                                          |
| ------------------------------------ | ------------------------------------------------ |
| ``py_ph_route_waveguides``           | Mesh routing with Manhattan distance + crossings |
| ``py_ph_analyze_power_budget``       | Laser → detector path loss budget (all paths)    |
| ``py_ph_analyze_crosstalk_bank``     | Uniform bank crosstalk (closed-form)             |
| ``py_ph_analyze_crosstalk_pairs``    | Per-pair crosstalk, parallel over pairs          |

Tests ``tests/test_optics/test_crosstalk.py::TestBackendParity`` enforce
bit-for-bit agreement between the Rust and Python paths to within 1e-9.

---

## Reference

- Source: ``src/sc_neurocore/optics/photonic_emitter.py``,
  ``src/sc_neurocore/optics/photonic_layer.py``.
- Tests: ``tests/test_optics/``.
- Rust engine: ``engine/src/photonic.rs``.
- Demo: ``examples/15_photonic_compilation_demo.py``.

::: sc_neurocore.optics.photonic_layer
    options:
      show_root_heading: true

::: sc_neurocore.optics.photonic_emitter
    options:
      show_root_heading: true
