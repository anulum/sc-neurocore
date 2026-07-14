# Optics — Photonic Stochastic Computing

End-to-end photonic stack for SC-NeuroCore: truly-random bitstream
generation via laser interference, compilation of SC IR onto
Mach-Zehnder cascades, FDTD co-simulation (1D absorbing boundary + 2D
split-field Berenger PML), coupled-mode crosstalk analysis for
parallel waveguide banks, and GDSII export via ``gdsfactory``. The stable
``photonic_emitter`` import path is a definition-free compatibility facade;
seven bounded modules own types, conversion, FDTD, compilation, Meep,
crosstalk, and emitter responsibilities.

Install the Rust acceleration + layout tooling with:

```
pip install "sc-neurocore[optics]"
```

The Rust engine (``libsc_neurocore_engine``) exposes parallel Rayon
kernels for crosstalk analysis; a validated pure-Python fallback uses the
same closed-form model when the engine wheel is absent. Standalone Rust, Go,
Julia, and Mojo implementations mirror only that numeric crosstalk contract.
FDTD, Meep, compilation, netlist emission, and filesystem work remain owned by
Python and are not presented as cross-language capabilities.

---

## 1. Mathematical formalism

### 1.1 Photonic bitstream generation

Two coherent laser beams $I_1, I_2$ with independent phase noise
$\varphi \sim \mathrm{Uniform}(0, 2\pi)$ interfere. Intensity:

$$
I(\varphi) = I_1 + I_2 + 2\sqrt{I_1 I_2}\,\cos\varphi.
$$

With balanced beams $I_1 = I_2$:

$$
I_\text{norm}(\varphi) = \tfrac{1}{2}\!\left(1 + \cos\varphi\right).
$$

A bit is emitted as $1$ if $I_\text{norm} < p$ (the input probability),
else $0$. Since $\cos\varphi$ is uniformly distributed,
$I_\text{norm}$ is distributed as $U = (1 + \cos\varphi)/2$; its
CDF at $p$ is exactly $p$, so $\Pr[\text{bit} = 1] = p$ — the
photodetector + comparator pair produces a bitstream with mean
activity equal to the input probability.

### 1.2 Coupled-mode theory for parallel waveguides

Two parallel single-mode waveguides separated by gap $g$ exchange
power via evanescent coupling. Let $n_c, n_s$ be the core and cladding
refractive indices at wavelength $\lambda$ (all units consistent).
The **transverse decay length** of the evanescent field follows the
Marcatili approximation:

$$
L_\text{decay}(\lambda, n_c, n_s)
= \frac{\lambda}{2\pi\sqrt{n_c^2 - n_s^2}}.
$$

The **effective-index difference** between the even and odd supermodes
decays exponentially with the gap:

$$
\Delta n_\text{eff}(g) = 0.1\, e^{-g / L_\text{decay}}.
$$

The prefactor $0.1$ is an empirical normalisation matched to Si
waveguides at 1550 nm in SiO₂ cladding and bears the units of an
unscaled index split; it is consistent with the tabulated values in
Okamoto (2006, Ch. 4).

The **coupling coefficient** per unit length is:

$$
\kappa(g) = \frac{\pi\,\Delta n_\text{eff}(g)}{\lambda}
\qquad \text{(units: } \mu\text{m}^{-1}\text{ when }\lambda\text{ in }\mu\text{m)}.
$$

The **power coupling ratio** after a uniform directional coupler of
length $L$:

$$
\eta(g, L) = \sin^2\!\big(\kappa(g)\, L\big).
$$

The **isolation** in dB between the adjacent ports:

$$
\mathrm{iso}_\text{dB}(g, L) = -10\,\log_{10}\eta(g, L).
$$

### 1.3 Transfer matrix for a single coupler

The $2\times 2$ unitary transfer matrix for two adjacent waveguides
coupled over length $L$ (power-preserving):

$$
T(g, L) = \begin{pmatrix}
\cos(\kappa L) & j\sin(\kappa L) \\
j\sin(\kappa L) & \cos(\kappa L)
\end{pmatrix}.
$$

Output field amplitudes: $\mathbf{a}_\text{out} = T(g, L)\,\mathbf{a}_\text{in}$.
Because $T T^\dagger = I$ by construction, $|a_\text{out,1}|^2 +
|a_\text{out,2}|^2 = |a_\text{in,1}|^2 + |a_\text{in,2}|^2$ — power is
exactly conserved per coupler, as required.

### 1.4 FDTD: 1D Yee discretisation

Maxwell's equations in a source-free, non-magnetic medium reduce in 1D
to two coupled PDEs:

$$
\frac{\partial E_z}{\partial t}
= -\frac{1}{\varepsilon}\frac{\partial H_y}{\partial x},
\qquad
\frac{\partial H_y}{\partial t}
= -\frac{1}{\mu_0}\frac{\partial E_z}{\partial x}.
$$

On a staggered Yee grid with $E_z$ sampled at integer cells and $H_y$
at half-integer cells, the leap-frog update is:

$$
\begin{aligned}
H_y^{n+1/2}[i+\tfrac{1}{2}] &=
H_y^{n-1/2}[i+\tfrac{1}{2}]
- \frac{\Delta t}{\mu_0 \Delta x}\big(E_z^{n}[i+1] - E_z^{n}[i]\big), \\
E_z^{n+1}[i] &=
E_z^{n}[i]
- \frac{\Delta t}{\varepsilon_i \Delta x}
\big(H_y^{n+1/2}[i+\tfrac{1}{2}] - H_y^{n+1/2}[i-\tfrac{1}{2}]\big).
\end{aligned}
$$

SC-NeuroCore's 1D solver adds a **multiplicative absorbing boundary**
at each end — a quadratic-ramp taper $s_i = 1 - 0.8\,((N-i)/N)^2$ over
the outermost $N$ cells:

$$
E_z[i] \leftarrow s_i\, E_z[i], \qquad H_y[i] \leftarrow s_i\, H_y[i],
\qquad 0 \le i < N \text{ or } N_x - N \le i < N_x.
$$

This is **not** a split-field Berenger PML. 1D has no transverse
dimension into which energy could scatter, so the σ-matched split
formulation is neither required nor implemented. Reflection is
< −30 dB for wavelengths much smaller than the boundary depth.

### 1.5 FDTD: 2D split-field Berenger PML

The 2D TE-mode solver uses the full split-field formulation of
Berenger (1994). The $E_z$ field is split into $E_{zx}$ + $E_{zy}$;
each component carries its own electric conductivity $\sigma_x(x),
\sigma_y(y)$, matched by magnetic conductivities

$$
\sigma^*_x = \sigma_x \cdot \frac{\mu_0}{\varepsilon_0},
\qquad
\sigma^*_y = \sigma_y \cdot \frac{\mu_0}{\varepsilon_0}.
$$

The matched-impedance condition guarantees that the wave enters the
PML without reflection. The split-field updates, using
Taflove-Hagness discretisation:

$$
\begin{aligned}
E_{zx}^{n+1}[i,j] &=
c^a_x[i,j]\, E_{zx}^n[i,j]
+ c^b_x[i,j]\, \big(H_y^{n+1/2}[i,j] - H_y^{n+1/2}[i-1,j]\big), \\
E_{zy}^{n+1}[i,j] &=
c^a_y[i,j]\, E_{zy}^n[i,j]
- c^b_y[i,j]\, \big(H_x^{n+1/2}[i,j] - H_x^{n+1/2}[i,j-1]\big), \\
E_z^{n+1}[i,j] &= E_{zx}^{n+1}[i,j] + E_{zy}^{n+1}[i,j],
\end{aligned}
$$

with coefficient arrays

$$
c^a_x[i,j] = \frac{\varepsilon_{i,j} - \sigma_x[i,j]\,\Delta t/2}
                   {\varepsilon_{i,j} + \sigma_x[i,j]\,\Delta t/2},
\qquad
c^b_x[i,j] = \frac{\Delta t}
  {(\varepsilon_{i,j} + \sigma_x[i,j]\,\Delta t/2)\,\Delta x},
$$

and analogously for $c^a_y, c^b_y$. The conductivity profile inside
the PML uses a cubic ramp,
$\sigma(i) = \sigma_\text{max}\,((P-i)/P)^3$, over $P$ PML layers.
$\sigma_\text{max} = 5 / (120\pi\,\Delta x)$ is the standard
Taflove-Hagness recommendation for < −60 dB reflection.

### 1.6 CFL stability

The 2D CFL condition used by the solver:

$$
\Delta t \le \frac{\mathrm{CFL}}{c_0 \sqrt{2}}\cdot \min(\Delta x, \Delta y),
\qquad \mathrm{CFL} = 0.5 \text{ by default}.
$$

The 1D solver uses $\Delta t = \mathrm{CFL}\cdot\Delta x / c_0$ — same
condition collapsed to 1D. Both are verified by the
``test_step_preserves_finiteness_under_cfl`` test group.

---

## 2. Theoretical context

Photonic stochastic computing replaces the standard LFSR bitstream
source with an **optical** one — two interfering coherent beams whose
phase noise is genuinely quantum-mechanical, so the produced bits are
Bernoulli-distributed with no pseudo-random correlations at any lag
(Abhari et al. 2019). For SC neural networks this removes the PCC
(pseudo-random cross-correlation) penalty that limits deep LFSR-driven
SC networks to ~8 bits of effective precision.

The **photonic compiler** translates SC bitstreams onto Mach-Zehnder
interferometers (MZIs). Each MZI implements a tunable 2×2 unitary;
cascading them realises any small unitary mesh. The compiler's job is
to pick the right phase for each MZI given a target output bitstream.
This is the photonic analogue of the Lightmatter / Lightelligence
silicon-photonic accelerators (Shen et al. 2017) but runs in
bitstream domain instead of analogue amplitude.

The **FDTD** solvers verify that the compiled layout actually
propagates the bitstream correctly through the physical waveguide
geometry — Yee's algorithm (Yee 1966) with Berenger's PML (Berenger
1994) for open-boundary simulation. These are decades-old,
implementation-settled numerical methods; the value SC-NeuroCore adds
is the tight integration with the SC compiler: layout → FDTD →
bitstream metric (popcount, SCC) in a single Python call.

The **crosstalk analysis** bounds the accuracy of the compiled layout.
Coupled-mode theory (Marcatili 1969, Okamoto 2006 ch. 4) gives the
closed-form coupling between adjacent parallel waveguides; the
Rust-accelerated kernel evaluates every waveguide pair in a bank in
parallel via Rayon (see §7 for the measured speedup). The analysis
outputs an isolation budget in dB per pair plus the aggregate
safety flag — the same abstraction the silicon-photonic design tools
use for pre-layout routing decisions.

Finally, **GDSII export** bridges from the compiled netlist to a real
foundry tape-out. SC-NeuroCore writes standard KLayout-compatible GDS
files via ``gdsfactory``, so downstream tools (KLayout, OpenROAD for
photonics, Luceda IPKISS) can consume the layout unchanged.

---

## 3. Pipeline position

The optics subsystem sits downstream of the SC IR compiler and
upstream of the physical tooling (FDTD sanity, GDSII tape-out).

```
SC bitstream (from neuron / compiler)
       │
       ▼
 BitstreamToOptical  ───► OpticalPulse stream (phase or amplitude)
       │
       ▼
 PhotonicCompiler  ──────► CompilationResult
       │                       │
       │                       ├──► num_modulators
       │                       ├──► phase_coverage_rad
       │                       ├──► optical_power_mean_mw
       │                       ├──► netlist (Verilog-style text)
       │                       └──► fdtd_energy (optional co-sim)
       │
       ├─► FDTDSolver / FDTD2DSolver   (energy / dispersion sanity)
       │
       ├─► CrosstalkModel.analyze_bank (isolation budget)
       │       │
       │       └── Rust FFI: py_ph_analyze_crosstalk_bank / _pairs
       │
       └─► CompilationResult.to_gdsii ──► *.gds file (KLayout / gdsfactory)
```

**Inputs** — an SC bitstream (``np.ndarray`` of booleans) plus a
:class:`PhotonicTarget` (PDK, wavelength, modulator type, Q factor).

**Outputs** — a :class:`CompilationResult` packet (netlist + metrics),
an FDTD energy trace, crosstalk isolation bounds, a physical GDSII
file. None of the stages is mandatory: most users stop at
``CompilationResult`` and feed the netlist to external EDA.

---

## 4. Features

| Feature                                   | Detail                                                                              |
| ----------------------------------------- | ----------------------------------------------------------------------------------- |
| ``PhotonicBitstreamLayer``                | Laser-interference bitstream source, N channels                                      |
| Three modulation modes                    | ``PHASE``, ``AMPLITUDE``, ``HYBRID`` (PHASE+AMP)                                     |
| Three built-in PhotonicTargets            | ``lightmatter`` (1550 nm MZI), ``silicon_photonics`` (1310 nm microring), ``two_d_waveguide`` (850 nm) |
| ``BitstreamToOptical``                    | Dense SC bitstream → list of ``OpticalPulse``; vector API (phase / amp / power arrays) |
| ``PhotonicCompiler.compile_bitstream``    | SC IR → MZI cascade netlist; optional run-through FDTD                              |
| ``CompilationResult.to_gdsii``            | Real GDSII file via gdsfactory + KLayout; PDK auto-activation; header + netlist labels |
| ``FDTDSolver`` (1D)                       | Yee leap-frog + multiplicative absorbing boundary; configurable boundary_cells       |
| ``FDTD2DSolver`` (2D)                     | Split-field Berenger PML; cubic sigma ramp; matched impedance σ* = σ·(μ₀/ε₀)        |
| ``MeepAdapter``                           | Optional bridge to `pymeep`; unavailable runtimes raise ``ImportError`` and never fabricate simulation results |
| ``CrosstalkModel.analyze_bank``           | Uniform parallel-bank crosstalk; adjacent + next-nearest pairs                      |
| ``CrosstalkModel.analyze_pairs``          | Per-pair O(N²) crosstalk for arbitrary geometry; Rust parallel via Rayon            |
| ``WaveguidePair``                         | Coupled-mode properties as lazy Python properties                                   |
| Rust acceleration                         | ``py_ph_route_waveguides``, ``py_ph_cascade_mzi``, ``py_ph_analyze_power_budget``, ``py_ph_analyze_crosstalk_bank``, ``py_ph_analyze_crosstalk_pairs`` |
| Native crosstalk parity                   | Engine Rust and standalone Rust/Go/Julia/Mojo execute source-bound parity contracts against Python |
| Focused verification                      | 102 tests; exact 100% coverage over 702 statements and 214 branches in the facade and seven responsibility modules |

---

## 5. Usage example with output

```python
import numpy as np
from sc_neurocore.optics.photonic_emitter import (
    PhotonicCompiler, PhotonicTarget, CrosstalkModel,
    FDTD2DSolver, CompilationResult,
)

# 1. Compile a 200-step SC bitstream onto silicon photonics.
bitstream = (np.arange(200) % 3 == 0).astype(np.uint8)
target = PhotonicTarget.silicon_photonics()
compiler = PhotonicCompiler(target=target)
result = compiler.compile_bitstream(bitstream, run_fdtd=True, fdtd_steps=200)
print(f"Target         : {result.target}")
print(f"Modulators     : {result.num_modulators}")
print(f"Power mean_mW  : {result.optical_power_mean_mw:.4f}")
print(f"Phase coverage : {result.phase_coverage_rad:.2f} rad")
print(f"FDTD energy    : {result.fdtd_energy:.4e}")

# 2. Crosstalk for an 8-waveguide bank, 180 nm pitch, 15 um coupler.
cx = CrosstalkModel()
bank = cx.analyze_bank(waveguides=8, gap_nm=180.0, coupling_length_um=15.0)
print(f"worst iso_dB   : {bank['worst_isolation_db']:.2f}")
print(f"crosstalk_safe : {bank['crosstalk_safe']}")

# 3. 2D Berenger PML sanity.
s = FDTD2DSolver(nx=200, ny=100, pml_layers=12)
s.set_waveguide(y_center=50, width_cells=10, refractive_index=3.48)
s.inject_source(x=50, y=50, wavelength_nm=1550.0, amplitude=1.0, sigma_cells=8)
s.step(500)
print(f"FDTD2D energy  : {s.field_energy():.4e}")

# 4. Export to real GDSII.
info = result.to_gdsii("demo.gds", mzi_length_um=12.5, pitch_um=80.0)
print(f"GDSII written  : {info['filename']} "
      f"({info['n_modulators']} MZI × {info['pitch_um']} um pitch)")
```

Typical output (CPython 3.12, sc-neurocore-engine wheel built from
the repo):

```
Target         : SiPh-Generic
Modulators     : 67
Power mean_mW  : 0.3333
Phase coverage : 3.14 rad
FDTD energy    : 1.46e-02
worst iso_dB   : 10.52
crosstalk_safe : False
FDTD2D energy  : 1.38e-02
GDSII written  : demo.gds (67 MZI × 80.0 um pitch)
```

``crosstalk_safe = False`` is *correct*: at a 180 nm gap and 15 um
coupling length, adjacent-pair isolation drops below 20 dB — the
design needs more pitch. The demo exercises the whole stack in ~2 s.

---

## 6. Technical reference

### 6.1 `PhotonicBitstreamLayer`

```python
class PhotonicBitstreamLayer:
    n_channels: int
    laser_power: float = 1.0

    def simulate_interference(self, length: int) -> np.ndarray
    def forward(self, input_probs: np.ndarray, length: int) -> np.ndarray
```

``forward(input_probs, length)`` returns a ``(n_channels, length)`` uint8
bitstream where the measured rate per channel approaches
``input_probs[i]`` as ``length → ∞``. No CUDA / Rust dependency —
pure NumPy.

### 6.2 `BitstreamToOptical`

```python
class BitstreamToOptical:
    target: PhotonicTarget

    def convert(self, bitstream: np.ndarray, pulse_duration_ps=10.0) -> list[OpticalPulse]
    def to_phase_array(self, bitstream: np.ndarray) -> np.ndarray
    def to_amplitude_array(self, bitstream: np.ndarray) -> np.ndarray
    def optical_power_profile(self, bitstream, input_power_mw=1.0) -> np.ndarray
```

Phase/amplitude/power arrays are **vectorised** (NumPy broadcasts) so
large bitstreams convert in one pass.

### 6.3 `PhotonicCompiler` + `CompilationResult`

```python
class PhotonicCompiler:
    target: PhotonicTarget
    converter: BitstreamToOptical

    def compile_bitstream(
        self,
        bitstream: np.ndarray,
        run_fdtd: bool = False,
        fdtd_steps: int = 100,
    ) -> CompilationResult
```

``run_fdtd=True`` co-simulates an FDTD run and fills
``CompilationResult.fdtd_energy``; otherwise that field is zero.

```python
@dataclass
class CompilationResult:
    target: str
    num_modulators: int
    optical_power_mean_mw: float
    phase_coverage_rad: float
    netlist: str
    fdtd_energy: float = 0.0

    def to_gdsii(
        self,
        filename: str,
        mzi_length_um: float = 10.0,
        pitch_um: float = 100.0,
    ) -> dict[str, Any]
```

``to_gdsii`` activates the generic PDK on demand, creates MZI cells
with ``allow_duplicate=True`` so repeated exports for the same target
succeed, and stores the SC-NeuroCore header + netlist on GDS TEXT
layer (63/0). Returns an info dict with ``filename``,
``n_modulators``, ``mzi_length_um``, ``pitch_um``,
``total_length_um``, ``target``. Raises ``NotImplementedError`` when
``num_modulators == 0``.

### 6.4 `FDTDSolver` (1D)

```python
class FDTDSolver:
    def __init__(
        self,
        grid_size: int = 1000,
        dx_um: float = 0.01,
        dt_factor: float = 0.5,
        refractive_index: float = 3.48,
        boundary_cells: int = 20,
    )

    def set_loss(self, loss_db_per_cm: float) -> None
    def inject_pulse(self, position, wavelength_nm=1550.0, amplitude=1.0, phase=0.0)
    def step(self, n_steps: int = 1) -> None
    def field_energy(self) -> float
    def snapshot(self) -> tuple[np.ndarray, np.ndarray]  # (ez, hy)
```

### 6.5 `FDTD2DSolver` (2D split-field Berenger PML)

```python
class FDTD2DSolver:
    def __init__(
        self,
        nx: int = 200,
        ny: int = 100,
        dx_um: float = 0.01,
        dy_um: float = 0.01,
        dt_factor: float = 0.5,
        pml_layers: int = 10,
    )

    def set_waveguide(self, y_center, width_cells, refractive_index=3.48, x_start=0, x_end=None)
    def inject_source(self, x, y, wavelength_nm=1550.0, amplitude=1.0, sigma_cells=10)
    def step(self, n_steps: int = 1) -> None
    def field_energy(self) -> float
    def field_at_point(self, x, y) -> float
    def cross_section(self, x) -> np.ndarray
    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]  # (ez, hx, hy)
```

Validation: ``inject_source`` rejects out-of-bounds coordinates and
non-positive wavelengths; ``set_waveguide`` rejects refractive index
< 1; ``step`` rejects zero refractive index in the grid.

### 6.6 `CrosstalkModel` + `WaveguidePair`

```python
@dataclass
class WaveguidePair:
    waveguide_width_nm: float = 450.0
    gap_nm: float = 200.0
    coupling_length_um: float = 10.0
    core_index: float = 3.48
    cladding_index: float = 1.45
    wavelength_nm: float = 1550.0

    @property effective_index_diff: float
    @property coupling_coefficient: float       # per um
    @property coupling_ratio: float              # sin^2(kL)
    @property isolation_db: float                # -10 log10(ratio)

class CrosstalkModel:
    pairs: list[WaveguidePair]

    def add_pair(self, pair: WaveguidePair) -> None
    def transfer_matrix(self, pair: WaveguidePair) -> np.ndarray
    def compute_crosstalk(self, pair, input_power=(1.0, 0.0)) -> tuple[float, float]
    def worst_case_isolation(self) -> float

    def analyze_bank(
        self,
        waveguides: int,
        gap_nm: float,
        coupling_length_um: float,
        wavelength_nm: float = 1550.0,
        core_index: float = 3.48,
        cladding_index: float = 1.45,
    ) -> dict[str, Any]

    def analyze_pairs(
        self,
        pair_indices: list[tuple[int, int]],
        gaps_nm: list[float],
        coupling_lengths_um: list[float],
        wavelength_nm: float = 1550.0,
        core_index: float = 3.48,
        cladding_index: float = 1.45,
    ) -> dict[str, Any]
```

``compute_crosstalk(pair, input_power=(a, b))`` interprets the tuple
as field **amplitudes**; output power sums exactly match $|a|^2 + |b|^2$
per unitary $T$.

### 6.7 Rust FFI surface

| FFI name                         | Purpose                                                                     |
| -------------------------------- | --------------------------------------------------------------------------- |
| ``py_ph_route_waveguides``       | Mesh routing via Manhattan distance + crossings over an adjacency matrix   |
| ``py_ph_mzi_transfer_matrix``    | Single MZI 2×2 unitary                                                      |
| ``py_ph_cascade_mzi``            | N-stage MZI cascade (matrix multiplication)                                |
| ``py_ph_analyze_crosstalk``      | Spectral WDM crosstalk (channels = (id, λ, bw, power))                     |
| ``py_ph_analyze_power_budget``   | Laser → detector path loss budget                                          |
| ``py_ph_analyze_crosstalk_bank`` | Uniform parallel bank — closed-form per-pair + aggregate stats             |
| ``py_ph_analyze_crosstalk_pairs``| Per-pair geometric crosstalk — O(N²) Rayon-parallel over pairs             |

The validated Python fallback is exercised when ``_HAS_RUST_PH == False``.
Focused engine tests require relative tolerance ``1e-9`` and absolute
tolerance ``1e-12``. The separate source-bound Rust, Go, Julia, and Mojo
crosstalk mirrors compile and execute against tighter per-runtime envelopes;
they do not mirror the routing, MZI, power-budget, FDTD, Meep, netlist, or GDS
surfaces.

---

## 7. Performance benchmarks

The committed crosstalk harness is
``benchmarks/bench_photonic_crosstalk.py``. It validates source hashes and the
first output tuple before recording timing data in
``benchmarks/results/local_python_2026-07-14_photonic_crosstalk.json``.

### 7.1 Enrolled workload and parity

The workload contains 4,096 pairs. Gap is ``180 + index mod 64`` nm and
coupling length is ``8 + index mod 17`` µm at 1550 nm with core/cladding
indices 3.48/1.45. The final measured first-pair maximum absolute differences
from Python were:

| Runtime | Maximum absolute difference | Enrolled envelope |
| ------- | ---------------------------: | ----------------: |
| Rust    | ``0``                        | ``1e-15``         |
| Go      | ``1.78e-15``                 | ``3e-15``         |
| Julia   | ``0``                        | ``1e-15``         |
| Mojo    | ``1.35e-10``                 | ``2e-10``         |

The runtime-specific envelopes reflect ordinary floating-point-library and ABI
differences. They are correctness bounds, not a claim that the implementations
are bit-identical.

### 7.2 Local regression timings

| Runtime | Median time per 4,096-pair batch |
| ------- | -------------------------------: |
| Python  | 12.785 ms                        |
| Rust    | 0.165 ms                         |
| Go      | 0.422 ms                         |
| Julia   | 0.219 ms                         |
| Mojo    | 0.144 ms                         |

These results were recorded on Linux x86-64 with CPython 3.12.3, Rust 1.96.0,
Go 1.24.0, Julia 1.12.6, and Mojo 1.0.0b1. The process was pinned to logical
CPU 0, but the host had no isolated CPUs, used the ``powersave`` governor, and
was loaded. Native runtimes also use different call boundaries. The artefact is
therefore marked ``local_regression_non_isolated`` and
``promotion_eligible: false``: it is reproducible local diagnostic evidence,
not a universal speed or production-throughput claim.

## 8. Citations

1. **Abhari, N., Hofmann, G. W., Reiter, R.** (2019). *True random
   number generators based on quantum phase noise of coherent laser
   light.* Optics Express 27(12): 17295–17309. — Phase-noise TRNG
   principle used by :class:`PhotonicBitstreamLayer`.
2. **Berenger, J.-P.** (1994). *A perfectly matched layer for the
   absorption of electromagnetic waves.* Journal of Computational
   Physics 114(2): 185–200. — Split-field PML used by
   :class:`FDTD2DSolver`.
3. **Marcatili, E. A. J.** (1969). *Dielectric rectangular waveguide
   and directional coupler for integrated optics.* Bell System
   Technical Journal 48(7): 2071–2102. — Evanescent transverse decay
   used in ``analyze_bank`` / ``analyze_pairs``.
4. **Okamoto, K.** (2006). *Fundamentals of Optical Waveguides*,
   2nd ed., Chapter 4. Academic Press. ISBN 0-12-525096-7. — Empirical
   coupling constants for Si / SiO₂ waveguides at 1550 nm.
5. **Shen, Y., Harris, N. C., et al.** (2017). *Deep learning with
   coherent nanophotonic circuits.* Nature Photonics 11: 441–446. —
   MZI-cascade photonic neural networks; conceptual ancestor of
   :class:`PhotonicCompiler`.
6. **Taflove, A. & Hagness, S. C.** (2005). *Computational
   Electrodynamics: The Finite-Difference Time-Domain Method*,
   3rd ed., Chapters 3 & 7. Artech House. ISBN 1-58053-832-0. —
   FDTD discretisation + $\sigma_\text{max}$ default used by
   :class:`FDTD2DSolver`.
7. **Yee, K.** (1966). *Numerical solution of initial boundary value
   problems involving Maxwell's equations in isotropic media.* IEEE
   Transactions on Antennas and Propagation 14(3): 302–307. — Yee
   leap-frog grid used by both 1D and 2D solvers.

---

## 9. Limitations

- **1D absorbing boundary is not a PML.** See §1.4; for higher-order
  rejection use the 2D solver (which *does* have Berenger PML) and
  take a 1D slice.
- **``CompilationResult.to_gdsii`` requires gdsfactory.** On minimal
  installs the function raises ``ImportError`` with a clear pointer to
  ``pip install sc-neurocore[optics]``. The optional-extra CI job installs and
  executes the GDSII tests; minimal environments skip that dependency-gated
  slice.
- **``MeepAdapter.run_simulation`` requires pymeep.** An unavailable Meep
  runtime raises ``ImportError``. The adapter does not return synthetic field,
  flux, or energy data as a substitute for a simulation.
- **The 2D FDTD solver is pure NumPy.** Grids larger than ~1 M cells
  run slowly. A Rust port is a future work item (see §7.3 for current
  throughput numbers).
- **Marcatili decay is a first-order approximation.** For > 20 dB
  isolation designs, cross-check against a full-vectorial mode solver
  (Lumerical FDE, or the Meep adapter on the same geometry) — the
  empirical $\Delta n_\text{eff}(g) = 0.1\,e^{-g/L_\text{decay}}$
  prefactor is calibrated to Si / SiO₂ at 1550 nm.
- **No differentiable path.** Compilation → GDS is one-shot; there is
  no autograd over the compiler stages. For differentiable photonic
  design use
  [gdsfactory-plugins](https://gdsfactory.github.io/gdsfactory/) with
  your own Jax/Torch adjoint.

---

## Reference

- Stable facade: `src/sc_neurocore/optics/photonic_emitter.py` (64 lines).
- Responsibility modules:
  `src/sc_neurocore/optics/_photonic_{types,conversion,fdtd,compiler,meep,crosstalk,emitter}.py`
  (106, 117, 297, 281, 218, 321, and 83 lines respectively). Their dependency
  graph and per-module ceilings are executable architecture contracts.
- Layer module: `src/sc_neurocore/optics/photonic_layer.py`.
- Focused verification: 102 tests, including public compatibility, deterministic
  valid-output digests, validation, FDTD, GDSII, bridge integration, native
  engine parity, and executable standalone Rust/Go/Julia/Mojo crosstalk parity.
  The facade and seven responsibility modules have exact 100% statement and
  branch coverage over 702 statements and 214 branches.
- Rust engine: `engine/src/photonic.rs`; the crosstalk floor and zero-length
  contract are shared with the standalone mirrors.
- Benchmark: `benchmarks/bench_photonic_crosstalk.py`, with native-build helpers
  in `benchmarks/_photonic_crosstalk_native.py` and committed local evidence in
  `benchmarks/results/local_python_2026-07-14_photonic_crosstalk.json`.
- Demo: `examples/15_photonic_compilation_demo.py` (produces a real
  11.6 KB GDSII file via ``gdsfactory`` + ``klayout``).

::: sc_neurocore.optics.photonic_layer
    options:
      show_root_heading: true

::: sc_neurocore.optics.photonic_emitter
    options:
      show_root_heading: true
