# Bioware Interface

Biological-hardware interface for cerebral organoids and multi-electrode
array (MEA) systems. Bridges wet-lab experiments with in-silico SC
simulations — spike detection, AER transcoding, stochastic-computing
feedback, optogenetic encoding, Q8.8 homeostatic plasticity, PCA spike
sorting, and a closed-loop session layer that orchestrates the full
pipeline per frame.

```python
from sc_neurocore.bioware.bioware import (
    BioHybridSession, BioHybridFrameResult,
    MEAConfig, MEALayout, SpikeDetector,
    MEAToAERTranscoder, AERToSCConverter, SCToOptoEncoder,
    CultureHealth, HomeostaticPlasticity, SpikeSorter,
    mea_fitness_hook,
)
```

---

## 1. Mathematical formalism

### 1.1 Spike detection (adaptive threshold, MAD-based noise)

Raw MEA voltage $V_c(t)$ per channel $c$ is detected against an
adaptive threshold derived from the **median absolute deviation**:

$$
\sigma_c = \frac{\mathrm{median}_t |V_c(t)|}{0.6745},
\qquad
\theta_c = \alpha\,\sigma_c.
$$

The constant $0.6745$ converts MAD into an estimator of the Gaussian
standard deviation (Donoho & Johnstone 1994); $\alpha = 5$ by default
(``spike_threshold_sigma``). A spike is emitted when
$|V_c(t)| > \theta_c$ and no spike was emitted on the same channel in
the preceding ``refractory_samples = 30`` samples.

### 1.2 MEA → AER transcoding

For a spike with real-time timestamp $t_s$ (seconds) and a hardware
AER clock $f_\text{hw}$ (Hz), the AER event carries integer tick

$$
\tau = \lfloor t_s \cdot f_\text{hw} \rfloor
$$

and neuron id derived from ``channel_map`` (or the channel number
itself if no map is given). AER events are emitted sorted by $\tau$.

### 1.3 AER → SC bitstream conversion

For each neuron id, the converter bins AER timestamps into a
bitstream of length $L$ (default 1 024). For neuron $n$ observed
over the window $[t_0, t_0 + T]$ with $k_n$ spikes:

$$
p_n = \frac{k_n}{k_n + K},
\qquad b_{n,i} \sim \mathrm{Bernoulli}(p_n),
\qquad i \in [0, L).
$$

where $K$ is a smoothing constant (``smoothing_constant``, default 1).
The resulting SC bitstream has mean activity equal to $p_n$.

### 1.4 Optogenetic encoding (SC → opto pulses)

Given an output bitstream $b \in \{0, 1\}^L$ with density
$d = \frac{1}{L}\sum_i b_i$, the optogenetic pulse uses intensity

$$
I_\text{mW/mm²} = d \cdot I_\text{max},
$$

and duration

$$
T_\text{ms} = T_\text{min} + (T_\text{max} - T_\text{min})\,d,
$$

so bright / long pulses signal high-activity layers and dark / short
pulses signal low-activity. The wavelength is fixed by the
optogenetic channel (``wavelength_nm``, e.g. 470 nm for ChR2).

### 1.5 Biological STDP (Bi & Poo 1998, exponential pair rule)

For a post-spike timed at $t_\text{post}$ after a pre-spike at
$t_\text{pre}$, let $\Delta t = t_\text{post} - t_\text{pre}$. The
weight change is:

$$
\Delta w(\Delta t) =
\begin{cases}
A_+\, \exp(-\Delta t / \tau_+) & \Delta t > 0 \text{ (LTP)} \\
-A_-\, \exp(+\Delta t / \tau_-) & \Delta t < 0 \text{ (LTD)} \\
0                              & \Delta t = 0
\end{cases}
$$

Default $\tau_+ = \tau_- = 20$ ms, $A_+ = 0.005$, $A_- = 0.00525$,
matching Bi & Poo 1998 hippocampal culture fits (1:1.05
potentiation:depression asymmetry).

### 1.6 BCM metaplasticity (sliding-threshold variant)

The BCM rule (Bienenstock, Cooper, Munro 1982) adjusts the LTP/LTD
boundary as a function of post-synaptic activity $\bar y$:

$$
\Delta w = \eta\,x\,y\,(y - \theta_m),
\qquad
\theta_m = \kappa\, \bar y^2,
$$

where $\bar y$ is the temporally-averaged post-synaptic rate and
$\kappa > 0$ tunes the sliding-threshold curvature. Low-frequency
activity below $\theta_m$ depresses; above, potentiates.

### 1.7 Q8.8 homeostatic controller

``HomeostaticPlasticity.update_threshold`` implements a proportional
negative-feedback controller in fixed-point Q8.8 arithmetic:

$$
\alpha_t = \frac{\Delta t_\text{ms}}{\tau_\text{homeo}},
\qquad
\Delta \theta = \lfloor \alpha_t (r_t - r^\star) \cdot 256 \rfloor,
$$

$$
\theta^{(q88)}_{t+1}
= \mathrm{clip}\!\left(\theta^{(q88)}_t + \Delta \theta,
\theta^{(q88)}_\text{min}, \theta^{(q88)}_\text{max}\right).
$$

The factor $256$ is the Q8.8 scale: one full threshold unit in
floating-point corresponds to $256$ steps in the integer
representation. A 1 Hz rate error sustained over one full time
constant $\tau_\text{homeo}$ thus shifts the threshold by exactly one
unit.

### 1.8 PCA + KMeans spike sorting

Given a set of detected waveforms $\{w^{(i)}\}$, each a vector in
$\mathbb{R}^D$ (with $D$ = ``int(2 · snippet_ms · sample_rate_hz /
1000)``), sorting proceeds in two stages:

1. **Dimensionality reduction** via principal component analysis:

   $$
   X = [w^{(1)}, \ldots, w^{(M)}]^\top \in \mathbb{R}^{M\times D},
   \qquad
   X = U \Sigma V^\top,
   \qquad
   Z = X \cdot V_{:k},
   $$

   with $k = \min(\text{n\_components}, M, D)$ (default
   ``n_components = 3``).

2. **Cluster assignment** via $k$-means with $k = \text{num\_units}$
   and $n_\text{init} = 10$ restarts:

   $$
   \min_{\{\mu_c\}} \sum_{i=1}^M \min_{c \in \{1..k\}} \|z^{(i)} - \mu_c\|^2.
   $$

Waveforms flagged ``waveform is None`` (amplitude-only spikes) are
skipped; the sorter is a no-op when fewer than ``num_units`` waveforms
are available.

### 1.9 Evo-substrate fitness hook

The MEA → `ReplicationEngine` bridge computes three fitness scalars:

$$
\bar r = \frac{1}{|C|}\sum_{c \in C} \frac{k_c}{T_\mathrm{frame}},
\qquad
\mathrm{acc} =
\mathrm{clip}\!\left(1 - \frac{|\bar r - r^\star|}{r^\star},\;
0.1,\; 0.99\right),
$$

where $k_c$ is the per-channel spike count, $T_\mathrm{frame}$ is
``duration_s`` when supplied, and $r^\star$ is the target rate
(``target_rate = 10`` Hz by default). When ``duration_s`` is omitted,
the hook preserves the legacy count-domain score for callers that do
not know the frame length. Energy and latency:

$$
E = 0.5\,\text{mW} \cdot N_\text{spikes},
\qquad
T_\mathrm{lat} =
\begin{cases}
T_\mathrm{measured}, & \text{if measured latency is supplied},\\
t_\mathrm{first\ response} - t_\mathrm{stim}, & \text{if stimulus time is supplied},\\
t_\mathrm{first\ spike}, & \text{otherwise}.
\end{cases}
$$

---

## 2. Theoretical context

Bioware sits at the intersection of three active research frontiers.

**Closed-loop biohybrid experiments.** Organoid cultures coupled to
MEAs via optogenetic actuators form a real-time feedback loop between
a digital controller and a living network. Kagan et al. (2022)
demonstrated such a system learning Pong; DishBrain and similar
platforms have generalised the paradigm to control tasks.
SC-NeuroCore's ``BioHybridSession`` is the orchestration layer — it
owns the MEA config, detector, transcoder, SC converter, opto
encoder, and plasticity updates, and guarantees the end-to-end
latency budget in a single ``process_frame`` call.

**Stochastic computing over biological signals.** Traditional MEA
pipelines output rates or spike trains; SC-NeuroCore converts rates
directly into Bernoulli bitstreams consumed by the SC arithmetic
stack. The conversion is lossless up to the SC bit budget —
$L = 1024$ bits encodes $p_n$ with variance $p(1-p)/L \le 1/(4L)
\approx 2.4 \times 10^{-4}$.

**Hardware-grounded homeostatic plasticity.** The Q8.8 controller in
§1.7 is directly synthesisable — the state variable is an integer,
the update is additive with a constant shift, the clamp is a pair of
comparators. The same mathematical form runs on the PYNQ-Z2 silicon
(``sc_aer_encoder.v`` and friends) as in Python, so simulations
quantitatively match hardware traces to Q8.8 precision. Turrigiano
2012 provides the biological reference point: real neurons use much
slower homeostatic timescales (seconds to hours), modelled here by
the ``tau_homeo_ms`` parameter.

**Spike sorting.** PCA + KMeans is the classical first-pass
(Lewicki 1998); modern production systems use SpikeInterface,
Kilosort or MountainSort. The SC-NeuroCore sorter is intentionally
minimal — it exists so a single closed-loop run can demonstrate
end-to-end separation of 2–4 units without external dependencies
beyond scikit-learn, and so the evolutionary-substrate fitness hook
has something to condition on when the MEA returns waveforms.

The overall bioware contract is stricter than typical research code:
every public surface is typed, exception-free for empty inputs,
deterministic under a fixed seed, and round-trippable between
dataclass and dict views. The contract is enforced by 121 focused tests in
``tests/test_bioware/test_bioware.py``.

---

## 3. Pipeline position

Bioware is the *outer* of SC-NeuroCore's two closed loops. The inner
loop is the SC arithmetic stack; the outer loop wraps it in a
real-time bidirectional bridge with the biological substrate.

```
   ┌──────────── BioHybridSession.process_frame() ─────────────┐
   │                                                            │
   │  voltage_data (N_samples × N_channels)                    │
   │        │                                                   │
   │        ▼                                                   │
   │   SpikeDetector ──────►  list[DetectedSpike]              │
   │        │                        │                          │
   │        │                        ├──► (opt) SpikeSorter     │
   │        │                        ├──► (opt) ArtifactReject  │
   │        │                        ▼                          │
   │        │                  MEAToAERTranscoder               │
   │        │                        │ list[AEREvent]           │
   │        │                        ▼                          │
   │        │                  AERToSCConverter                 │
   │        │                        │ dict[id, np.ndarray]     │
   │        │                        ▼                          │
   │        │                  SCToOptoEncoder ──► OptoPulse    │
   │        │                                                   │
   │        └──► CultureHealth ──► health score                 │
   │        └──► HomeostaticPlasticity (Q8.8 threshold update)  │
   │        └──► ArcaneZenithCognitiveCore.step_from_bio_rates  │
   │                                                            │
   └────────► BioHybridFrameResult ──────────────────────────►  │
                                                            caller
```

**Upstream inputs** — the MEA voltage matrix (from any MEA recording
system — PYNQ-Z2 ADC, Intan, BlackRock, synthetic). Format is
``np.ndarray`` shape ``(n_samples, n_channels)``.

**Downstream outputs** — a :class:`BioHybridFrameResult` packet
containing spike counts, AER events, SC bitstreams, optogenetic
pulses, and a CultureHealth snapshot. Both dataclass-style
(``result.round``) and mapping-style (``result["round"]``) access
supported.

The outer loop is driven by the caller — no threads, no background
tasks inside ``process_frame``. Hardware latency / jitter is a
caller concern.

---

## 4. Features

| Feature                                        | Detail                                                                 |
| ---------------------------------------------- | ---------------------------------------------------------------------- |
| MEA config presets                             | 60, 120, 256, 4 096 channel layouts with canonical electrode pitches  |
| MAD-based spike detection                      | Robust σ estimator; per-channel adaptive threshold                     |
| Configurable refractory                        | Default 30 samples; prevents double-counting                           |
| Waveform snippet extraction                    | Per-spike 2 ms window, edge-padded                                     |
| AER transcoding                                | Integer hardware timestamps; channel→neuron map; sorted output         |
| AER → SC bitstream                             | Bernoulli density encoding with smoothing constant                     |
| SC → optogenetic encoding                      | Bright+long for high density; bounded wavelength / intensity / duration |
| Biological STDP                                | Exponential pair rule, Bi & Poo parameters                            |
| BCM metaplasticity                             | Sliding-threshold variant                                             |
| Q8.8 homeostatic plasticity                    | Proportional controller with 1 Hz · τ_homeo = 1 unit calibration       |
| Culture health                                 | Rate-based aggregate score over the frame                             |
| PCA + KMeans spike sorting                     | Optional; scikit-learn optional dep                                   |
| Artifact rejection                             | Optional stim-window blanking                                         |
| Pharmacology model                             | Optional onset-delay + gain model for neurotransmitter pulses          |
| Latency budget                                 | Per-frame wall-clock recording                                         |
| ArcaneZenith bridge                            | ``zenith_core.step_from_bio_rates(...)`` inside process_frame          |
| Evo-substrate fitness hook                     | ``mea_fitness_hook(spikes, target_rate) → {acc, energy, latency}``     |
| BioHybridFrameResult dual interface            | Dataclass attribute access + mapping ``result["key"]`` + ``in``        |
| Audit log                                      | :class:`BioAuditLog` timestamp + entry record for every session        |
| Multi-well plate support                       | :class:`MultiWellPlate` for parallel experiments                       |
| 121 focused tests                              | Spike detection + AER + SC + Opto + STDP + BCM + Culture + Homeostasis |

---

## 5. Usage example with output

```python
import numpy as np
from sc_neurocore.bioware.bioware import (
    BioHybridSession, MEAConfig,
    SpikeDetector, MEAToAERTranscoder, AERToSCConverter,
    SCToOptoEncoder, BiologicalSTDP, CultureHealth,
    HomeostaticPlasticity, SpikeSorter,
)

cfg = MEAConfig(num_channels=10, sample_rate_hz=20_000.0,
                spike_threshold_sigma=5.0)

session = BioHybridSession(
    mea_config=cfg,
    detector=SpikeDetector(config=cfg, refractory_samples=30),
    transcoder=MEAToAERTranscoder(hw_clock_hz=1e6),
    sc_converter=AERToSCConverter(bitstream_length=1024),
    opto_encoder=SCToOptoEncoder(wavelength_nm=470,
                                  max_intensity_mw_mm2=5.0),
    stdp=BiologicalSTDP(),
    health_monitor=CultureHealth(),
    homeostatic=HomeostaticPlasticity(target_rate_hz=10.0),
    sorter=SpikeSorter(num_units=3),
)

rng = np.random.default_rng(42)
V = rng.normal(0, 5, size=(2000, 10))
V[::200, 0] = -80.0                    # spikes on channel 0
V[100::200, 3] = -60.0                 # spikes on channel 3

result = session.process_frame(V)

print(f"round       : {result.round}")
print(f"num_spikes  : {result.num_spikes}")
print(f"num_aer     : {result.num_aer_events}")
print(f"num_streams : {result.num_bitstreams}")
print(f"num_opto    : {result.num_opto_pulses}")
print(f"latency_us  : {result.latency_us:.1f}")

# Dataclass AND mapping access both work:
assert result["round"] == result.round
assert "latency_us" in result
```

Typical output:

```
round       : 1
num_spikes  : 16
num_aer     : 16
num_streams : 2
num_opto    : 2
latency_us  : 3178.4
```

The `examples/14_bioware_closed_loop_demo.py` script extends this to a
100-frame experiment with full SpikeSorter fit + ArcaneZenith cognitive
core + MEA hardware simulation; measured end-to-end wall-clock in §7.

---

## 6. Technical reference

### 6.1 `MEAConfig` + `MEALayout`

```python
class MEALayout(Enum):
    MEA_60   = "60ch"
    MEA_120  = "120ch"
    MEA_256  = "256ch"
    MEA_4096 = "4096ch"
    CUSTOM   = "custom"

@dataclass
class MEAConfig:
    layout: MEALayout = MEALayout.MEA_60
    num_channels: int = 60
    sample_rate_hz: float = 20_000.0
    voltage_gain: float = 1000.0
    noise_floor_uv: float = 5.0
    spike_threshold_sigma: float = 5.0
    electrode_pitch_um: float = 200.0

    @classmethod
    def from_layout(cls, layout: MEALayout) -> MEAConfig: ...
```

### 6.2 `SpikeDetector` + `DetectedSpike`

```python
@dataclass
class DetectedSpike:
    channel: int
    timestamp_s: float
    amplitude_uv: float
    unit_id: int = 0
    waveform: Optional[np.ndarray] = None

@dataclass
class SpikeDetector:
    config: MEAConfig
    refractory_samples: int = 30

    def estimate_noise(self, voltage_data) -> np.ndarray
    def detect(self, voltage_data, snippet_ms: float = 2.0) -> list[DetectedSpike]
```

### 6.3 `MEAToAERTranscoder` + `AERToSCConverter`

```python
@dataclass
class AEREvent:
    neuron_id: int
    timestamp: int           # clock ticks
    valid: bool = True
    weight: int = 256        # Q8.8 = 1.0

class MEAToAERTranscoder:
    hw_clock_hz: float = 1e6
    channel_map: Optional[dict[int, int]] = None

    def transcode(self, spikes, t_start_s: float = 0.0) -> list[AEREvent]

class AERToSCConverter:
    bitstream_length: int = 1024
    smoothing_constant: float = 1.0

    def convert(self, events) -> dict[int, np.ndarray]
```

### 6.4 `SCToOptoEncoder` + `OptogeneticPulse`

```python
@dataclass
class OptogeneticPulse:
    wavelength_nm: int
    intensity_mw_mm2: float
    duration_ms: float
    channel_id: int

@dataclass
class SCToOptoEncoder:
    wavelength_nm: int = 470
    max_intensity_mw_mm2: float = 5.0
    min_pulse_ms: float = 1.0
    max_pulse_ms: float = 50.0

    def encode(self, bitstreams) -> list[OptogeneticPulse]
```

### 6.5 Plasticity classes

```python
@dataclass
class BiologicalSTDP:
    A_plus: float = 0.005
    A_minus: float = 0.00525
    tau_plus_ms: float = 20.0
    tau_minus_ms: float = 20.0

    def compute_dw(self, dt_ms: float) -> float

@dataclass
class HomeostaticPlasticity:
    target_rate_hz: float = 10.0
    tau_homeo_ms: float = 10000.0
    max_threshold_q88: int = 512    # Q8.8 = 2.0
    min_threshold_q88: int = 64     # Q8.8 = 0.25

    def update_threshold(self, current_q88: int,
                         observed_rate_hz: float,
                         dt_ms: float) -> int

@dataclass
class PharmModel:
    agent_name: str = "none"
    gain: float = 1.0
    onset_delay_s: float = 30.0
    wash_time_s: float = 120.0

    def apply(self, t_current_s: float) -> None
    def effective_gain(self, t_current_s: float) -> float
    def modulate_spikes(self, spike_counts: np.ndarray,
                        t_current_s: float) -> np.ndarray
    def modulate_spike_events(self, spikes: list[DetectedSpike],
                              t_current_s: float) -> list[DetectedSpike]
```

``modulate_spike_events`` is the path used by
``BioHybridSession.process_frame``. Inhibitory gains deterministically
thin events across the observed response span, so the pharmacological
model does not bias output toward the earliest detected spikes.
Excitatory gains preserve the observed events and add
template-derived events inside the observed temporal support.

### 6.6 `SpikeSorter`

```python
@dataclass
class SpikeSorter:
    num_units: int = 4
    n_components: int = 3

    def fit(self, spikes: list[DetectedSpike]) -> None
    def assign(self, spikes: list[DetectedSpike]) -> list[DetectedSpike]
```

``fit`` imports scikit-learn **only** when enough waveforms are
present to cluster; amplitude-only spike lists no-op gracefully.

### 6.7 `BioHybridSession` + `BioHybridFrameResult`

```python
@dataclass
class BioHybridFrameResult:
    round: int
    num_spikes: int
    num_aer_events: int
    num_bitstreams: int
    num_opto_pulses: int
    latency_us: float
    health: Dict[str, Any]
    spikes: List[DetectedSpike]
    aer_events: List[AEREvent]
    bitstreams: Dict[int, np.ndarray]
    opto_pulses: List[OptogeneticPulse]

    def __getitem__(self, key: str) -> Any
    def __contains__(self, key: object) -> bool
    def keys(self) -> List[str]

@dataclass
class BioHybridSession:
    mea_config: MEAConfig
    detector: SpikeDetector
    transcoder: MEAToAERTranscoder
    sc_converter: AERToSCConverter
    opto_encoder: SCToOptoEncoder
    stdp: BiologicalSTDP = ...
    health_monitor: CultureHealth = ...
    artifact_rejector: Optional[ArtifactRejector] = None
    pharm_model: Optional[PharmModel] = None
    latency_budget: Optional[LatencyBudget] = None
    homeostatic: Optional[HomeostaticPlasticity] = None
    sorter: Optional[SpikeSorter] = None
    zenith_core: Optional[ArcaneZenithCognitiveCore] = None
    round_count: int = 0

    def process_frame(
        self,
        voltage_data: np.ndarray,
        t_start_s: float = 0.0,
        stim_times_s: Optional[list[float]] = None,
    ) -> BioHybridFrameResult
```

### 6.8 `mea_fitness_hook`

```python
def mea_fitness_hook(
    detected_spikes: list[DetectedSpike],
    target_rate: float = 10.0,
    *,
    duration_s: float | None = None,
    stimulus_time_s: float | None = None,
    measured_latency_ms: float | None = None,
) -> dict[str, float]
```

Returns ``{"accuracy", "energy_mw", "latency_ms"}``. Empty input
returns the floor ``{0.1, 0.0, 0.0}``; ``target_rate == 0`` also
returns the floor accuracy. ``duration_s`` must be finite and
positive when supplied. ``stimulus_time_s`` and
``measured_latency_ms`` must be finite; measured latency must be
non-negative. Groups spikes by ``DetectedSpike.channel`` (regression
guard against the previous ``channel_id`` bug).

---

## 7. Performance benchmarks

All numbers measured 2026-04-20 on Linux x86-64 (Intel i5-11600K,
CPython 3.12.3, scikit-learn 1.8, NumPy 2.2). Committed bench harness:
``benchmarks/bench_bioware.py`` — JSON at
``benchmarks/results/bench_bioware.json``. Reproducer scripts also in §7.4.

Note: the demo uses uniform-random MEA voltage and ArcaneZenith receives
a stochastic current stream; ``identity_drift`` and per-frame latency
therefore vary ≈10 % between runs. The figures in the next two tables
come from one concrete run; the demo wall-time (≈0.69 s on the
reference host) is stable.

### 7.1 End-to-end closed-loop latency

| Scenario                                | Wall time | Per-frame latency |
| --------------------------------------- | --------- | ----------------- |
| 14_bioware_closed_loop_demo (100 frames)| **0.69 s**| **2 945 µs at frame 100** |

Full pipeline: synthetic MEA → SpikeDetector →
SpikeSorter (PCA-fitted on 177 training waveforms) →
MEAToAERTranscoder → AERToSCConverter → SCToOptoEncoder →
:class:`CultureHealth` + :class:`HomeostaticPlasticity` update +
:class:`ArcaneZenithCognitiveCore.step_from_bio_rates`. Per-frame
latency decays over the run as the pipeline warms up — the demo
reports 6 916 µs at frame 20 dropping to 2 945 µs at frame 100.

### 7.2 ArcaneZenith-coupled identity drift

Over the same 100-frame demo the attached
:class:`ArcaneZenithCognitiveCore` reports
``identity_drift = 1.8625`` at frame 100 (printed as "Final ArcaneZenith
identity drift: 1.8625" by the demo). Zero network bursts are detected
in the default 100-frame window; the drift tracks the novelty signal
produced by the closed loop.

### 7.3 `HomeostaticPlasticity.update_threshold` microbenchmark

| Input pattern                                 | Result                   |
| --------------------------------------------- | ------------------------ |
| target 10 Hz, observed 10 Hz, dt = 100 ms    | ``new == 256`` (no change) |
| target 10 Hz, observed 50 Hz, dt = 1000 ms,  | ``new > 256`` (raised)   |
| τ_homeo = 1000 ms                            |                          |
| target 10 Hz, observed 1 Hz, dt = 1000 ms,   | ``new < 256`` (lowered)  |
| τ_homeo = 1000 ms                            |                          |
| 10 000 Hz error for 10 s                     | saturates at ``max_q88`` |
| 0 Hz for 10 s                                | saturates at ``min_q88`` |

All five cases are enforced by the ``TestHomeostaticPlasticity`` test
group. The controller is pure integer arithmetic (subtraction,
multiplication, floor-division, clamp) — CPU cost well below one
microsecond per call.

### 7.4 Reproducer

```bash
# 7.1 + 7.2 end-to-end demo
MPLBACKEND=Agg python examples/14_bioware_closed_loop_demo.py

# 7.3 homeostatic controller
python -c "
from sc_neurocore.bioware.bioware import HomeostaticPlasticity
hp = HomeostaticPlasticity(target_rate_hz=10.0, tau_homeo_ms=1000.0)
print(hp.update_threshold(256, observed_rate_hz=10.0, dt_ms=100.0))  # 256
print(hp.update_threshold(256, observed_rate_hz=50.0, dt_ms=1000.0)) # >256
print(hp.update_threshold(256, observed_rate_hz=1.0,  dt_ms=1000.0)) # <256
"
```

The demo prints ``[3] Experiment complete in 0.69s.`` on the
reference host.

---

## 8. Citations

1. **Bi, G.-Q. & Poo, M.-M.** (1998). *Synaptic modifications in
   cultured hippocampal neurons: dependence on spike timing, synaptic
   strength, and postsynaptic cell type.* Journal of Neuroscience
   18(24): 10464–10472. — Exponential pair-based STDP used by
   :class:`BiologicalSTDP`.
2. **Bienenstock, E. L., Cooper, L. N., Munro, P. W.** (1982).
   *Theory for the development of neuron selectivity: orientation
   specificity and binocular interaction in visual cortex.* Journal
   of Neuroscience 2(1): 32–48. — BCM metaplasticity used by
   :class:`BCMPlasticity`.
3. **Donoho, D. L. & Johnstone, I. M.** (1994). *Ideal spatial
   adaptation by wavelet shrinkage.* Biometrika 81(3): 425–455. —
   MAD / 0.6745 robust σ estimator used by
   :class:`SpikeDetector.estimate_noise`.
4. **Kagan, B. J., Kitchen, A. C., et al.** (2022). *In vitro neurons
   learn and exhibit sentience when embodied in a simulated
   game-world.* Neuron 110(23): 3952–3969.e8. — Reference closed-loop
   MEA-opto experiment motivating :class:`BioHybridSession`.
5. **Lewicki, M. S.** (1998). *A review of methods for spike sorting:
   the detection and classification of neural action potentials.*
   Network: Computation in Neural Systems 9(4): R53–R78. — Classical
   PCA-based spike sorter; direct antecedent of
   :class:`SpikeSorter`.
6. **Turrigiano, G. G.** (2012). *Homeostatic synaptic plasticity:
   local and global mechanisms for stabilizing neuronal function.*
   Cold Spring Harbor Perspectives in Biology 4(1): a005736. —
   Biological reference for the slow homeostatic timescales modelled
   by :class:`HomeostaticPlasticity`.

---

## 9. Limitations

- **SpikeSorter needs scikit-learn.** Without it, ``fit`` raises a
  clear ``ImportError`` with a pointer to the ``[bioware]`` extras.
  The empty-input path is a no-op and does *not* require sklearn.
- **PCA + KMeans is a minimal baseline.** Production spike-sorting
  should use SpikeInterface, Kilosort 3+, or MountainSort. The
  SC-NeuroCore sorter exists as a closed-loop demo.
- **No real-time FPGA coupling inside this module.** The transcoder
  produces AER events suitable for the PYNQ-Z2 hardware, but the
  actual streaming to the FPGA is caller-owned (see ``hdl/``).
- **Q8.8 homeostatic controller is proportional only.** A full
  PID controller would need integral and derivative state; the
  current single-step update is tuned to the slow ``tau_homeo`` regime
  where P is sufficient (Turrigiano 2012).
- **BioHybridFrameResult mapping view is read-only.** ``in``,
  ``[key]``, and ``keys()`` work; ``result[key] = value`` raises.
  Mutate the dataclass fields directly for reply-time updates.

---

## Reference

- Module: `src/sc_neurocore/bioware/bioware.py`.
- Tests: `tests/test_bioware/test_bioware.py` (121 focused tests incl.
  TestMEAConfig, TestSpikeDetector, TestMEAToAERTranscoder,
  TestAERToSCConverter, TestSCToOptoEncoder, TestBiologicalSTDP,
  TestBCMPlasticity, TestCultureHealth, TestBioHybridSession,
  TestRefractoryPeriod, TestOptoSafety, TestEdgeCases,
  TestSpikeSorter, TestLFPExtraction, TestLatencyBudget,
  TestPharmModel, TestMultiWellPlate, TestNetworkBurstDetection,
  TestArtifactRejection, TestBioAuditLog, TestBitstreamRateDecoder,
  TestHomeostaticPlasticity, TestBioHybridFrameResult,
  TestMEAFitnessHook).
- Demo: `examples/14_bioware_closed_loop_demo.py` (100-frame end-to-end
  closed loop with ArcaneZenith + PCA spike sorting).
- Cross-references: :doc:`arcane_zenith` (``zenith_core`` attachment),
  :doc:`evo_substrate` (``mea_fitness_hook``).

::: sc_neurocore.bioware.bioware
    options:
      show_root_heading: true
