# Bioware interface

`sc_neurocore.bioware` is an **experimental research interface** for moving a
finite multi-electrode-array (MEA) frame through spike detection, Address-Event
Representation (AER), stochastic bitstream encoding, and an optogenetic pulse
proposal. It is not a medical device, a clinical controller, or a tissue-safety
certification surface.

The maintained implementation is Python-only. Earlier generated Go, Julia,
Mojo, and Rust files were non-executable placeholders with no dispatch path and
have been removed. The separately maintained Julia plasticity solvers remain
documented in [Julia solvers](julia_solvers.md); they are not a second
implementation of this closed-loop orchestration.

```python
from sc_neurocore.bioware import (
    AERToSCConverter,
    BioHybridSession,
    CultureHealth,
    MEAConfig,
    MEAToAERTranscoder,
    SCToOptoEncoder,
    SpikeDetector,
)
```

## Maintained boundaries

The historical module
`sc_neurocore.bioware.bioware` is a compatibility facade. Implementations are
owned by focused modules:

| Module | Responsibility |
| --- | --- |
| `bioware_contracts.py` | Stable MEA, spike, AER, pulse, and frame-result records |
| `bioware_validation.py` | Shared finite-value, shape, integer, and bitstream guards |
| `bioware_acquisition.py` | MAD noise estimation, threshold detection, sorting, artifact blanking |
| `bioware_encoding.py` | MEA→AER, AER→SC, SC→opto, and rate decoding |
| `bioware_plasticity.py` | Pair-STDP, BCM, and homeostatic Q8.8 adapters |
| `bioware_analysis.py` | Culture heuristic, LFP bands, latency, and network bursts |
| `bioware_experiment.py` | Pharmacology prototype and multi-well metadata |
| `bioware_audit.py` | Deterministic in-memory audit records and checksum |
| `bioware_session.py` | One-frame closed-loop orchestration |
| `bioware_fitness.py` | Legacy evolutionary fitness adapter |

Package-root objects, historical qualified names, and pickle lookup paths are
preserved. The facade contains no implementation definitions.

## Signal contract

### MEA frame and spike detection

Input must be a non-empty, finite numeric NumPy matrix with shape
`(samples, channels)`. Its channel count must exactly match `MEAConfig`.

For channel (c), the detector estimates a robust noise scale

\[
\hat\sigma_c = \frac{\operatorname{median}_t |V_{t,c}|}{0.6745}
\]

and detects crossings of

\[
|V_{t,c}| > \alpha \hat\sigma_c,
\]

where `spike_threshold_sigma` is (alpha). The `0.6745` scaling is the
standard Gaussian MAD conversion used in extracellular spike detection; see
the primary-culture methods discussion in
[Maccione et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3213406/).
The refractory interval is enforced independently per channel. Waveform
snippets are edge-padded to a fixed length.

`SpikeSorter` is a deliberately small PCA + K-Means research adapter. It uses a
fixed `random_state=0` by default, validates uniform waveform lengths, and
requires the optional `scikit-learn` dependency only when enough waveforms are
available to fit clusters.

### AER epoch

`MEAToAERTranscoder` converts a spike timestamp and explicit origin to a clock
tick:

\[
\tau = \left\lfloor (t_{spike} - t_0) f_{hw} \right\rfloor.
\]

The maintained packet has a 16-bit unsigned timestamp. Therefore
(0 \le \tau \le 65535); negative relative time and overflow are errors.
The implementation never silently wraps timestamps. At the default 1 MHz
clock, one frame must fit within approximately 65.536 ms. Longer recordings
must be split into explicit epochs by the caller.

`BioHybridSession` treats detector timestamps as frame-relative. Its
`t_start_s` argument is non-negative experiment time for optional experiment
models; it does not change the frame-local AER origin.

### AER to stochastic bitstreams

For valid events inside `window_ticks`, the converter counts events per neuron.
If (k_n) is the count for neuron (n), it uses

\[
p_n = \frac{k_n}{\max_j k_j}.
\]

`p_n` is encoded by a deterministic 16-bit LFSR into a bitstream of length
`bitstream_length` (default 256). This is not an IID Bernoulli sampler and does
not use a smoothing constant. The configured neuron count and window are hard
bounds rather than unused metadata.

### Stochastic bitstreams to optical pulses

For bitstream density (d_n), the encoder proposes

\[
I_n = d_n I_{max}, \qquad
T_n = T_{min} + d_n(T_{max} - T_{min}).
\]

`intensity_mw_mm2` is irradiance. Optical power is computed with the illuminated
area:

\[
P_n\,[\mathrm{mW}] = I_n\,[\mathrm{mW/mm^2}]\,A_n\,[\mathrm{mm^2}].
\]

The encoder accumulates (P_n) and omits a channel if including it would exceed
`max_total_power_mw`. This software budget is a consistency guard, not a
biological safety limit. Real optical safety depends on wavelength, duty cycle,
geometry, absorption, scattering, thermal transport, and experimental
calibration.

## Plasticity adapters

### Pair STDP

`BiologicalSTDP` implements the parameterised exponential pair rule

\[
\Delta w =
\begin{cases}
A_+e^{-\Delta t/\tau_+}, & \Delta t > 0,\\
-A_-e^{\Delta t/\tau_-}, & \Delta t < 0,\\
0, & \Delta t = 0.
\end{cases}
\]

Defaults are `tau_plus_ms = tau_minus_ms = 20`, `a_plus = 0.01`, and
`a_minus = 0.012`. These are configurable model parameters, not claimed fits to
one specific preparation. The biological timing reference is
[Bi and Poo (1998)](https://doi.org/10.1523/JNEUROSCI.18-24-10464.1998).

### BCM and homeostasis

`BCMPlasticity` uses

\[
\Delta w = \eta x y(y-\theta), \qquad
\theta \leftarrow \theta + \frac{\Delta t}{\tau_\theta}(y^2-\theta),
\]

following the sliding-threshold family introduced by
[Bienenstock, Cooper, and Munro](https://doi.org/10.1523/JNEUROSCI.02-01-00032.1982).

`HomeostaticPlasticity` applies a bounded Q8.8 proportional update:

\[
\theta_{next}^{q88} = \operatorname{clip}\left(
\theta^{q88} + \left\lfloor
\frac{\Delta t}{\tau_h}(r-r^*)\,256
\right\rfloor,\theta_{min}^{q88},\theta_{max}^{q88}\right).
\]

`BioHybridSession` stores `stdp` and optional `homeostatic` policies for caller
coordination, but `process_frame` does **not** update plasticity implicitly.

## Session semantics

`BioHybridSession.process_frame` executes these stages synchronously:

1. validate the frame, experiment time, AER epoch, and converter window;
2. optionally blank stimulus artifacts;
3. detect and optionally sort spikes;
4. optionally apply the pharmacology rate prototype;
5. transcode to AER and deterministic SC bitstreams;
6. optionally pass decoded rates to an ArcaneZenith object;
7. propose optogenetic pulses and calculate a culture-health snapshot;
8. create a typed `BioHybridFrameResult`, record optional latency, and only
   then advance `round_count`.

An exception before completion leaves `round_count` unchanged. The detector may
still retain its internal noise estimate, and an external Zenith object may
have its own state; callers needing distributed transactions must manage those
resources explicitly.

```python
import numpy as np

config = MEAConfig(num_channels=8, sample_rate_hz=20_000.0)
session = BioHybridSession(
    mea_config=config,
    detector=SpikeDetector(config),
    transcoder=MEAToAERTranscoder(hw_clock_hz=1e6),
    sc_converter=AERToSCConverter(
        window_ticks=0x10000,
        bitstream_length=512,
        num_neurons=config.num_channels,
    ),
    opto_encoder=SCToOptoEncoder(
        illuminated_area_mm2=1.0,
        max_total_power_mw=50.0,
    ),
)

# 1,000 / 20,000 Hz = 50 ms, inside one 16-bit epoch at 1 MHz.
frame = np.zeros((1_000, config.num_channels), dtype=np.float64)
result = session.process_frame(frame, t_start_s=0.0)
assert result.round == result["round"] == 1
```

## Analysis, experiment, and audit limits

- `CultureHealth` returns a bounded aggregate heuristic from channel rates. It
  is not a viability assay or clinical endpoint.
- `extract_lfp_power` is an FFT band-power helper, not a full spectral
  estimator with windowing, leakage correction, or uncertainty intervals.
- `detect_network_bursts` is a binned threshold heuristic.
- `PharmModel` implements application time, onset interpolation, and a firing
  gain. `wash_time_s` is reserved configuration and is not yet applied as a
  washout curve.
- `BioAuditLog` is an ordered, tamper-evident **in-memory** record. Its canonical
  SHA-256 includes schema name, experiment identity, and entries. It does not
  provide durable storage, signatures, access control, or regulatory
  compliance.
- The legacy fitness key `energy_mw` equals `0.5 * spike_count` for compatibility.
  It is a dimensionless optimisation proxy, not measured power or energy.

## Verification and benchmark evidence

Focused verification executes 200 tests and covers all 1,052 production
statements and 384 branches in the Bioware package. Source and tests are
responsibility-sized, the module import graph is acyclic, the facade and package
objects are identical, and historical pickle paths remain valid.

`benchmarks/results/bench_bioware.json` is a 30-sample, interleaved comparison
of parent `c4e492ff5` and the modular working tree. Both produced exactly 6,865
canonical bytes with SHA-256
`2491dc73a2de93a45a1cc944539c170b151403e42b973b18806143f318b7d669`:

| Local diagnostic metric | Parent median | Modular candidate median | Delta |
| --- | ---: | ---: | ---: |
| Pipeline | 2.801 ms | 3.345 ms | +19.41% |
| Import | 34.771 ms | 50.265 ms | +44.56% |
| Subprocess wall | 579.937 ms | 535.945 ms | -7.59% |
| Maximum RSS | 37,076 KiB | 37,154 KiB | +0.21% |

The capture used taskset affinity but no exclusive isolated core, the CPU
governor was `powersave`, and host load was high. These numbers are local
regression context only. Rerun on reserved isolated hardware before publishing
performance claims.

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_bioware -q

.venv/bin/python benchmarks/bench_bioware.py \
  --baseline-root <clean-parent-tree> \
  --baseline-ref c4e492ff5 \
  --candidate-root . \
  --candidate-ref working-tree \
  --iterations 30 \
  --warmups 2 \
  --output benchmarks/results/bench_bioware.json
```

The closed-loop biohybrid research context is exemplified by
[Kagan et al. (2022)](https://doi.org/10.1016/j.neuron.2022.09.001); that work
does not validate this software implementation or its safety.

## API reference

::: sc_neurocore.bioware.bioware
