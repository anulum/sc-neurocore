# Audio

Audio entrainment pipeline: adaptive session engine, entrainment verification scoring (EVS),
SSGF-based geometry-to-audio mapping, and per-user profile persistence.

The package facade is the stable import path for application code:

```python
from sc_neurocore.audio import (
    AdaptiveAudioEngine,
    EVSEngine,
    EVSSnapshot,
    SSGFEngine,
    UserProfile,
)
```

`tests/test_audio_package_api.py` locks the package-level exports and verifies
that an imported `SSGFEngine` can advance one step and produce the documented
audio mapping keys.

## Adaptive Engine

`AdaptiveAudioEngine` is the public closed-loop controller for adaptive audio
sessions. Each `EVSSnapshot` advances one controller tick, updates the session
phase, adjusts the `SSGFEngine` configuration when feedback trends change, runs
one geometry step, and returns the current audio mapping.

The controller exposes three stable session phases:

- `discovery`: sweeps the target frequency and keeps geometry parameters in an
  exploratory range.
- `lock_on`: responds to EVS decline or improvement by adjusting geometry
  coupling and learning rate.
- `deepening`: increases field pressure and geometry coupling while reducing
  the learning rate for stability.

`AdaptiveSessionReport` summarises total ticks, EVS statistics, verified
percentage, phase durations, adaptation count, and the final audio mapping in a
JSON-compatible dictionary.

::: sc_neurocore.audio.adaptive_engine

## EVS Engine

`EVSEngine` verifies whether an audio session is tracking its target band by
combining target-band power increase, spectral peak alignment, band dominance,
and recent score stability. Focused EVS contract tests cover baseline gating,
finite target validation, tiny-window fail-closed FFT behaviour, zero-baseline
relative-increase handling, score-history copying, snapshot serialisation, and
reset semantics.

::: sc_neurocore.audio.evs_engine

## SSGF Engine

::: sc_neurocore.audio.ssgf_engine

## User Profile

`UserProfile` persists chronotype defaults, preferred target frequency,
per-band baseline powers, SSGF cost weights, and sensitivity multipliers as a
JSON-compatible profile dictionary. Focused contract tests cover explicit target
overrides, high-EVS target adoption and smoothing, baseline band-power updates,
and package-facade exports.

::: sc_neurocore.audio.user_profile
