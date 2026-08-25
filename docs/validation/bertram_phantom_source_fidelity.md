# Bertram phantom source-fidelity validation

The catalogue identity follows Bertram, Previte, Sherman, Kinard, and Satin
(2000), DOI `10.1016/S0006-3495(00)76525-8`, equations 1–10 and the authors'
`BJ_00.ode` parameter file.

The primary audit established four continuous states `(V,n,s1,s2)`. In
particular, `n` is a dynamic fast potassium gate; treating it as instantaneous
changes the system. The author defaults are `V=-43`, `n=.03`, `s1=.1`,
`s2=.434`, `gCa=280`, `gK=1300`, `gs1=20`, `gs2=32`, `gL=25`, `Cm=4524`,
`tau_n_bar=9.09 ms`, `tau_s1=1000 ms`, and `tau_s2=120000 ms`.

The production specialization uses simultaneous fixed-step RK4 at `0.5 ms`
with constant current during each sample. The author program selected adaptive
CVODE; therefore continuous equations and parameters are source-faithful while
adaptive interpolation and continuous event timing are not claimed. The
additive external current and sampled upward `-20 mV` event are explicit
extensions, and events do not reset state.

Independent evidence consists of:

- a separately written NumPy RK4 oracle for all four derivatives;
- a 512-step mixed-drive receipt with three events, four final states, and
  SHA-256 `61f83582949ed90cf07c9c1c294324ee1524ba4509bfb1b8be0274ad64cdce29`;
- one-step Python/Rust/Julia/Go/Mojo parity within `5e-13`;
- a 10,000-step, 18-event native trajectory with maximum state error below
  `5e-9` and source-hashed local benchmark evidence.

The old three-state recurrence is preserved independently as
`SCThreeStatePhantomBurster` and makes no literature claim.
