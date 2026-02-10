# SC-NeuroCore v3.0 — Phase 3 Codex Work Packets (G-0 through M)

**Status:** Phase 3 — Ready for Codex
**Author:** Principal Systems Architect
**Date:** February 9, 2026
**Depends On:** Phase 2 (Packets D-0 through J) fully delivered and verified
**Constraint:** v2.2.0 Python code under `src/sc_neurocore/` is SACRED. Zero modifications.

---

## PHASE 2 REVIEW SUMMARY

Phase 2 delivered 55 tests (20 Rust + 35 Python), all passing:
- D-0: Phase 1 fixups (config.toml, layer equiv test, NEON fix, SIMD bench)
- D: Surrogate gradients (SurrogateType, SurrogateLif, DifferentiableDenseLayer)
- E: Attention + GNN (rate-mode and SC-mode attention, rate-mode GNN)
- F: SCPN Kuramoto solver (correct phase-difference coupling, Box-Muller noise)
- I: Benchmark suite (Python head-to-head + Rust criterion)
- J: CI/CD pipeline (v3-engine.yml, 3 OS x 2 Python matrix)

**Issues found during Phase 2 code review (to be fixed in Packet G-0):**

1. **FastSigmoid and SuperSpike are algebraically identical:**
   Both compute `1/(1+k|x|)^2`. They differ only in default k (25 vs 100).
   The original papers use different normalizations:
   - FastSigmoid (Zenke & Vogels 2021): `1 / (2k * (1 + k|x|)^2)`
   - SuperSpike (Zenke & Ganguli 2018): `1 / (1 + k|x|)^2`

2. **SCPNMetrics not exposed via PyO3:**
   The `global_coherence()` and `consciousness_index()` functions are
   Rust-only with no Python bindings.

3. **Criterion bench omits attention and GNN:**
   `full_bench.rs` covers pack/popcount/encoder/LIF/dense/Kuramoto but
   not StochasticAttention or StochasticGraphLayer.

4. **Kuramoto solver lacks SSGF-compatible coupling terms:**
   The current solver implements: `dtheta = omega + K*sin(diff) + noise`
   The SSGF micro-cycle needs: `dtheta = omega + K*sin(diff) + sigma_g*W*sin(diff) + pgbo_w*h*sin(diff) + F*cos(theta) + noise`
   These additional terms are needed to wire the Rust solver into the
   SSGF/TCBO/PGBO pipeline.

---

## PHASE 3 OVERVIEW

Phase 3 focuses on three themes:
1. **Integration** — Wire Rust Kuramoto into the SSGF pipeline (the biggest performance win)
2. **Hardening** — Property-based testing for all numeric modules
3. **Completeness** — Multi-head attention, SC-mode GNN, training demo, documentation

Original blueprint Packets G/H (MLIR/CIRCT) are deferred to Phase 4 —
they require LLVM toolchain expertise and are independent of the current
integration work.

### Phase 3 Execution Order

```
G-0 (fixups) ────────────────────────────────→ done
                                                │
          ┌─────────────────┬───────────────────┤
          │                 │                   │
          v                 v                   v
    G (SSGF solver)    H (proptest)    K (attn+gnn ext)
          │                 │                   │
          └────────┬────────┘                   │
                   │                            │
                   v                            │
             L (training demo) <────────────────┘
                   │
                   v
             M (documentation)
```

---

## PACKET G-0: PHASE 2 FIXUPS

```
===============================================================
HANDOVER PROMPT FOR CODEX — PACKET G-0: Phase 2 Fixups
===============================================================

CONTEXT:
Phase 2 was delivered and verified. Code review found 4 issues
that need targeted fixes before Phase 3 work begins.

Repository: sc-neurocore/
Working directory: 03_CODE/sc-neurocore/
Do NOT modify anything under src/sc_neurocore/.

=============================================================
FIX 1: Differentiate FastSigmoid and SuperSpike formulas
=============================================================

File: engine/src/grad/surrogate.rs

CURRENT (BROKEN — both are identical):
```rust
Self::FastSigmoid { k } => 1.0 / (1.0 + k * x.abs()).powi(2),
Self::SuperSpike { k } => 1.0 / (k * x.abs() + 1.0).powi(2),
```

CORRECT (per original papers):
```rust
Self::FastSigmoid { k } => {
    // Zenke & Vogels 2021: includes 1/(2k) normalization
    // so that integral over R is finite and the function
    // integrates to a proper sigmoid.
    let denom = 1.0 + k * x.abs();
    1.0 / (2.0 * k * denom * denom)
}
Self::SuperSpike { k } => {
    // Zenke & Ganguli 2018: unnormalized form.
    let denom = 1.0 + k * x.abs();
    1.0 / (denom * denom)
}
```

UPDATE TESTS in engine/tests/test_surrogate.rs:

```rust
#[test]
fn fast_sigmoid_gradient_at_zero() {
    // FastSigmoid: 1 / (2*k * (1 + k*0)^2) = 1 / (2*25) = 0.02
    let sg = SurrogateType::FastSigmoid { k: 25.0 };
    assert!((sg.grad(0.0) - 0.02).abs() < 1e-6);
}

#[test]
fn superspike_gradient_at_zero_is_one() {
    // SuperSpike: 1 / (1 + k*0)^2 = 1.0
    let sg = SurrogateType::SuperSpike { k: 100.0 };
    assert!((sg.grad(0.0) - 1.0).abs() < 1e-6);
}

#[test]
fn fast_sigmoid_differs_from_superspike() {
    // With the same k, the two must differ by the 1/(2k) factor
    let fs = SurrogateType::FastSigmoid { k: 25.0 };
    let ss = SurrogateType::SuperSpike { k: 25.0 };
    let fs_grad = fs.grad(0.0);
    let ss_grad = ss.grad(0.0);
    // fs_grad = 1/(2*25) = 0.02, ss_grad = 1.0
    assert!((ss_grad / fs_grad - 50.0).abs() < 1e-4);
}
```

NOTE: This changes the numerical output of FastSigmoid for existing
users. Since surrogate gradients are new v3 functionality (not in v2),
there are no backward-compatibility concerns.

=============================================================
FIX 2: Expose SCPNMetrics via PyO3
=============================================================

File: engine/src/lib.rs

ADD PyO3 wrapper for SCPNMetrics:

```rust
#[pyclass(
    name = "SCPNMetrics",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PySCPNMetrics;

#[pymethods]
impl PySCPNMetrics {
    #[new]
    fn new() -> Self { Self }

    #[staticmethod]
    fn global_coherence(weights: [f64; 7], metrics: [f64; 7]) -> f64 {
        scpn::SCPNMetrics::global_coherence(&weights, &metrics)
    }

    #[staticmethod]
    fn consciousness_index(phases_l4: Vec<f64>, glyph_l7: [f64; 6]) -> f64 {
        scpn::SCPNMetrics::consciousness_index(&phases_l4, &glyph_l7)
    }
}

// In #[pymodule]:
m.add_class::<PySCPNMetrics>()?;
```

File: bridge/sc_neurocore_engine/__init__.py

ADD import:
```python
from sc_neurocore_engine.sc_neurocore_engine import SCPNMetrics
```

ADD to __all__: "SCPNMetrics"

File: bridge/sc_neurocore_engine/scpn.py

ADD re-export at top:
```python
from sc_neurocore_engine.sc_neurocore_engine import SCPNMetrics
```

=============================================================
FIX 3: Add attention + GNN to criterion bench
=============================================================

File: engine/benches/full_bench.rs

ADD after Kuramoto benchmark:

```rust
// -- Attention (rate-mode) --
{
    let attn = StochasticAttention::new(16);
    let rng = &mut rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let q: Vec<f64> = (0..10*16).map(|_| rng.gen()).collect();
    let k: Vec<f64> = (0..20*16).map(|_| rng.gen()).collect();
    let v: Vec<f64> = (0..20*32).map(|_| rng.gen()).collect();

    c.bench_function("attention_10x16_20x32", |b| {
        b.iter(|| {
            black_box(attn.forward(
                black_box(&q), 10, 16,
                black_box(&k), 20, 16,
                black_box(&v), 20, 32,
            ).unwrap())
        })
    });
}

// -- Graph Layer --
{
    let adj: Vec<f64> = {
        let mut a = vec![0.0; 20 * 20];
        for i in 0..20 {
            for j in 0..20 {
                if (i as i32 - j as i32).abs() <= 2 {
                    a[i * 20 + j] = 1.0;
                }
            }
        }
        a
    };
    let gnn = StochasticGraphLayer::new(adj, 20, 8, 42);
    let features: Vec<f64> = (0..20*8).map(|i| (i as f64) * 0.01).collect();

    c.bench_function("gnn_20x8_forward", |b| {
        b.iter(|| {
            black_box(gnn.forward(black_box(&features)).unwrap())
        })
    });
}
```

ADD required imports at top of full_bench.rs:
```rust
use sc_neurocore_engine::attention::StochasticAttention;
use sc_neurocore_engine::graph::StochasticGraphLayer;
use rand::Rng;
use rand::SeedableRng;
```

=============================================================
FIX 4: Remove unused import suppression in benchmark
=============================================================

File: scripts/bench_v2_vs_v3.py

REMOVE line 88:
```python
_ = (v2_unpack, v2_and)
```

REMOVE the imports on lines 26-27:
```python
    unpack_bitstream as v2_unpack,
    vec_and as v2_and,
```

These were imported but never used. Clean them up.

=============================================================

VERIFICATION:
```powershell
cd 03_CODE/sc-neurocore

# Build engine
cd bridge && ..\.venv\Scripts\python -m maturin develop --release && cd ..

# Rust tests (updated surrogate tests must pass)
cd engine && cargo clippy --all-targets -- -D warnings && cargo test --tests && cd ..

# Python tests
$env:PYTHONPATH='src'
.\.venv\Scripts\python -m pytest tests/equivalence tests/test_surrogate_python.py tests/test_kuramoto_python.py -q

# Benchmark (criterion)
cd engine && cargo bench --bench full_bench && cd ..
```

Expected: All tests pass. Attention + GNN benchmarks appear in criterion output.

===============================================================
```

---

## PACKET G: SSGF-COMPATIBLE EXTENDED KURAMOTO SOLVER

```
===============================================================
HANDOVER PROMPT FOR CODEX — PACKET G: SSGF Integration Solver
===============================================================

CONTEXT:
The SSGF (Stochastic Synthesis of Geometric Fields) engine is the
primary SCPN simulation pipeline. Its hot inner loop is the
MicroCycleEngine in:
  SCPN-CODEBASE/optimizations/ssgf/micro.py

The micro-cycle runs a Kuramoto-like ODE with 4 ADDITIONAL coupling
terms beyond the basic Kuramoto equation. Currently this is pure
NumPy. The Rust KuramotoSolver from Phase 2 only supports the basic
Kuramoto equation. This packet extends it to support ALL coupling
terms used by the SSGF pipeline.

CURRENT Rust Kuramoto (Phase 2):
  dtheta_n = omega_n + Sigma_m K_nm sin(theta_m - theta_n) + noise

REQUIRED SSGF-compatible Kuramoto:
  dtheta_n = omega_n
           + Sigma_m K_nm sin(theta_m - theta_n)          [baseline]
           + sigma_g * Sigma_m W_nm sin(theta_m - theta_n) [geometry]
           + pgbo_w * Sigma_m h_nm sin(theta_m - theta_n)  [PGBO]
           + F * cos(theta_n)                               [field]
           + noise_amp * eta_n                              [noise]

All four sin(diff) terms share the SAME sin_diff matrix. The key
optimization is computing sin_diff ONCE and reusing it.

Repository: sc-neurocore/engine/src/scpn/
Depends on: Packet G-0 (fixups) and Phase 2 Packet F (existing solver).

GOAL:
Extend KuramotoSolver with an SSGF-compatible step function. Keep the
existing step() API unchanged (backward compatible). Add a new
step_ssgf() method that accepts the additional coupling terms.

=============================================================
FILE 1: engine/src/scpn/kuramoto.rs (MODIFY)
=============================================================

ADD the following fields to KuramotoSolver:

```rust
pub struct KuramotoSolver {
    // ... existing fields unchanged ...

    /// Field pressure strength F (default 0.0 = disabled)
    pub field_pressure: f64,

    /// Scratch: cos(theta) for field term
    cos_theta: Vec<f64>,
    /// Scratch: weighted sin_diff for geometry coupling
    geo_coupling: Vec<f64>,
    /// Scratch: weighted sin_diff for PGBO coupling
    pgbo_coupling: Vec<f64>,
}
```

UPDATE the constructor to accept field_pressure and allocate new scratch:

```rust
impl KuramotoSolver {
    pub fn new(
        omega: Vec<f64>,
        coupling_flat: Vec<f64>,
        initial_phases: Vec<f64>,
        noise_amp: f64,
    ) -> Self {
        // ... existing validation ...
        Self {
            // ... existing fields ...
            field_pressure: 0.0,
            cos_theta: vec![0.0; n],
            geo_coupling: vec![0.0; n],
            pgbo_coupling: vec![0.0; n],
        }
    }

    /// Set field pressure strength.
    pub fn set_field_pressure(&mut self, f: f64) {
        self.field_pressure = f;
    }
```

ADD the SSGF-compatible step function:

```rust
    /// SSGF-compatible step with geometry and PGBO coupling.
    ///
    /// w_flat: row-major geometry matrix W (n*n). Pass empty slice to skip.
    /// sigma_g: geometry coupling strength.
    /// h_flat: row-major PGBO tensor h_munu (n*n). Pass empty slice to skip.
    /// pgbo_weight: PGBO coupling strength.
    ///
    /// The sin_diff matrix is computed ONCE and shared across all terms.
    pub fn step_ssgf(
        &mut self,
        dt: f64,
        seed: u64,
        w_flat: &[f64],
        sigma_g: f64,
        h_flat: &[f64],
        pgbo_weight: f64,
    ) -> f64 {
        let n = self.n;
        let phases = &self.phases;

        // 1. Compute sin_diff matrix (shared by all coupling terms)
        self.sin_diff
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let theta_n = phases[row_idx];
                for (col_idx, value) in row.iter_mut().enumerate() {
                    *value = (phases[col_idx] - theta_n).sin();
                }
            });

        // 2. Generate noise
        if seed == 0 || self.noise_amp == 0.0 {
            self.noise.fill(0.0);
        } else {
            fill_standard_normals(&mut self.noise, seed);
        }

        // 3. Compute geometry coupling: sigma_g * Sigma_m W_nm sin(diff)
        let has_geo = !w_flat.is_empty() && sigma_g != 0.0;
        if has_geo {
            assert_eq!(
                w_flat.len(), n * n,
                "w_flat length mismatch: got {}, expected {}",
                w_flat.len(), n * n
            );
            self.geo_coupling
                .par_iter_mut()
                .enumerate()
                .for_each(|(row_idx, geo_n)| {
                    let w_row = &w_flat[row_idx * n..(row_idx + 1) * n];
                    let sin_row = &self.sin_diff[row_idx * n..(row_idx + 1) * n];
                    *geo_n = sigma_g
                        * w_row.iter().zip(sin_row.iter())
                            .map(|(w, s)| w * s)
                            .sum::<f64>();
                });
        } else {
            self.geo_coupling.fill(0.0);
        }

        // 4. Compute PGBO coupling: pgbo_w * Sigma_m h_nm sin(diff)
        let has_pgbo = !h_flat.is_empty() && pgbo_weight != 0.0;
        if has_pgbo {
            assert_eq!(
                h_flat.len(), n * n,
                "h_flat length mismatch: got {}, expected {}",
                h_flat.len(), n * n
            );
            self.pgbo_coupling
                .par_iter_mut()
                .enumerate()
                .for_each(|(row_idx, pgbo_n)| {
                    let h_row = &h_flat[row_idx * n..(row_idx + 1) * n];
                    let sin_row = &self.sin_diff[row_idx * n..(row_idx + 1) * n];
                    *pgbo_n = pgbo_weight
                        * h_row.iter().zip(sin_row.iter())
                            .map(|(h, s)| h * s)
                            .sum::<f64>();
                });
        } else {
            self.pgbo_coupling.fill(0.0);
        }

        // 5. Compute cos(theta) for field pressure
        if self.field_pressure != 0.0 {
            for (c, &theta) in self.cos_theta.iter_mut().zip(phases.iter()) {
                *c = theta.cos();
            }
        }

        // 6. Assemble dtheta
        self.dtheta
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, dtheta_n)| {
                // Baseline Kuramoto coupling
                let coupling_row = &self.coupling[i * n..(i + 1) * n];
                let sin_row = &self.sin_diff[i * n..(i + 1) * n];
                let coupling_sum = coupling_row.iter()
                    .zip(sin_row.iter())
                    .map(|(k, s)| k * s)
                    .sum::<f64>();

                *dtheta_n = self.omega[i]
                    + coupling_sum
                    + self.geo_coupling[i]
                    + self.pgbo_coupling[i]
                    + self.field_pressure * self.cos_theta[i]
                    + self.noise_amp * self.noise[i];
            });

        // 7. Euler update with phase wrapping
        for (phase, dtheta) in self.phases.iter_mut().zip(self.dtheta.iter()) {
            *phase = (*phase + dtheta * dt).rem_euclid(TWO_PI);
        }

        self.order_parameter()
    }

    /// Run N SSGF-compatible steps.
    /// Returns R after each step.
    pub fn run_ssgf(
        &mut self,
        n_steps: usize,
        dt: f64,
        seed: u64,
        w_flat: &[f64],
        sigma_g: f64,
        h_flat: &[f64],
        pgbo_weight: f64,
    ) -> Vec<f64> {
        let mut order_values = Vec::with_capacity(n_steps);
        for step_idx in 0..n_steps {
            let step_seed = if seed == 0 { 0 } else {
                seed.wrapping_add(step_idx as u64)
            };
            order_values.push(self.step_ssgf(
                dt, step_seed, w_flat, sigma_g, h_flat, pgbo_weight,
            ));
        }
        order_values
    }
```

IMPORTANT: The existing step() and run() methods MUST remain unchanged.
step_ssgf() is an EXTENSION, not a replacement.

=============================================================
FILE 2: engine/src/lib.rs (MODIFY — PyKuramotoSolver)
=============================================================

ADD to the #[pymethods] impl PyKuramotoSolver block:

```rust
    fn set_field_pressure(&mut self, f: f64) {
        self.inner.set_field_pressure(f);
    }

    #[pyo3(signature = (
        dt,
        seed=0,
        w_flat=vec![],
        sigma_g=0.0,
        h_flat=vec![],
        pgbo_weight=0.0,
    ))]
    fn step_ssgf(
        &mut self,
        dt: f64,
        seed: u64,
        w_flat: Vec<f64>,
        sigma_g: f64,
        h_flat: Vec<f64>,
        pgbo_weight: f64,
    ) -> f64 {
        self.inner.step_ssgf(
            dt, seed, &w_flat, sigma_g, &h_flat, pgbo_weight,
        )
    }

    #[pyo3(signature = (
        n_steps,
        dt,
        seed=0,
        w_flat=vec![],
        sigma_g=0.0,
        h_flat=vec![],
        pgbo_weight=0.0,
    ))]
    fn run_ssgf(
        &mut self,
        n_steps: usize,
        dt: f64,
        seed: u64,
        w_flat: Vec<f64>,
        sigma_g: f64,
        h_flat: Vec<f64>,
        pgbo_weight: f64,
    ) -> Vec<f64> {
        self.inner.run_ssgf(
            n_steps, dt, seed, &w_flat, sigma_g, &h_flat, pgbo_weight,
        )
    }
```

=============================================================
FILE 3: bridge/sc_neurocore_engine/scpn.py (MODIFY)
=============================================================

ADD methods to KuramotoSolver class:

```python
    def set_field_pressure(self, f: float):
        """Set the external field pressure strength F."""
        self._engine.set_field_pressure(float(f))

    def step_ssgf(
        self,
        dt: float,
        seed: int = 0,
        W: np.ndarray | None = None,
        sigma_g: float = 0.0,
        h_munu: np.ndarray | None = None,
        pgbo_weight: float = 0.0,
    ) -> float:
        """SSGF-compatible step with geometry and PGBO coupling.

        Parameters
        ----------
        dt : float
            Euler timestep.
        seed : int
            Noise seed (0 = no noise).
        W : np.ndarray or None
            (N, N) geometry matrix. None = skip geometry coupling.
        sigma_g : float
            Geometry coupling strength.
        h_munu : np.ndarray or None
            (N, N) PGBO tensor. None = skip PGBO coupling.
        pgbo_weight : float
            PGBO coupling strength.

        Returns
        -------
        float
            Order parameter R after this step.
        """
        w_flat = (
            np.asarray(W, dtype=np.float64).ravel().tolist()
            if W is not None else []
        )
        h_flat = (
            np.asarray(h_munu, dtype=np.float64).ravel().tolist()
            if h_munu is not None else []
        )
        return float(self._engine.step_ssgf(
            float(dt), int(seed),
            w_flat, float(sigma_g),
            h_flat, float(pgbo_weight),
        ))

    def run_ssgf(
        self,
        n_steps: int,
        dt: float,
        seed: int = 0,
        W: np.ndarray | None = None,
        sigma_g: float = 0.0,
        h_munu: np.ndarray | None = None,
        pgbo_weight: float = 0.0,
    ) -> np.ndarray:
        """Run N SSGF-compatible steps.

        Returns array of R values, one per step.
        """
        w_flat = (
            np.asarray(W, dtype=np.float64).ravel().tolist()
            if W is not None else []
        )
        h_flat = (
            np.asarray(h_munu, dtype=np.float64).ravel().tolist()
            if h_munu is not None else []
        )
        return np.array(self._engine.run_ssgf(
            int(n_steps), float(dt), int(seed),
            w_flat, float(sigma_g),
            h_flat, float(pgbo_weight),
        ), dtype=np.float64)
```

=============================================================
TESTS
=============================================================

CREATE: engine/tests/test_kuramoto_ssgf.rs

```rust
use sc_neurocore_engine::scpn::KuramotoSolver;

#[test]
fn ssgf_step_without_extras_matches_basic_step() {
    // With W=empty, h=empty, field=0, step_ssgf should equal step.
    let n = 16;
    let omega = vec![1.0; n];
    let coupling = vec![0.3; n * n];
    let phases: Vec<f64> = (0..n)
        .map(|i| 0.3 * i as f64)
        .collect();

    let mut solver_a = KuramotoSolver::new(
        omega.clone(), coupling.clone(), phases.clone(), 0.0,
    );
    let mut solver_b = KuramotoSolver::new(
        omega, coupling, phases, 0.0,
    );

    let r_basic = solver_a.step(0.01, 42);
    let r_ssgf = solver_b.step_ssgf(0.01, 42, &[], 0.0, &[], 0.0);

    assert!(
        (r_basic - r_ssgf).abs() < 1e-14,
        "step_ssgf with no extras should match step: basic={r_basic}, ssgf={r_ssgf}",
    );
    assert_eq!(solver_a.get_phases(), solver_b.get_phases());
}

#[test]
fn geometry_coupling_accelerates_synchronization() {
    let n = 50;
    let omega = vec![1.0; n];
    let coupling = vec![0.1; n * n]; // Weak baseline
    let phases: Vec<f64> = (0..n)
        .map(|i| 2.0 * std::f64::consts::PI * ((i * 37 % n) as f64) / (n as f64))
        .collect();
    // Strong geometry coupling (all-to-all)
    let w = vec![1.0; n * n];

    let mut solver_no_geo = KuramotoSolver::new(
        omega.clone(), coupling.clone(), phases.clone(), 0.0,
    );
    let mut solver_with_geo = KuramotoSolver::new(
        omega, coupling, phases, 0.0,
    );

    let r_no_geo = solver_no_geo.run(300, 0.01, 0);
    let r_with_geo = solver_with_geo.run_ssgf(300, 0.01, 0, &w, 1.0, &[], 0.0);

    let r_final_no = *r_no_geo.last().unwrap();
    let r_final_geo = *r_with_geo.last().unwrap();

    assert!(
        r_final_geo > r_final_no + 0.05,
        "Geometry coupling should boost R: no_geo={r_final_no}, geo={r_final_geo}",
    );
}

#[test]
fn field_pressure_creates_preferred_phase() {
    let n = 20;
    let omega = vec![0.0; n]; // Zero natural freq
    let coupling = vec![0.0; n * n]; // No coupling
    let phases: Vec<f64> = (0..n)
        .map(|i| 2.0 * std::f64::consts::PI * (i as f64) / (n as f64))
        .collect();

    let mut solver = KuramotoSolver::new(omega, coupling, phases, 0.0);
    solver.set_field_pressure(1.0);

    // With F*cos(theta), phases should cluster near pi/2
    // (where cos is zero, the stable fixed point)
    solver.run_ssgf(1000, 0.01, 0, &[], 0.0, &[], 0.0);
    let final_phases = solver.get_phases();

    // Check that R is moderate (field creates structure)
    let r = solver.order_parameter();
    assert!(r > 0.1, "Field pressure should create some phase structure, R={r}");
}

#[test]
fn pgbo_coupling_modulates_dynamics() {
    let n = 20;
    let omega = vec![1.0; n];
    let coupling = vec![0.5; n * n];
    let phases: Vec<f64> = (0..n)
        .map(|i| 0.1 * i as f64)
        .collect();
    // PGBO tensor: strong coupling between first half
    let mut h = vec![0.0; n * n];
    for i in 0..n/2 {
        for j in 0..n/2 {
            h[i * n + j] = 2.0;
        }
    }

    let mut solver = KuramotoSolver::new(
        omega, coupling, phases, 0.0,
    );
    let r_values = solver.run_ssgf(200, 0.01, 0, &[], 0.0, &h, 1.0);
    let r_final = *r_values.last().unwrap();

    // PGBO should change dynamics (hard to predict direction,
    // but R should be non-trivial)
    assert!(r_final > 0.0 && r_final <= 1.0);
}
```

CREATE: tests/test_kuramoto_ssgf_python.py

```python
"""Tests for SSGF-compatible Kuramoto solver (Python bridge)."""

import numpy as np
from sc_neurocore_engine import KuramotoSolver


class TestSSGFKuramoto:
    def test_step_ssgf_matches_step_without_extras(self):
        n = 16
        omega = np.ones(n)
        K = np.full((n, n), 0.3)
        phases = np.arange(n) * 0.3

        solver_a = KuramotoSolver(omega, K, phases.copy(), noise_amp=0.0)
        solver_b = KuramotoSolver(omega, K, phases.copy(), noise_amp=0.0)

        r_basic = solver_a.step(0.01, seed=42)
        r_ssgf = solver_b.step_ssgf(0.01, seed=42)

        assert abs(r_basic - r_ssgf) < 1e-14
        np.testing.assert_allclose(solver_a.phases, solver_b.phases, atol=1e-14)

    def test_geometry_coupling_changes_output(self):
        n = 20
        omega = np.ones(n)
        K = np.full((n, n), 0.5)
        phases = np.random.RandomState(42).uniform(0, 2 * np.pi, n)

        solver = KuramotoSolver(omega, K, phases.copy(), noise_amp=0.0)
        W = np.ones((n, n))

        r_values = solver.run_ssgf(
            n_steps=100, dt=0.01,
            W=W, sigma_g=0.5,
        )
        assert len(r_values) == 100
        assert r_values[-1] > r_values[0]

    def test_field_pressure(self):
        n = 20
        omega = np.zeros(n)
        K = np.zeros((n, n))
        phases = np.random.RandomState(42).uniform(0, 2 * np.pi, n)

        solver = KuramotoSolver(omega, K, phases.copy(), noise_amp=0.0)
        solver.set_field_pressure(1.0)

        solver.run_ssgf(n_steps=500, dt=0.01)
        r = solver.order_parameter()
        assert r > 0.1, f"Field pressure should create structure, R={r}"

    def test_pgbo_coupling(self):
        n = 20
        omega = np.ones(n)
        K = np.full((n, n), 0.3)
        phases = np.random.RandomState(42).uniform(0, 2 * np.pi, n)

        solver = KuramotoSolver(omega, K, phases.copy(), noise_amp=0.0)
        h = np.eye(n) * 2.0  # diagonal PGBO tensor

        r = solver.step_ssgf(0.01, W=None, sigma_g=0.0,
                              h_munu=h, pgbo_weight=1.0)
        assert 0.0 <= r <= 1.0
```

CONSTRAINTS:
- step() and run() MUST remain backward-compatible (no signature changes).
- step_ssgf() MUST compute sin_diff ONCE and reuse for all coupling terms.
- Pre-allocated scratch arrays (no allocation per step).
- All rayon parallelism must be data-race-free (sin_diff is immutable
  during dtheta computation).
- Empty slices (len=0) for w_flat/h_flat mean "skip this coupling term".
- Asserting on w_flat/h_flat length only when non-empty.

VERIFICATION:
```powershell
cd 03_CODE/sc-neurocore/engine
cargo clippy --all-targets -- -D warnings
cargo test --tests
cd ../bridge && ..\.venv\Scripts\python -m maturin develop --release && cd ..
$env:PYTHONPATH='src'
.\.venv\Scripts\python -m pytest tests/test_kuramoto_python.py tests/test_kuramoto_ssgf_python.py -v
```

Expected: All existing Kuramoto tests pass unchanged + new SSGF tests pass.

===============================================================
```

---

## PACKET H: PROPERTY-BASED TESTING (proptest)

```
===============================================================
HANDOVER PROMPT FOR CODEX — PACKET H: Property-Based Testing
===============================================================

CONTEXT:
SC-NeuroCore v3 has hand-written tests for specific cases. Phase 3
adds PROPERTY-BASED TESTING using the proptest crate. This finds
edge cases that human-written tests miss: overflows, NaN propagation,
zero-length inputs, extreme values, etc.

Repository: sc-neurocore/engine/
Depends on: Packet G-0 (fixups).

GOAL:
Add proptest dependency and create comprehensive property tests
for all numeric modules.

=============================================================
FILE 1: engine/Cargo.toml (MODIFY)
=============================================================

ADD to [dev-dependencies]:
```toml
proptest = "1.4"
```

=============================================================
FILE 2: engine/tests/prop_bitstream.rs (CREATE)
=============================================================

```rust
use proptest::prelude::*;
use sc_neurocore_engine::bitstream::{pack, unpack, bitwise_and, popcount_words_portable};
use sc_neurocore_engine::simd::popcount_dispatch;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn pack_unpack_roundtrip(bits in prop::collection::vec(0u8..=1, 1..=4096)) {
        let packed = pack(&bits);
        let unpacked = unpack(&packed);
        prop_assert_eq!(&unpacked[..bits.len()], &bits[..]);
    }

    #[test]
    fn popcount_equals_sum(bits in prop::collection::vec(0u8..=1, 1..=4096)) {
        let expected: u64 = bits.iter().map(|b| *b as u64).sum();
        let packed = pack(&bits);
        let portable = popcount_words_portable(&packed.data);
        let dispatch = popcount_dispatch(&packed.data);
        prop_assert_eq!(portable, expected);
        prop_assert_eq!(dispatch, expected);
    }

    #[test]
    fn and_popcount_leq_min(
        a_bits in prop::collection::vec(0u8..=1, 64..=1024),
    ) {
        let b_bits: Vec<u8> = a_bits.iter()
            .map(|x| if *x == 1 { 0 } else { 1 })
            .collect();
        let a = pack(&a_bits);
        let b = pack(&b_bits);
        let result = bitwise_and(&a, &b);
        let count = popcount_words_portable(&result.data);
        // AND of complementary bitstreams should give 0
        prop_assert_eq!(count, 0);
    }

    #[test]
    fn and_self_equals_self(bits in prop::collection::vec(0u8..=1, 64..=1024)) {
        let packed = pack(&bits);
        let result = bitwise_and(&packed, &packed);
        let self_count = popcount_words_portable(&packed.data);
        let result_count = popcount_words_portable(&result.data);
        prop_assert_eq!(self_count, result_count);
    }
}
```

=============================================================
FILE 3: engine/tests/prop_neuron.rs (CREATE)
=============================================================

```rust
use proptest::prelude::*;
use sc_neurocore_engine::neuron::FixedPointLif;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn lif_voltage_bounded(
        leak_k in 1i16..=100,
        gain_k in 1i16..=512,
        i_t in -500i16..=500i16,
        n_steps in 1usize..=500,
    ) {
        let mut lif = FixedPointLif::new(16, 8, 0, 0, 256, 2);
        for _ in 0..n_steps {
            let (spike, v) = lif.step(leak_k, gain_k, i_t, 0);
            // Spike is 0 or 1
            prop_assert!(spike == 0 || spike == 1);
            // Voltage is within Q8.8 range
            prop_assert!(v >= -32768 && v <= 32767);
        }
    }

    #[test]
    fn lif_zero_input_decays_to_rest(
        leak_k in 10i16..=100,
    ) {
        let mut lif = FixedPointLif::new(16, 8, 0, 0, 256, 2);
        // Push voltage up
        for _ in 0..10 {
            lif.step(leak_k, 256, 200, 0);
        }
        // Zero input: voltage should decay toward v_rest (0)
        for _ in 0..1000 {
            lif.step(leak_k, 256, 0, 0);
        }
        let (_, v) = lif.step(leak_k, 256, 0, 0);
        // Should be near rest after many zero-input steps
        prop_assert!(v.abs() < 10, "Expected v near 0, got {}", v);
    }
}
```

=============================================================
FILE 4: engine/tests/prop_kuramoto.rs (CREATE)
=============================================================

```rust
use proptest::prelude::*;
use sc_neurocore_engine::scpn::KuramotoSolver;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn order_parameter_in_range(
        n in 2usize..=50,
        seed in 0u64..=1000,
    ) {
        let omega = vec![1.0; n];
        let coupling = vec![0.0; n * n];
        let phases: Vec<f64> = (0..n)
            .map(|i| std::f64::consts::TAU * (i as f64) / (n as f64) * 0.99)
            .collect();
        let solver = KuramotoSolver::new(omega, coupling, phases, 0.0);
        let r = solver.order_parameter();
        prop_assert!(r >= 0.0 && r <= 1.0 + 1e-10,
            "R out of range: {}", r);
    }

    #[test]
    fn step_preserves_phase_range(
        n in 2usize..=30,
        dt in 0.001f64..=0.1,
    ) {
        let omega = vec![1.0; n];
        let coupling = vec![0.1; n * n];
        let phases: Vec<f64> = (0..n)
            .map(|i| (i as f64) * 0.5)
            .collect();
        let mut solver = KuramotoSolver::new(omega, coupling, phases, 0.0);
        solver.step(dt, 42);
        for &p in solver.get_phases() {
            prop_assert!(p >= 0.0 && p < std::f64::consts::TAU,
                "Phase out of [0, 2pi): {}", p);
        }
    }

    #[test]
    fn identical_phases_stay_coherent(
        n in 2usize..=50,
        coupling_strength in 0.0f64..=5.0,
    ) {
        let omega = vec![1.0; n];
        let coupling = vec![coupling_strength; n * n];
        let phases = vec![1.0; n]; // All identical
        let mut solver = KuramotoSolver::new(omega, coupling, phases, 0.0);

        // When all phases start identical, they should stay synchronized
        // (identical omega + coupling only affects through phase differences)
        for _ in 0..10 {
            solver.step(0.01, 0);
        }
        let r = solver.order_parameter();
        prop_assert!(r > 0.99, "Identical phases should stay coherent, R={}", r);
    }
}
```

=============================================================
FILE 5: engine/tests/prop_layer.rs (CREATE)
=============================================================

```rust
use proptest::prelude::*;
use sc_neurocore_engine::layer::DenseLayer;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn forward_output_non_negative(
        n_inputs in 2usize..=32,
        n_neurons in 1usize..=16,
    ) {
        let layer = DenseLayer::new(n_inputs, n_neurons, 256, 42);
        let inputs = vec![0.5; n_inputs];
        let out = layer.forward(&inputs, 42).unwrap();
        prop_assert_eq!(out.len(), n_neurons);
        for &v in &out {
            prop_assert!(v >= 0.0, "Negative output: {}", v);
        }
    }

    #[test]
    fn forward_deterministic(
        seed in 1u64..=10000,
    ) {
        let layer = DenseLayer::new(8, 4, 512, 42);
        let inputs = vec![0.5; 8];
        let out1 = layer.forward(&inputs, seed).unwrap();
        let out2 = layer.forward(&inputs, seed).unwrap();
        prop_assert_eq!(out1, out2, "Same seed should give same output");
    }

    #[test]
    fn forward_zero_input_gives_near_zero(
        n_neurons in 1usize..=16,
    ) {
        let layer = DenseLayer::new(8, n_neurons, 1024, 42);
        let inputs = vec![0.0; 8];
        let out = layer.forward(&inputs, 42).unwrap();
        for &v in &out {
            // Bernoulli(0) should give all-zero bitstreams
            prop_assert!(v < 0.02, "Expected near-zero output for zero input, got {}", v);
        }
    }
}
```

CONSTRAINTS:
- All proptest files are standalone (no dependency on Python bridge).
- Use ProptestConfig to limit case counts (performance-sensitive tests
  should use fewer cases).
- Tests must complete in < 60 seconds total.
- No proptest failures should be committed — fix any discovered bugs.
- Tests cover: bitstream roundtrip, popcount correctness, neuron bounds,
  Kuramoto order parameter range, phase wrapping, layer determinism.

VERIFICATION:
```powershell
cd 03_CODE/sc-neurocore/engine
cargo test --tests -- --test-threads=4
```

Expected: All hand-written tests + all proptest tests pass.

===============================================================
```

---

## PACKET K: MULTI-HEAD ATTENTION + SC-MODE GNN

```
===============================================================
HANDOVER PROMPT FOR CODEX — PACKET K: Attention & GNN Extensions
===============================================================

CONTEXT:
Phase 2 delivered single-head StochasticAttention and rate-mode-only
StochasticGraphLayer. Phase 3 extends both:

1. Multi-head attention (split Q, K, V across heads, concatenate)
2. SC-mode GNN (bitstream-based message passing, matching attention's
   forward_sc pattern)

These are NEW v3 capabilities (not in v2) — no equivalence tests needed,
only correctness tests.

Repository: sc-neurocore/engine/src/
Depends on: Packet G-0 (fixups).

=============================================================
FILE 1: engine/src/attention.rs (MODIFY)
=============================================================

ADD method to StochasticAttention:

```rust
    /// Multi-head attention: split Q/K/V across n_heads, run forward
    /// on each head, concatenate results.
    ///
    /// q: (N, dim_k * n_heads) row-major
    /// k: (M, dim_k * n_heads) row-major
    /// v: (M, dim_v * n_heads) row-major
    /// n_heads: number of attention heads
    ///
    /// Returns: (N, dim_v * n_heads) row-major (concatenated head outputs)
    pub fn forward_multihead(
        &self,
        q: &[f64], q_rows: usize, q_total_cols: usize,
        k: &[f64], k_rows: usize, k_total_cols: usize,
        v: &[f64], v_rows: usize, v_total_cols: usize,
        n_heads: usize,
    ) -> Result<Vec<f64>, String> {
        if q_total_cols % n_heads != 0 || k_total_cols % n_heads != 0 || v_total_cols % n_heads != 0 {
            return Err(format!(
                "Total columns must be divisible by n_heads={}. Got Q={}, K={}, V={}.",
                n_heads, q_total_cols, k_total_cols, v_total_cols
            ));
        }
        let dk = q_total_cols / n_heads;
        let dk_k = k_total_cols / n_heads;
        let dv = v_total_cols / n_heads;

        if dk != dk_k {
            return Err(format!(
                "Q/K head dimensions must match: Q_head={}, K_head={}.", dk, dk_k
            ));
        }

        // Split and process each head in parallel
        let head_outputs: Vec<Vec<f64>> = (0..n_heads)
            .into_par_iter()
            .map(|h| {
                // Extract head h from Q: columns [h*dk..(h+1)*dk]
                let q_head = extract_head_columns(q, q_rows, q_total_cols, h, dk);
                let k_head = extract_head_columns(k, k_rows, k_total_cols, h, dk);
                let v_head = extract_head_columns(v, v_rows, v_total_cols, h, dv);

                self.forward(
                    &q_head, q_rows, dk,
                    &k_head, k_rows, dk,
                    &v_head, v_rows, dv,
                ).expect("head forward failed")
            })
            .collect();

        // Concatenate: for each row, append all head outputs
        let out_cols = dv * n_heads;
        let mut out = Vec::with_capacity(q_rows * out_cols);
        for i in 0..q_rows {
            for h in 0..n_heads {
                let head_row = &head_outputs[h][i * dv..(i + 1) * dv];
                out.extend_from_slice(head_row);
            }
        }
        Ok(out)
    }
```

ADD helper function (outside impl block, in same file):

```rust
fn extract_head_columns(
    matrix: &[f64], rows: usize, total_cols: usize,
    head_idx: usize, head_cols: usize,
) -> Vec<f64> {
    let offset = head_idx * head_cols;
    let mut out = Vec::with_capacity(rows * head_cols);
    for i in 0..rows {
        let row_start = i * total_cols + offset;
        out.extend_from_slice(&matrix[row_start..row_start + head_cols]);
    }
    out
}
```

=============================================================
FILE 2: engine/src/graph.rs (MODIFY — add SC-mode forward)
=============================================================

ADD method to StochasticGraphLayer:

```rust
    /// SC-mode forward pass using bitstream AND+popcount.
    ///
    /// Encodes adjacency, features, and weights as bitstreams,
    /// then uses AND+popcount for all matrix multiplies.
    ///
    /// length: bitstream length for SC encoding.
    /// seed: RNG seed for Bernoulli encoding.
    ///
    /// Returns: flat row-major (n_nodes * n_features)
    pub fn forward_sc(
        &self,
        node_features: &[f64],
        length: usize,
        seed: u64,
    ) -> Result<Vec<f64>, String> {
        if node_features.len() != self.n_nodes * self.n_features {
            return Err(format!(
                "node_features length mismatch: got {}, expected {}.",
                node_features.len(),
                self.n_nodes * self.n_features
            ));
        }
        if length == 0 {
            return Err("length must be > 0 for SC mode.".to_string());
        }

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let words = length.div_ceil(64);

        // Encode adjacency (only non-zero entries)
        let adj_packed = encode_matrix_prob_to_packed(
            &self.adj, self.n_nodes, self.n_nodes, length, words, &mut rng,
        );

        // Encode node features
        let feat_packed = encode_matrix_prob_to_packed(
            node_features, self.n_nodes, self.n_features, length, words, &mut rng,
        );

        // Step 1: Message passing via SC: agg[i][f] = popcount(AND(adj_bits[i][j], feat_bits[j][f])) / length
        let mut agg = vec![0.0_f64; self.n_nodes * self.n_features];
        for i in 0..self.n_nodes {
            for f in 0..self.n_features {
                let mut pop_total = 0_u64;
                for j in 0..self.n_nodes {
                    let a = &adj_packed[i * self.n_nodes + j];
                    let b = &feat_packed[j * self.n_features + f];
                    for w in 0..words {
                        pop_total += crate::bitstream::swar_popcount_word(a[w] & b[w]);
                    }
                }
                agg[i * self.n_features + f] = pop_total as f64 / length as f64;
            }
            // Degree normalization
            if self.degrees[i] != 0.0 {
                for f in 0..self.n_features {
                    agg[i * self.n_features + f] /= self.degrees[i];
                }
            }
        }

        // Encode aggregated features
        let agg_packed = encode_matrix_prob_to_packed(
            &agg, self.n_nodes, self.n_features, length, words, &mut rng,
        );

        // Encode weights (clamped to [0,1] for probability encoding)
        let w_clamped: Vec<f64> = self.weights.iter().map(|w| w.clamp(0.0, 1.0)).collect();
        let w_packed = encode_matrix_prob_to_packed(
            &w_clamped, self.n_features, self.n_features, length, words, &mut rng,
        );

        // Step 2: Transform via SC: out[i][f'] = tanh(popcount(AND(agg_bits[i][f], w_bits[f][f'])) / length)
        let mut out = Vec::with_capacity(self.n_nodes * self.n_features);
        for i in 0..self.n_nodes {
            for f_out in 0..self.n_features {
                let mut pop_total = 0_u64;
                for g in 0..self.n_features {
                    let a = &agg_packed[i * self.n_features + g];
                    let b = &w_packed[g * self.n_features + f_out];
                    for w in 0..words {
                        pop_total += crate::bitstream::swar_popcount_word(a[w] & b[w]);
                    }
                }
                out.push((pop_total as f64 / length as f64).tanh());
            }
        }

        Ok(out)
    }
```

ADD required imports at top of graph.rs:
```rust
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
```

ADD helper function (reuse from attention.rs or make it a shared utility):
If encode_matrix_prob_to_packed is not accessible from graph.rs, create
a minimal copy or move it to a shared module (e.g., bitstream.rs).

PREFERRED: Move encode_matrix_prob_to_packed from attention.rs to
bitstream.rs as a public function, then use it from both modules.

=============================================================
FILE 3: engine/src/lib.rs (MODIFY — expose new methods)
=============================================================

ADD to PyStochasticAttention:
```rust
    #[pyo3(signature = (q, k, v, n_heads))]
    fn forward_multihead(
        &self,
        q: &Bound<'_, PyAny>,
        k: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
        n_heads: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let (q_data, q_rows, q_cols) = extract_matrix_f64(q, "Q")?;
        let (k_data, k_rows, k_cols) = extract_matrix_f64(k, "K")?;
        let (v_data, v_rows, v_cols) = extract_matrix_f64(v, "V")?;

        let out = self.inner.forward_multihead(
            &q_data, q_rows, q_cols,
            &k_data, k_rows, k_cols,
            &v_data, v_rows, v_cols,
            n_heads,
        ).map_err(PyValueError::new_err)?;

        let out_cols = v_cols; // Total output cols = v_cols (unchanged)
        Ok(reshape_flat_to_rows(out, q_rows, out_cols))
    }
```

ADD to PyStochasticGraphLayer:
```rust
    #[pyo3(signature = (node_features, length=1024, seed=44257))]
    fn forward_sc(
        &self,
        node_features: &Bound<'_, PyAny>,
        length: usize,
        seed: u64,
    ) -> PyResult<Vec<Vec<f64>>> {
        let (x_flat, x_rows, x_cols) = extract_matrix_f64(node_features, "node_features")?;
        if x_rows != self.inner.n_nodes || x_cols != self.inner.n_features {
            return Err(PyValueError::new_err(format!(
                "Expected node_features shape ({}, {}), got ({}, {}).",
                self.inner.n_nodes, self.inner.n_features, x_rows, x_cols
            )));
        }
        let out = self.inner.forward_sc(&x_flat, length, seed)
            .map_err(PyValueError::new_err)?;
        Ok(reshape_flat_to_rows(out, self.inner.n_nodes, self.inner.n_features))
    }
```

=============================================================
BRIDGE WRAPPERS
=============================================================

File: bridge/sc_neurocore_engine/attention.py

ADD method to StochasticAttention:
```python
    def forward_multihead(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        n_heads: int,
    ) -> np.ndarray:
        """Multi-head attention. Q/K/V columns split across heads."""
        Q = np.asarray(Q, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        if Q.ndim == 1: Q = Q[None, :]
        if K.ndim == 1: K = K[None, :]
        if V.ndim == 1: V = V[None, :]
        result = self._engine.forward_multihead(Q, K, V, int(n_heads))
        return np.asarray(result, dtype=np.float64)
```

File: bridge/sc_neurocore_engine/graphs.py

ADD method to StochasticGraphLayer:
```python
    def forward_sc(
        self,
        node_features: np.ndarray,
        length: int = 1024,
        seed: int = 44257,
    ) -> np.ndarray:
        """SC-mode forward pass using bitstream AND+popcount."""
        X = np.asarray(node_features, dtype=np.float64)
        result = self._engine.forward_sc(X, int(length), int(seed))
        return np.asarray(result, dtype=np.float64).reshape(
            self.n_nodes, self.n_features
        )
```

=============================================================
TESTS
=============================================================

CREATE: tests/test_multihead_attention.py

```python
"""Tests for multi-head attention (new v3 capability)."""

import numpy as np
from sc_neurocore_engine.attention import StochasticAttention


class TestMultiHeadAttention:
    def test_single_head_matches_forward(self):
        """With n_heads=1, forward_multihead == forward."""
        rng = np.random.RandomState(42)
        Q = rng.uniform(0, 1, (5, 8))
        K = rng.uniform(0, 1, (10, 8))
        V = rng.uniform(0, 1, (10, 4))

        attn = StochasticAttention(dim_k=8)
        out_single = attn.forward(Q, K, V)
        out_multi = attn.forward_multihead(Q, K, V, n_heads=1)

        np.testing.assert_allclose(out_single, out_multi, atol=1e-12)

    def test_multi_head_output_shape(self):
        """Multi-head output should have same total columns as V."""
        rng = np.random.RandomState(42)
        n_heads = 4
        Q = rng.uniform(0, 1, (5, 32))   # 4 heads * 8 per head
        K = rng.uniform(0, 1, (10, 32))
        V = rng.uniform(0, 1, (10, 16))   # 4 heads * 4 per head

        attn = StochasticAttention(dim_k=32)
        out = attn.forward_multihead(Q, K, V, n_heads=n_heads)

        assert out.shape == (5, 16)

    def test_multi_head_not_equal_to_single(self):
        """Multi-head should differ from treating all columns as one head."""
        rng = np.random.RandomState(42)
        Q = rng.uniform(0, 1, (5, 16))
        K = rng.uniform(0, 1, (10, 16))
        V = rng.uniform(0, 1, (10, 8))

        attn = StochasticAttention(dim_k=16)
        out_1head = attn.forward_multihead(Q, K, V, n_heads=1)
        out_2head = attn.forward_multihead(Q, K, V, n_heads=2)

        # With 2 heads, each head uses 8-dim Q/K — different from 16-dim
        assert not np.allclose(out_1head, out_2head)
```

CREATE: tests/test_gnn_sc_mode.py

```python
"""Tests for SC-mode GNN (new v3 capability)."""

import numpy as np
from sc_neurocore_engine.graphs import StochasticGraphLayer


class TestGNNScMode:
    def test_sc_mode_output_shape(self):
        adj = np.eye(5) + np.roll(np.eye(5), 1, axis=0)
        adj = (adj + adj.T).clip(0, 1)
        np.fill_diagonal(adj, 0.0)
        gnn = StochasticGraphLayer(adj, n_features=4)
        X = np.random.RandomState(42).uniform(0, 1, (5, 4))

        out = gnn.forward_sc(X, length=1024)
        assert out.shape == (5, 4)

    def test_sc_mode_deterministic(self):
        adj = np.eye(5)
        gnn = StochasticGraphLayer(adj, n_features=4)
        X = np.full((5, 4), 0.5)

        out1 = gnn.forward_sc(X, length=1024, seed=42)
        out2 = gnn.forward_sc(X, length=1024, seed=42)
        np.testing.assert_array_equal(out1, out2)

    def test_sc_mode_approximates_rate_mode(self):
        """Long bitstreams should converge toward rate-mode result."""
        rng = np.random.RandomState(42)
        adj = rng.randint(0, 2, (5, 5)).astype(np.float64)
        adj = (adj + adj.T) / 2
        np.fill_diagonal(adj, 0.0)

        gnn = StochasticGraphLayer(adj, n_features=4)
        X = rng.uniform(0.2, 0.8, (5, 4))

        out_rate = gnn.forward(X)
        out_sc = gnn.forward_sc(X, length=32768)

        np.testing.assert_allclose(out_rate, out_sc, atol=0.1,
            err_msg="SC mode should approximate rate mode with long bitstreams")

    def test_isolated_node_sc(self):
        adj = np.zeros((3, 3))
        adj[0, 1] = adj[1, 0] = 1.0
        gnn = StochasticGraphLayer(adj, n_features=4)
        X = np.ones((3, 4)) * 0.5

        out = gnn.forward_sc(X, length=1024)
        # Node 2 is isolated -> aggregation is 0 -> tanh(0) = 0
        np.testing.assert_allclose(out[2], 0.0, atol=0.05)
```

CONSTRAINTS:
- forward_multihead must use rayon parallelism across heads.
- forward_sc must use the same encode_matrix_prob_to_packed function
  from attention.rs (move to shared location if needed).
- No equivalence tests for multi-head or SC-mode GNN (both are new v3).
- SC-mode GNN accuracy test tolerance: atol=0.1 (SC is approximate).

VERIFICATION:
```powershell
cd 03_CODE/sc-neurocore/engine
cargo clippy --all-targets -- -D warnings
cargo test --tests
cd ../bridge && ..\.venv\Scripts\python -m maturin develop --release && cd ..
$env:PYTHONPATH='src'
.\.venv\Scripts\python -m pytest tests/test_multihead_attention.py tests/test_gnn_sc_mode.py -v
```

===============================================================
```

---

## PACKET L: END-TO-END TRAINING DEMO

```
===============================================================
HANDOVER PROMPT FOR CODEX — PACKET L: Training Loop Example
===============================================================

CONTEXT:
Phase 2 delivered the surrogate gradient primitives but no end-to-end
training example. This packet creates a self-contained demo that:
1. Creates a DifferentiableDenseLayer
2. Generates synthetic classification data
3. Trains using forward → backward → update_weights loop
4. Shows loss decreasing over epochs

Repository: sc-neurocore/examples/
Depends on: Packet G-0 (surrogate fix).

GOAL:
Create a runnable Python example demonstrating training.

=============================================================
FILE 1: examples/01_sc_training_demo.py (CREATE)
=============================================================

```python
#!/usr/bin/env python
"""
SC-NeuroCore v3 — Surrogate Gradient Training Demo
=====================================================

Demonstrates end-to-end training of a stochastic computing
dense layer using surrogate gradients.

Task: Binary classification on a simple 2D dataset (XOR-like).
The SC layer learns to separate the classes using bitstream
computation with surrogate-gradient-based weight updates.

Usage:
    cd 03_CODE/sc-neurocore
    $env:PYTHONPATH='src'
    .\.venv\Scripts\python examples/01_sc_training_demo.py
"""

from __future__ import annotations

import numpy as np

from sc_neurocore_engine import DifferentiableDenseLayer


def generate_xor_data(n_samples: int, rng: np.random.RandomState):
    """Generate a noisy XOR classification dataset."""
    X = rng.uniform(0, 1, (n_samples, 4))
    # Target: XOR of whether each pair of inputs is above/below 0.5
    y = np.zeros(n_samples)
    for i in range(n_samples):
        a = int(X[i, 0] > 0.5) ^ int(X[i, 1] > 0.5)
        b = int(X[i, 2] > 0.5) ^ int(X[i, 3] > 0.5)
        y[i] = float(a ^ b)
    return X, y


def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((pred - target) ** 2))


def mse_grad(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Gradient of MSE loss w.r.t. predictions."""
    return 2.0 * (pred - target) / len(pred)


def main():
    rng = np.random.RandomState(42)
    n_train = 200
    n_epochs = 50
    lr = 0.005
    length = 2048  # Bitstream length (longer = less variance)

    print("SC-NeuroCore v3 — Surrogate Gradient Training Demo")
    print("=" * 55)

    # Generate data
    X_train, y_train = generate_xor_data(n_train, rng)

    # Create layer: 4 inputs -> 1 output neuron
    layer = DifferentiableDenseLayer(
        n_inputs=4,
        n_neurons=1,
        length=length,
        surrogate="fast_sigmoid",
        k=25.0,
    )

    print(f"Layer: 4 -> 1, L={length}, surrogate=fast_sigmoid")
    print(f"Training: {n_train} samples, {n_epochs} epochs, lr={lr}")
    print()
    print(f"{'Epoch':<8} {'Loss':<12} {'Accuracy':<12}")
    print("-" * 32)

    for epoch in range(n_epochs):
        total_loss = 0.0
        correct = 0

        for i in range(n_train):
            x = X_train[i]
            target = np.array([y_train[i]])

            # Forward
            pred = layer.forward(x, seed=42 + epoch * n_train + i)

            # Loss
            loss = mse_loss(pred, target)
            total_loss += loss

            # Accuracy (threshold at 0.5 of max possible output)
            threshold = 2.0  # Roughly half of max sum for 4 inputs
            predicted_class = 1.0 if pred[0] > threshold else 0.0
            if predicted_class == target[0]:
                correct += 1

            # Backward
            grad_out = mse_grad(pred, target)
            _, grad_w = layer.backward(grad_out)

            # Update
            layer.update_weights(grad_w, lr=lr)

        avg_loss = total_loss / n_train
        accuracy = correct / n_train * 100

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"{epoch:<8} {avg_loss:<12.6f} {accuracy:<10.1f}%")

    print()
    print("Training complete.")
    print("Note: SC layers are stochastic — loss may fluctuate.")
    print("The surrogate gradient enables weight updates despite")
    print("the non-differentiable bitstream AND+popcount forward pass.")


if __name__ == "__main__":
    main()
```

VERIFICATION:
```powershell
cd 03_CODE/sc-neurocore
$env:PYTHONPATH='src'
.\.venv\Scripts\python examples/01_sc_training_demo.py
```

Expected: Loss decreases over epochs. Script runs in < 30 seconds.

===============================================================
```

---

## PACKET M: RUSTDOC + API DOCUMENTATION

```
===============================================================
HANDOVER PROMPT FOR CODEX — PACKET M: Documentation
===============================================================

CONTEXT:
The engine crate has minimal documentation. Phase 3 adds:
1. Comprehensive rustdoc for all public types and methods
2. Updated migration guide covering Phase 2+3 features
3. Module-level doc comments explaining architecture

Repository: sc-neurocore/engine/src/
Depends on: All other Phase 3 packets.

GOAL:
Add documentation so that `cargo doc --open` produces a useful
reference for all v3 engine types.

=============================================================
RUSTDOC REQUIREMENTS (modify existing files)
=============================================================

For EACH public struct, enum, function, and method in:
- bitstream.rs
- encoder.rs
- neuron.rs
- layer.rs
- attention.rs
- graph.rs
- grad/surrogate.rs
- scpn/kuramoto.rs
- scpn/metrics.rs
- simd/mod.rs

ADD or IMPROVE doc comments covering:
1. What the type/function does (1-2 sentences)
2. Parameters (for functions with > 2 params)
3. Return value semantics
4. Mathematical formula where applicable
5. Example usage (for key public types)

Module-level doc comments for each .rs file:
```rust
//! # Bitstream Operations
//!
//! Core data structure and operations for stochastic computing.
//! Bitstreams encode probabilities as sequences of {0, 1} values,
//! packed into `u64` words for efficient SIMD processing.
//!
//! ## Key Types
//! - [`BitStreamTensor`] — Packed bitstream with original length tracking
//! - [`pack`] — Convert `&[u8]` to packed `u64` representation
//! - [`unpack`] — Convert back to `&[u8]`
//!
//! ## Performance
//! SIMD-accelerated popcount via [`crate::simd::popcount_dispatch`].
```

=============================================================
FILE 2: docs/v3_migration.md (UPDATE)
=============================================================

ADD Phase 2+3 feature sections:

```markdown
## Phase 2 Features (February 2026)

### Surrogate Gradients
SC-NeuroCore v3 introduces backpropagation support for stochastic
computing layers via surrogate gradients:

- `SurrogateLif` — LIF neuron with differentiable backward pass
- `DifferentiableDenseLayer` — SC layer with weight gradient computation
- Supported surrogates: FastSigmoid, SuperSpike, ArcTan, StraightThrough

### Stochastic Attention
- Rate-mode: bit-exact match with v2 (atol < 1e-12)
- SC-mode: bitstream-based matrix multiply (new v3 capability)
- Multi-head support (Phase 3)

### Graph Neural Network
- Rate-mode: bit-exact match with v2 (atol < 1e-12)
- SC-mode: bitstream-based message passing (Phase 3)

### Kuramoto Oscillator Solver
- High-performance phase-difference coupling
- SSGF-compatible extended solver with geometry + PGBO terms
- Pre-allocated scratch arrays, rayon parallelism
- Box-Muller noise generation with ChaCha8Rng

## Phase 3 Features (February 2026)

### SSGF Integration
- `step_ssgf()` — Extended Kuramoto with geometry (W), PGBO (h_munu),
  and field pressure (F*cos) coupling terms
- Direct integration with SSGF MicroCycleEngine pipeline
- Single sin_diff computation shared across all coupling terms

### Property-Based Testing
- proptest coverage for all numeric modules
- Catches edge cases: overflows, NaN, extreme values
```

=============================================================

VERIFICATION:
```powershell
cd 03_CODE/sc-neurocore/engine
cargo doc --no-deps
```

Expected: No doc warnings. All public types documented.

===============================================================
```

---

## DELIVERY CHECKLIST

### Packet G-0 (Phase 2 Fixups)
- [ ] Fix FastSigmoid formula: add 1/(2k) normalization
- [ ] Update surrogate tests for new FastSigmoid value
- [ ] Add `fast_sigmoid_differs_from_superspike` test
- [ ] Expose SCPNMetrics via PyO3 in lib.rs
- [ ] Add SCPNMetrics import in bridge __init__.py
- [ ] Add attention + GNN benchmarks to full_bench.rs
- [ ] Remove unused v2_unpack/v2_and from benchmark script
- [ ] All tests pass

### Packet G (SSGF Solver)
- [ ] Add field_pressure, cos_theta, geo_coupling, pgbo_coupling to KuramotoSolver
- [ ] Implement step_ssgf() with all 4 coupling terms
- [ ] Implement run_ssgf()
- [ ] Add set_field_pressure()
- [ ] Add PyO3 bindings for step_ssgf, run_ssgf, set_field_pressure
- [ ] Add Python bridge methods (step_ssgf, run_ssgf, set_field_pressure)
- [ ] Create engine/tests/test_kuramoto_ssgf.rs (4 tests)
- [ ] Create tests/test_kuramoto_ssgf_python.py (4 tests)
- [ ] step_ssgf with no extras matches basic step (bit-identical)
- [ ] Existing step()/run() backward-compatible

### Packet H (Property Testing)
- [ ] Add proptest = "1.4" to dev-dependencies
- [ ] Create engine/tests/prop_bitstream.rs (4 property tests)
- [ ] Create engine/tests/prop_neuron.rs (2 property tests)
- [ ] Create engine/tests/prop_kuramoto.rs (3 property tests)
- [ ] Create engine/tests/prop_layer.rs (3 property tests)
- [ ] All proptest tests pass

### Packet K (Attention + GNN Extensions)
- [ ] Implement forward_multihead() in attention.rs
- [ ] Implement forward_sc() in graph.rs
- [ ] Move encode_matrix_prob_to_packed to shared location
- [ ] Add PyO3 bindings for both new methods
- [ ] Add Python bridge methods
- [ ] Create tests/test_multihead_attention.py (3 tests)
- [ ] Create tests/test_gnn_sc_mode.py (4 tests)

### Packet L (Training Demo)
- [ ] Create examples/01_sc_training_demo.py
- [ ] Script runs and shows loss decreasing

### Packet M (Documentation)
- [ ] Add module-level rustdoc for all .rs files
- [ ] Add doc comments for all public types/methods
- [ ] Update docs/v3_migration.md with Phase 2+3 features
- [ ] cargo doc --no-deps produces no warnings

---

## STRICT RULES FOR CODEX

1. **NEVER modify any file under `src/sc_neurocore/`.** Sacred v2.2.0.
2. **NEVER modify `pyproject.toml` in the repo root.** Sacred v2.2.0 packaging.
3. **NEVER modify `.github/workflows/ci.yml`.** Use `v3-engine.yml` only.
4. **NEVER modify existing tests under `tests/`.** Only ADD new test files.
5. **NEVER modify the existing step() and run() signatures** on KuramotoSolver.
   The new SSGF methods are EXTENSIONS.
6. All Rust code must compile with `cargo clippy --all-targets -- -D warnings`.
7. All Rust code must pass `cargo fmt -- --check`.
8. The sin_diff matrix in step_ssgf MUST be computed ONCE and reused.
9. encode_matrix_prob_to_packed should be in a shared location accessible
   from both attention.rs and graph.rs.
10. Property tests must complete in < 60 seconds total.

---

## VERIFICATION SEQUENCE (Run All)

```powershell
cd 03_CODE/sc-neurocore

# 1. Rust quality gates
cd engine
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings
cargo test --tests
cargo doc --no-deps
cd ..

# 2. Build Python extension
cd bridge && ..\.venv\Scripts\python -m maturin develop --release && cd ..

# 3. Python tests (all)
$env:PYTHONPATH='src'
.\.venv\Scripts\python -m pytest tests/equivalence tests/test_surrogate_python.py tests/test_kuramoto_python.py tests/test_kuramoto_ssgf_python.py tests/test_multihead_attention.py tests/test_gnn_sc_mode.py -v --tb=short

# 4. Training demo
.\.venv\Scripts\python examples/01_sc_training_demo.py

# 5. Benchmarks
cd engine && cargo bench --bench full_bench && cd ..
.\.venv\Scripts\python scripts/bench_v2_vs_v3.py
```

Expected: All quality gates pass, all tests pass, demo runs, benchmarks complete.

---

Anulum CH&LI / Anulum Institute
Miroslav Sotek
ORCID: 0009-0009-3560-0851

(c) 1998-2026 Anulum Institute. All rights reserved.
