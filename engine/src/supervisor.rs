// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! Pure-Rust supervisory execution and verification pipeline.

use core_affinity;
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::Duration;
use std::{error::Error, fmt};
use z3::{
    ast::{Ast, Bool, Int},
    Config, Context, SatResult, Solver,
};

const NUM_PLACES: usize = 4;
const NUM_TRANSITIONS: usize = 3;
const VERIFICATION_DEPTH: usize = 4;
const SAFETY_THRESHOLD_P3: i64 = 100;
const DEFAULT_SNAPSHOT_CAPACITY: usize = 2;
const DEFAULT_SNAPSHOT_PERIOD: u64 = 30;
const DEFAULT_STEP_INTERVAL_NS: u64 = 0;

// Transition structure for the transfer net used by the verifier.
const W_IN: [[i64; NUM_PLACES]; NUM_TRANSITIONS] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]];

const W_OUT: [[i64; NUM_PLACES]; NUM_TRANSITIONS] = [[0, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0]];

/// Snapshot transferred from the RT loop to the Z3 worker.
#[derive(Clone, Debug)]
pub struct PetriNetSnapshot {
    pub step_index: u64,
    pub active_markings: Vec<i64>,
    pub transition_rates: Vec<f64>,
}

/// Shared supervisor state for RT loop + worker lane.
#[derive(Clone, Debug)]
pub struct SupervisorState {
    pub safe_shutdown_flag: Arc<AtomicBool>,
    pub tx_snapshot: Sender<PetriNetSnapshot>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SupervisorExecutionError {
    InvalidNeuronCount,
    SafetyViolation,
}

impl fmt::Display for SupervisorExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidNeuronCount => f.write_str("n_neurons must be > 0"),
            Self::SafetyViolation => f.write_str("safety contract violation detected"),
        }
    }
}

impl Error for SupervisorExecutionError {}

#[derive(Debug)]
struct LightweightSnnPool {
    step_index: u64,
    n_neurons: usize,
    rng: u64,
    transition_rates: [f64; NUM_TRANSITIONS],
    markings: [i64; NUM_PLACES],
}

impl LightweightSnnPool {
    fn new(n_neurons: usize, seed: u64) -> Self {
        Self {
            step_index: 0,
            n_neurons,
            rng: seed ^ 0xA3BF_0000_1234_5678u64,
            transition_rates: [0.0; NUM_TRANSITIONS],
            markings: [12, 15, 8, 0],
        }
    }

    fn step(&mut self) -> f64 {
        self.step_index = self.step_index.saturating_add(1);
        self.rng = self
            .rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);

        let drift = ((self.rng >> 32) as i64) & 0xF;
        let drift = drift.saturating_sub(6);

        // Deterministic signed updates keep markings bounded.
        for (idx, mark) in self.markings.iter_mut().enumerate() {
            let local = ((drift + idx as i64) as f64) / 10.0;
            *mark = (*mark + drift)
                .clamp(0, 200)
                .saturating_add((self.n_neurons as i64) % 2);
            self.transition_rates[idx % NUM_TRANSITIONS] = local.abs();
        }

        // Add a slow rise in the error channel to exercise safety detection.
        self.markings[3] = (self.markings[3] + ((self.step_index as i64) / 8).min(2)) % 210;

        (self.rng as f64) / (u64::MAX as f64)
    }

    fn snapshot(&self, snapshot_step: u64, control_output: f64) -> PetriNetSnapshot {
        let mut transition_rates = self.transition_rates.to_vec();
        transition_rates.push(control_output);

        PetriNetSnapshot {
            step_index: snapshot_step,
            active_markings: self.markings.to_vec(),
            transition_rates,
        }
    }
}

fn bind_core(core_index: usize) {
    let Some(core_ids) = core_affinity::get_core_ids() else {
        return;
    };
    if let Some(core_id) = core_ids.get(core_index) {
        let _ = core_affinity::set_for_current(*core_id);
    }
}

pub fn verify_bounds_at_depth(snapshot: &PetriNetSnapshot, depth: usize) -> bool {
    let cfg = Config::new();
    let context = Context::new(&cfg);
    let solver = Solver::new(&context);

    let mut markings = Vec::with_capacity(depth + 1);
    for step in 0..=depth {
        let mut step_markings = Vec::with_capacity(NUM_PLACES);
        for place in 0..NUM_PLACES {
            let initial = i64::from(*snapshot.active_markings.get(place).unwrap_or(&0));
            if step == 0 {
                step_markings.push(Int::from_i64(&context, initial));
            } else {
                step_markings.push(Int::new_const(&context, format!("mark_{step}_{place}")));
            }
        }
        markings.push(step_markings);
    }

    let mut firings = Vec::with_capacity(depth);
    for step in 0..depth {
        let mut step_firings = Vec::with_capacity(NUM_TRANSITIONS);
        for transition in 0..NUM_TRANSITIONS {
            step_firings.push(Bool::new_const(
                &context,
                format!("fire_{step}_{transition}"),
            ));
        }
        firings.push(step_firings);
    }

    for place in 0..NUM_PLACES {
        solver.assert(&markings[0][place].ge(&Int::from_i64(&context, 0)));
    }

    for step in 0..depth {
        for place in 0..NUM_PLACES {
            let mut next_value = markings[step][place].clone();

            for transition in 0..NUM_TRANSITIONS {
                let fire = &firings[step][transition];
                let as_int = fire.ite(&Int::from_i64(&context, 1), &Int::from_i64(&context, 0));

                let win = Int::from_i64(&context, W_IN[transition][place]);
                let wout = Int::from_i64(&context, W_OUT[transition][place]);
                if W_IN[transition][place] != 0 {
                    next_value -= &win * &as_int;
                }
                if W_OUT[transition][place] != 0 {
                    next_value += &wout * &as_int;
                }
            }

            solver.assert(&markings[step + 1][place]._eq(&next_value));
            solver.assert(&markings[step + 1][place].ge(&Int::from_i64(&context, 0)));
        }
    }

    // Safety condition: no marking in error sink exceeds threshold.
    let threshold = Int::from_i64(&context, SAFETY_THRESHOLD_P3);
    let mut violation_conditions: Vec<Bool> = Vec::with_capacity(depth);
    for step in 1..=depth {
        violation_conditions.push(markings[step][3].gt(&threshold));
    }
    if !violation_conditions.is_empty() {
        let violation_refs: Vec<&Bool> = violation_conditions.iter().collect();
        let violation = Bool::or(&context, &violation_refs);
        solver.assert(&violation);
    } else {
        return true;
    }

    match solver.check() {
        SatResult::Unsat => true,
        SatResult::Sat => false,
        SatResult::Unknown => false,
    }
}

pub fn spawn_z3_verification_worker(
    rx_snapshot: Receiver<PetriNetSnapshot>,
    shutdown_flag: Arc<AtomicBool>,
    target_core: usize,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        bind_core(target_core);

        for snapshot in rx_snapshot {
            let valid = verify_bounds_at_depth(&snapshot, VERIFICATION_DEPTH);
            if !valid {
                shutdown_flag.store(true, Ordering::Release);
                break;
            }
        }
    })
}

fn execute_snn_control_loop(
    mut pool: LightweightSnnPool,
    supervisor: &SupervisorState,
    snapshot_period: u64,
    target_core: usize,
    max_steps: u64,
    step_interval_ns: u64,
) -> u64 {
    bind_core(target_core);

    let snapshot_period = snapshot_period.max(1);
    let mut executed_steps = 0;

    loop {
        if supervisor.safe_shutdown_flag.load(Ordering::Acquire) {
            break;
        }

        if max_steps != 0 && executed_steps >= max_steps {
            break;
        }

        let control_output = pool.step();
        executed_steps = executed_steps.saturating_add(1);

        if pool.step_index.is_multiple_of(snapshot_period) {
            let snapshot = pool.snapshot(pool.step_index, control_output);
            match supervisor.tx_snapshot.try_send(snapshot) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {}
                Err(TrySendError::Disconnected(_)) => break,
            }
        }

        if step_interval_ns != 0 {
            thread::sleep(Duration::from_nanos(step_interval_ns));
        }
    }

    executed_steps
}

fn run_supervisor_steps_with_flag(
    n_neurons: usize,
    seed: u64,
    snapshot_period: u64,
    step_interval_ns: u64,
    core_snn: usize,
    core_z3: usize,
    max_steps: u64,
    safe_shutdown_flag: Arc<AtomicBool>,
) -> Result<u64, SupervisorExecutionError> {
    if n_neurons == 0 {
        return Err(SupervisorExecutionError::InvalidNeuronCount);
    }

    safe_shutdown_flag.store(false, Ordering::Release);

    let (tx_snapshot, rx_snapshot) = bounded::<PetriNetSnapshot>(DEFAULT_SNAPSHOT_CAPACITY);
    let z3_handle = spawn_z3_verification_worker(rx_snapshot, safe_shutdown_flag.clone(), core_z3);

    let pool = LightweightSnnPool::new(n_neurons, seed);
    let executed = {
        let supervisor = SupervisorState {
            safe_shutdown_flag: safe_shutdown_flag.clone(),
            tx_snapshot,
        };
        execute_snn_control_loop(
            pool,
            &supervisor,
            snapshot_period,
            core_snn,
            max_steps,
            step_interval_ns,
        )
    };

    // All snapshot senders are dropped here so the Z3 worker can observe EOF.
    let _ = z3_handle.join();

    if safe_shutdown_flag.load(Ordering::Acquire) {
        return Err(SupervisorExecutionError::SafetyViolation);
    }

    Ok(executed)
}

pub fn run_supervisor_steps(
    n_neurons: usize,
    seed: u64,
    snapshot_period: u64,
    step_interval_ns: u64,
    core_snn: usize,
    core_z3: usize,
    max_steps: u64,
) -> Result<u64, SupervisorExecutionError> {
    run_supervisor_steps_with_flag(
        n_neurons,
        seed,
        snapshot_period,
        step_interval_ns,
        core_snn,
        core_z3,
        max_steps,
        Arc::new(AtomicBool::new(false)),
    )
}

#[pyclass(
    name = "PySpikingControllerPool",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PySpikingControllerPool {
    n_neurons: usize,
    seed: u64,
    snapshot_period: u64,
    step_interval_ns: u64,
    safe_shutdown_flag: Arc<AtomicBool>,
}

#[pymethods]
impl PySpikingControllerPool {
    #[new]
    #[pyo3(signature = (n_neurons=64, seed=7, step_interval_ns=DEFAULT_STEP_INTERVAL_NS, snapshot_period=DEFAULT_SNAPSHOT_PERIOD))]
    fn new(
        n_neurons: usize,
        seed: u64,
        step_interval_ns: u64,
        snapshot_period: u64,
    ) -> PyResult<Self> {
        if n_neurons == 0 {
            return Err(PyRuntimeError::new_err("n_neurons must be > 0."));
        }

        Ok(Self {
            n_neurons,
            seed,
            snapshot_period,
            step_interval_ns,
            safe_shutdown_flag: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Start native supervisor execution.
    ///
    /// * `core_snn`: Core affinity target for the real-time SNN lane.
    /// * `core_z3`: Core affinity target for the verifier lane.
    /// * `max_steps`: Hard runtime limit (0 for no hard limit).
    #[pyo3(signature = (core_snn=1, core_z3=2, max_steps=0))]
    fn start(&self, core_snn: usize, core_z3: usize, max_steps: usize) -> PyResult<usize> {
        match run_supervisor_steps_with_flag(
            self.n_neurons,
            self.seed,
            self.snapshot_period,
            self.step_interval_ns,
            core_snn,
            core_z3,
            max_steps as u64,
            self.safe_shutdown_flag.clone(),
        ) {
            Ok(executed) => Ok(executed as usize),
            Err(SupervisorExecutionError::SafetyViolation) => Err(PyRuntimeError::new_err(
                "Hardware execution terminated: safety contract violation detected by Z3 worker.",
            )),
            Err(SupervisorExecutionError::InvalidNeuronCount) => Err(PyRuntimeError::new_err(
                SupervisorExecutionError::InvalidNeuronCount.to_string(),
            )),
        }
    }

    fn is_safety_tripped(&self) -> bool {
        self.safe_shutdown_flag.load(Ordering::Acquire)
    }

    fn force_shutdown(&self) {
        self.safe_shutdown_flag.store(true, Ordering::Release);
    }
}
