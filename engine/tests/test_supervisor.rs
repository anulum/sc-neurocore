// SPDX-License-Identifier: AGPL-3.0-or-later

use crossbeam_channel::bounded;
use sc_neurocore_engine::supervisor::{
    run_supervisor_steps, spawn_z3_verification_worker, verify_bounds_at_depth, PetriNetSnapshot,
    SupervisorExecutionError,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

fn snapshot(markings: [i64; 4]) -> PetriNetSnapshot {
    PetriNetSnapshot {
        step_index: 0,
        active_markings: markings.to_vec(),
        transition_rates: vec![0.0; 3],
    }
}

#[test]
fn verifier_accepts_bounded_transfer_net_snapshot() {
    let bounded = snapshot([1, 1, 1, 0]);

    assert!(verify_bounds_at_depth(&bounded, 4));
}

#[test]
fn verifier_rejects_reachable_error_sink_overflow() {
    let unsafe_snapshot = snapshot([0, 0, 0, 101]);

    assert!(!verify_bounds_at_depth(&unsafe_snapshot, 4));
}

#[test]
fn worker_sets_shutdown_flag_for_unsafe_snapshot() {
    let (tx, rx) = bounded(1);
    let shutdown = Arc::new(AtomicBool::new(false));
    let worker = spawn_z3_verification_worker(rx, shutdown.clone(), 0);

    tx.send(snapshot([0, 0, 0, 101]))
        .expect("test channel should accept unsafe snapshot");
    drop(tx);
    worker
        .join()
        .expect("Z3 worker should exit after violation");

    assert!(shutdown.load(Ordering::Acquire));
}

#[test]
fn bounded_supervisor_run_returns_exact_step_count() {
    let executed = run_supervisor_steps(8, 7, 1, 0, 0, 1, 2)
        .expect("bounded supervisor run should stay inside Z3 safety envelope");

    assert_eq!(executed, 2);
}

#[test]
fn supervisor_rejects_zero_neuron_execution_request() {
    let err = run_supervisor_steps(0, 7, 1, 0, 0, 1, 2)
        .expect_err("zero-neuron execution must fail closed");

    assert_eq!(err, SupervisorExecutionError::InvalidNeuronCount);
}
