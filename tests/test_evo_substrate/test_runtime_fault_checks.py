# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary substrate runtime fault checks

from __future__ import annotations

from sc_neurocore.evo_substrate.evo_substrate import (
    FitnessResult,
    Genome,
    Organism,
    ReplicationEngine,
    RuntimeFaultConfig,
)
from sc_neurocore.fault_injection import DegradationAction, FaultModel, GracefulDegradationPolicy
from sc_neurocore.stochastic_doctor.diagnostics import StochasticDoctor


def _quiet_policy(
    *,
    warning_affected_ratio: float,
    critical_affected_ratio: float,
) -> GracefulDegradationPolicy:
    return GracefulDegradationPolicy(
        doctor=StochasticDoctor(correlation_threshold=2.0, critical_threshold=3.0),
        warning_affected_ratio=warning_affected_ratio,
        critical_affected_ratio=critical_affected_ratio,
    )


def test_runtime_fault_check_records_nominal_seeded_diagnosis() -> None:
    genome = Genome()
    genome.compute_id()
    organism = Organism(genome=genome)
    engine = ReplicationEngine(
        degradation_policy=_quiet_policy(
            warning_affected_ratio=1.0,
            critical_affected_ratio=1.0,
        )
    )

    check = engine.verify_runtime_faults(
        organism,
        RuntimeFaultConfig(fault_model=FaultModel.BIT_FLIP, ber=0.0, seed_offset=5),
    )

    assert check.action == DegradationAction.NOMINAL.value
    assert check.replay_seed == genome.weight_seed + 5
    assert check.recommended_bitstream_length == genome.topology.bitstream_length
    assert organism.runtime_fault_checks == [check]
    assert check.to_dict()["audit_status"] == "OK"


def test_runtime_fault_check_extends_bitstream_with_deterministic_replay() -> None:
    config = RuntimeFaultConfig(
        fault_model=FaultModel.BIT_FLIP,
        ber=0.05,
        seed_offset=11,
        sample_neurons=2,
    )
    policy = _quiet_policy(warning_affected_ratio=0.001, critical_affected_ratio=1.0)
    first = Organism(genome=Genome())
    second = Organism(genome=Genome())
    first.genome.topology.bitstream_length = 32
    second.genome.topology.bitstream_length = 32
    first.genome.compute_id()
    second.genome.compute_id()

    first_check = ReplicationEngine(degradation_policy=policy).verify_runtime_faults(first, config)
    second_check = ReplicationEngine(degradation_policy=policy).verify_runtime_faults(
        second, config
    )

    assert first_check.action == DegradationAction.EXTEND_BITSTREAM.value
    assert first.genome.topology.bitstream_length == 64
    assert first_check.to_dict() == second_check.to_dict()


def test_evaluate_all_applies_runtime_fault_penalty_before_ranking() -> None:
    config = RuntimeFaultConfig(
        fault_model=FaultModel.BIT_FLIP,
        ber=0.05,
        sample_neurons=2,
        fitness_penalty_on_extend=0.5,
    )
    engine = ReplicationEngine(
        runtime_fault_config=config,
        degradation_policy=_quiet_policy(warning_affected_ratio=0.001, critical_affected_ratio=1.0),
    )
    organism = engine.seed(Genome())

    engine.evaluate_all(lambda _genome: {"accuracy": 1.0})

    assert organism.runtime_fault_checks
    assert organism.runtime_fault_checks[-1].action == DegradationAction.EXTEND_BITSTREAM.value
    assert organism.fitness is not None
    assert organism.fitness.composite < FitnessResult("baseline", accuracy=1.0).compute_composite()
