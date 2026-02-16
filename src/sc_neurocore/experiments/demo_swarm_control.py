#!/usr/bin/env python3
"""
Neuromorphic Swarm Control Demo
================================

End-to-end demo: evolve SNN-controlled swarm agents, then run the
best policy and visualize emergent behavior.

Usage:
    python -m sc_neurocore.experiments.demo_swarm_control

Author: Claude (Session 2026-02-16)
"""

from __future__ import annotations

import sys

import numpy as np

from ..swarm import (
    SwarmAgent,
    SwarmEnvironment,
    CollectiveFields,
    SwarmCommunication,
    SwarmFitness,
    SwarmEvolver,
)
from ..swarm.swarm_env import EnvConfig
from ..swarm.collective_fields import FieldConfig
from ..swarm.neuroevolution_swarm import EvolverConfig
from ..swarm.agent import AgentConfig


def run_demo():
    """Run evolution + evaluation demo."""
    print("=" * 72)
    print("  Neuromorphic Swarm Control Demo")
    print("  SC-NeuroCore — SNN-Controlled Robot Swarm")
    print("=" * 72)

    # Phase 1: Evolution
    print("\n--- Phase 1: Neuroevolution ---")
    print(f"  Population: 8 | Generations: 5 | Ticks/eval: 100")
    print(f"  Agents: 10 | Arena: 80x80 | Obstacles: 3 | Targets: 2\n")

    evolver = SwarmEvolver(EvolverConfig(
        population_size=8,
        ticks_per_eval=100,
        mutation_rate=0.08,
        elite_fraction=0.25,
        env_config=EnvConfig(
            n_agents=10,
            width=80.0,
            height=80.0,
            n_obstacles=3,
            n_targets=2,
            seed=42,
        ),
        agent_config=AgentConfig(n_hidden=12),
        seed=42,
    ))

    def on_gen(gen, fitness, info):
        bar = "█" * int(fitness * 40) + "░" * (40 - int(fitness * 40))
        print(f"  Gen {gen:3d} | Best: {fitness:.4f} | Mean: {info['mean_fitness']:.4f} | {bar}")

    best_weights = evolver.evolve(generations=5, callback=on_gen)
    print(f"\n  Best fitness achieved: {evolver.best_fitness:.4f}")

    # Phase 2: Run best policy
    print("\n--- Phase 2: Best Policy Evaluation ---")

    env_cfg = EnvConfig(n_agents=10, width=80, height=80, n_obstacles=3, n_targets=2, seed=99)
    env = SwarmEnvironment(config=env_cfg)
    fields = CollectiveFields(FieldConfig(arena_width=80, arena_height=80))
    comm = SwarmCommunication(env, fields)
    fitness_eval = SwarmFitness()

    # Load best weights into all agents
    for agent in env.agents:
        agent.weights = best_weights.copy()
        agent.reset_neural_state()

    print(f"  Running 200 ticks with evolved policy...\n")
    print(f"  {'Tick':>5} | {'Cov':>5} | {'Coh':>5} | {'Tgt':>5} | {'Composite':>9}")
    print(f"  {'─' * 5} | {'─' * 5} | {'─' * 5} | {'─' * 5} | {'─' * 9}")

    for tick in range(200):
        for i, agent in enumerate(env.agents):
            neighbor_d = env.get_neighbor_distances(i)
            obstacle_d = env.get_obstacle_distances(i)
            target_d = env.get_target_distances(i)
            comm_data = comm.get_sensory_data(i)

            sensory = agent.sense(
                neighbor_dists=neighbor_d,
                obstacle_dists=obstacle_d,
                target_dists=target_d,
                chem_gradient=comm_data["chem_gradient"],
                symbolic_value=comm_data["symbolic_value"],
            )
            motor = agent.think(sensory)
            agent.act(motor[0], motor[1])

        comm.step()
        env.step()

        if tick % 40 == 0 or tick == 199:
            bd = fitness_eval.get_breakdown(env)
            print(
                f"  {tick:5d} | {bd['coverage']:.3f} | {bd['cohesion']:.3f} | "
                f"{bd['target_reach']:.3f} | {bd['composite']:.5f}"
            )

    final = fitness_eval.get_breakdown(env)
    print(f"\n  Final Fitness Breakdown:")
    for k, v in final.items():
        print(f"    {k:20s}: {v:.4f}")

    # Phase 3: Agent neural state summary
    print("\n--- Phase 3: Agent Neural State Summary ---")
    for agent in env.agents[:3]:
        state = agent.get_neural_state()
        print(
            f"  Agent {state['agent_id']:2d} | "
            f"pos=({state['position'][0]:.1f}, {state['position'][1]:.1f}) | "
            f"activity={state['mean_activity']:.3f} | "
            f"chem={state['chemical_output']:.4f}"
        )

    print(f"\n{'=' * 72}")
    print("  Demo complete.")
    print("=" * 72)


if __name__ == "__main__":
    run_demo()
