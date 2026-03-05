# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

#!/usr/bin/env python3
"""
Neuromorphic Swarm Control Demo
================================

Evolves spiking-neural-network agents via genetic algorithm, then
visualises the best swarm in a 100x100 arena with obstacles and targets.

Usage:
    python -m sc_neurocore.experiments.demo_swarm_control

"""


import numpy as np

from sc_neurocore.swarm import (
    AgentConfig,
    EnvConfig,
    EvolverConfig,
    SwarmAgent,
    SwarmEnvironment,
    SwarmEvolver,
    SwarmFitness,
)


def _bar(value: float, width: int = 30) -> str:
    """Tiny ASCII bar chart helper."""
    filled = int(np.clip(value, 0, 1) * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def run_demo() -> None:
    print("=" * 64)
    print("  Neuromorphic Swarm Control -- Neuroevolution Demo")
    print("  SNN agents (soft-LIF) evolved via genetic algorithm")
    print("=" * 64)

    # --- Configuration ---
    agent_cfg = AgentConfig(n_sensory=20, n_hidden=16, n_motor=2, seed=42)
    env_cfg = EnvConfig(
        width=100,
        height=100,
        n_agents=10,
        n_obstacles=3,
        n_targets=3,
        agent_config=agent_cfg,
        seed=7,
    )
    evolver_cfg = EvolverConfig(
        pop_size=12,
        n_elite=3,
        mutation_rate=0.15,
        mutation_std=0.3,
        n_eval_steps=100,
        agent_config=agent_cfg,
        env_config=env_cfg,
        seed=123,
    )

    evolver = SwarmEvolver(evolver_cfg)

    print(f"\n  Population : {evolver_cfg.pop_size}")
    print(f"  Weights/ind: {evolver.n_weights}")
    print(f"  Eval steps : {evolver_cfg.n_eval_steps}")
    print(f"  Agents     : {env_cfg.n_agents}")
    print(f"  Obstacles  : {env_cfg.n_obstacles}")
    print(f"  Targets    : {env_cfg.n_targets}")

    # --- Evolution ---
    n_gen = 5
    print(f"\n  Evolving {n_gen} generations ...\n")
    for g in range(n_gen):
        best = evolver.evolve_generation()
        print(f"    Gen {g + 1:2d}  best fitness = {best:.4f}  {_bar(best)}")

    hist = evolver.best_fitness_history
    delta = hist[-1] - hist[0]
    sign = "+" if delta >= 0 else ""
    print(f"\n  Improvement over {n_gen} gens: {sign}{delta:.4f}")

    # --- Replay best individual ---
    print("\n" + "-" * 64)
    print("  Replaying best agent for 100 steps")
    print("-" * 64)

    best_weights = evolver.get_best_weights()
    replay_env = SwarmEnvironment(
        EnvConfig(
            width=100,
            height=100,
            n_agents=10,
            n_obstacles=3,
            n_targets=3,
            agent_config=agent_cfg,
            seed=999,
        )
    )
    for agent in replay_env.agents:
        agent.weights = best_weights

    for step in range(100):
        replay_env.step(dt=1.0)
        if step % 20 == 0:
            positions = replay_env.get_positions()
            cx = positions[:, 0].mean()
            cy = positions[:, 1].mean()
            spread = np.sqrt(((positions - positions.mean(axis=0)) ** 2).sum(axis=-1)).mean()
            fitness = SwarmFitness.composite(replay_env)
            print(
                f"    step {step:3d}  centroid=({cx:5.1f},{cy:5.1f})  "
                f"spread={spread:5.1f}  fitness={fitness:.3f}"
            )

    # --- Final state ---
    state = replay_env.get_state()
    print(f"\n  Targets captured : {state['targets_captured']}")
    print(f"  Final positions:")
    for i, (x, y) in enumerate(state["positions"]):
        print(f"    Agent {i:2d}: ({x:6.1f}, {y:6.1f})")

    print("\n" + "=" * 64)
    print("  Demo complete.")
    print("=" * 64)


if __name__ == "__main__":
    run_demo()
