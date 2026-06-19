import time
import numpy as np

from sc_neurocore.evo_substrate.evo_substrate import (
    ReplicationEngine,
    Genome,
    OrganismEmitter,
    MutationEngine,
    MutationConfig,
)


def mock_fpga_runner(genome: Genome) -> dict:
    """Simulates hardware execution of an SC organism."""
    # Compute accuracy as a function of optimal neuron / topology mapping.
    # An ideal network in this scenario wants exactly 64 neurons and high tau_fast.
    n_error = abs(genome.topology.num_neurons - 64)
    tau_bonus = min(20.0, genome.neuron.tau_fast) / 20.0

    accuracy = 1.0 - (n_error / 1024.0) + (0.1 * tau_bonus)
    accuracy = float(np.clip(accuracy, 0.1, 0.99))

    return {
        "accuracy": accuracy,
        "energy_mw": genome.topology.num_neurons * 1.5,
        "latency_ms": genome.topology.num_layers * 0.5,
    }


def run_evolution():
    print("==================================================================")
    print(" SC-NEUROCORE EVOLUTIONARY SUBSTRATE : EXPERIMENTAL DEMO")
    print("==================================================================\n")

    mut_config = MutationConfig(point_rate=0.4, structural_rate=0.2, duplication_rate=0.05)
    mutator = MutationEngine(config=mut_config, rng_seed=42)

    # Enable industrial features (FormalSafetyGuard, Extinction, Tournament)
    engine = ReplicationEngine(
        mutation_engine=mutator, max_population=20, elitism=2, industrial_mode=True
    )

    # 1. Seed Population
    print("[1] Seeding ancient primordial ArcaneZenith organisms...")
    for i in range(10):
        g = Genome()
        g.topology.num_neurons = 4 + i
        g.topology.bitstream_length = 64
        engine.seed(g)

    generations = 15
    print(f"\n[2] Evolving for {generations} generations across simulated FPGA targets...")

    for gen in range(generations):
        stats = engine.evolve_generation(mock_fpga_runner)
        best = stats["best_fitness"]
        mean = stats["mean_fitness"]
        div = stats["diversity"]
        ext = stats.get("extinctions", 0)

        warn = " [MASS EXTINCTION TRIGGERED]" if ext > 0 else ""
        print(
            f"   Gen {gen + 1:02d} | Pop: {stats['population_size']:02d} | Best Fit: {best:.4f} | Mean Fit: {mean:.4f} | Diversity: {div:.2f}{warn}"
        )
        time.sleep(0.05)

    # 3. Harvest Champion
    champion = engine.best_organism
    if not champion:
        return

    print("\n[3] Harvesting Champion Organism")
    print(f"   ID: {champion.genome.genome_id}")
    print(f"   Fitness: {champion.fitness.composite:.4f}")
    print(f"   Neurons: {champion.genome.topology.num_neurons}")
    print(f"   Tau Fast: {champion.genome.neuron.tau_fast:.2f}")

    # 4. Synthesize Physical Artifacts
    print("\n[4] Emitting Synthesizable Verilog RTL...")
    v_code = OrganismEmitter.to_verilog(champion.genome, module_name="evolved_zenith_core")

    for line in v_code.split("\n")[:15]:
        print(f"   {line}")
    print("   ... (truncated)")

    print("\n[5] Biological / Photonic Bridges Enabled")
    print("   → Ready to bridge Bioware (MEA signals as fitness)")
    print("   → Ready to bridge Photonic Netlists (`pc.emit_from_genome(champion)`)")

    print("\nEvolution complete.")


if __name__ == "__main__":
    run_evolution()
