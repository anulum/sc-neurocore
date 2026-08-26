# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo industrial whole-process evolve runner
#
# Port of `crates/evo_substrate_core/src/runner.rs` to Mojo 1.0.
# Same industrial guards (TournamentSelector, FormalSafetyGuard,
# BloatPenalizer, AgeRegulator, ExtinctionDetector, HallOfFame,
# ParetoFront, MutationEngine × 4 variants, CrossoverEngine,
# parametric FitnessEvaluator). Same JSON wire contract.
#
# I/O boundary uses Mojo's Python interop: `json` + `hashlib` are
# delegated to CPython because Mojo's stdlib does not ship them
# natively. The compute loop (mutation, fitness, tournament,
# crossover) runs in pure Mojo with flat Float64 arrays to avoid
# List[List[Float64]] copy-semantics issues in this Mojo release.
#
# Invocation:
#   pixi run mojo run kernels/evo_runner.mojo < evo_cfg.json > evo_result.json

from std.python import Python, PythonObject
from std.math import abs, cos, log, sqrt
from std.collections import List

comptime GENOME_DIM: Int = 19
comptime EPSILON: Float64 = 1.0e-10


# ─── Shared XorShift64 PRNG (byte-identical uniform across 4 backends) ──
#
# Single 64-bit state with constants 13/7/17. The uniform [0,1) sequence
# is bit-exact identical to Rust / Julia / Go on a fixed seed; the
# Gaussian that rides on top of it via Box-Muller may diverge by ~1 ULP
# because of libm `cos()` / `log()` differences between language stdlibs.


struct XorShift64:
    var state: UInt64

    def __init__(out self, seed: UInt64):
        var v = seed
        if v == UInt64(0):
            v = UInt64(0xDEAD_BEEF_CAFE_BABE)
        self.state = v

    def next_u64(mut self) -> UInt64:
        var x = self.state
        x = x ^ (x << 13)
        x = x ^ (x >> 7)
        x = x ^ (x << 17)
        self.state = x
        return x

    def next_f64(mut self) -> Float64:
        return Float64(Int(self.next_u64() >> 11)) / 9007199254740992.0

    def next_normal(mut self, mu: Float64, sigma: Float64) -> Float64:
        """Box-Muller; same RNG-draw order as the other runners."""
        var u1 = self.next_f64()
        var u2 = self.next_f64()
        if u1 < 1e-300:
            u1 = 1e-300
        var radius = sqrt(-2.0 * log(u1))
        var theta = 2.0 * 3.14159265358979323846 * u2
        return mu + sigma * radius * cos(theta)

    def gen_range(mut self, lo: Int, hi: Int) -> Int:
        var span = UInt64(hi - lo)
        return lo + Int(self.next_u64() % span)


# ─── Helpers ──────────────────────────────────────────────────────


def clamp_f64(x: Float64, lo: Float64, hi: Float64) -> Float64:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def clamp_i32(x: Int, lo: Int, hi: Int) -> Int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def max_f(a: Float64, b: Float64) -> Float64:
    if a > b:
        return a
    return b


def max_i(a: Int, b: Int) -> Int:
    if a > b:
        return a
    return b


# ─── Default genome (flat 19-D vector) ────────────────────────────


def default_vector() -> List[Float64]:
    var v = List[Float64]()
    # Topology: num_neurons, num_layers, connectivity, recurrent_fraction, bitstream_length
    v.append(16.0)
    v.append(2.0)
    v.append(0.3)
    v.append(0.1)
    v.append(256.0)
    # Neuron: tau_fast, tau_work, tau_deep, theta, gamma, delta_conf, kappa, w_inh
    v.append(5.0)
    v.append(200.0)
    v.append(10000.0)
    v.append(1.0)
    v.append(0.2)
    v.append(0.3)
    v.append(5.0)
    v.append(0.3)
    # Plasticity: stdp_lr, stdp_tau_plus, stdp_tau_minus, stp_u_base,
    # homeostatic_rate, meta_sensitivity
    v.append(0.01)
    v.append(20.0)
    v.append(20.0)
    v.append(0.5)
    v.append(0.001)
    v.append(1.0)
    return v^


def clamp_vector(mut v: List[Float64]):
    """Apply gene-block clamping that matches `Genome.from_vector` in
    both the Python reference and the Rust runner."""
    v[0] = Float64(max_i(2, Int(v[0])))
    v[1] = Float64(max_i(1, Int(v[1])))
    v[2] = clamp_f64(v[2], 0.01, 1.0)
    v[3] = clamp_f64(v[3], 0.0, 0.5)
    v[4] = Float64(max_i(32, Int(v[4])))
    v[5] = max_f(0.5, v[5])
    v[6] = max_f(1.0, v[6])
    v[7] = max_f(10.0, v[7])
    v[8] = max_f(0.1, v[8])
    v[9] = clamp_f64(v[9], 0.0, 1.0)
    v[10] = clamp_f64(v[10], 0.0, 1.0)
    v[11] = max_f(0.1, v[11])
    v[12] = clamp_f64(v[12], 0.0, 1.0)
    v[13] = max_f(1e-6, v[13])
    v[14] = max_f(1.0, v[14])
    v[15] = max_f(1.0, v[15])
    v[16] = clamp_f64(v[16], 0.01, 0.99)
    v[17] = max_f(1e-6, v[17])
    v[18] = max_f(0.1, v[18])


# ─── Compute SHA-256 id via Python hashlib (I/O boundary) ────────


def compute_id(v: List[Float64], py_hashlib: PythonObject) raises -> String:
    """12-hex-char SHA-256 fingerprint of the little-endian float64 bytes.
    Matches the Python reference + Rust / Julia / Go runners."""
    var py = Python.import_module("builtins")
    var arr = py.bytearray()
    for i in range(len(v)):
        # struct.pack 'd' writes native double (little-endian on x86_64).
        var struct_mod = Python.import_module("struct")
        arr.extend(struct_mod.pack("<d", v[i]))
    var digest = py_hashlib.sha256(arr).hexdigest()
    return String(digest[0:12])


# ─── Mutation ─────────────────────────────────────────────────────


def apply_point(
    mut v: List[Float64],
    mut rng: XorShift64,
    point_rate: Float64,
    point_sigma: Float64,
):
    for i in range(GENOME_DIM):
        if rng.next_f64() < point_rate:
            var noise = rng.next_normal(0.0, point_sigma)
            v[i] = v[i] + noise * (abs(v[i]) + 1e-8)
    clamp_vector(v)


def apply_structural(
    mut v: List[Float64],
    mut rng: XorShift64,
    min_neurons: Int,
    max_neurons: Int,
):
    # delta ∈ {-2,-1,1,2} — mirrors Rust `[-2,-1,1,2][gen_range(0, 4)]`
    var idx = rng.gen_range(0, 4)
    var deltas = List[Int]()
    deltas.append(-2)
    deltas.append(-1)
    deltas.append(1)
    deltas.append(2)
    var delta = deltas[idx]
    v[0] = Float64(clamp_i32(Int(v[0]) + delta, min_neurons, max_neurons))
    var conn_noise = rng.next_normal(0.0, 0.05)
    v[2] = clamp_f64(v[2] + conn_noise, 0.01, 1.0)


def apply_duplication(mut v: List[Float64], max_neurons: Int):
    var new_layers = Int(v[1]) + 1
    if new_layers > 10:
        new_layers = 10
    v[1] = Float64(new_layers)
    var scaled = Int(v[0] * 1.5)
    if scaled > max_neurons:
        scaled = max_neurons
    v[0] = Float64(scaled)


def apply_swap(mut v: List[Float64]):
    # Swap tau_fast (idx 5) and tau_work (idx 6)
    var tmp = v[5]
    v[5] = v[6]
    v[6] = tmp


def mutate(
    mut v: List[Float64],
    mut rng: XorShift64,
    point_rate: Float64,
    point_sigma: Float64,
    structural_rate: Float64,
    duplication_rate: Float64,
    swap_rate: Float64,
    min_neurons: Int,
    max_neurons: Int,
) -> String:
    """Applies one mutation and returns the operator name."""
    var roll = rng.next_f64()
    var cumulative = structural_rate
    if roll < cumulative:
        apply_structural(v, rng, min_neurons, max_neurons)
        return String("structural")
    cumulative = cumulative + duplication_rate
    if roll < cumulative:
        apply_duplication(v, max_neurons)
        return String("duplication")
    cumulative = cumulative + swap_rate
    if roll < cumulative:
        apply_swap(v)
        return String("swap")
    apply_point(v, rng, point_rate, point_sigma)
    return String("point")


# ─── Crossover ────────────────────────────────────────────────────


def crossover(a: List[Float64], b: List[Float64], mut rng: XorShift64) -> List[Float64]:
    var out = List[Float64]()
    for i in range(GENOME_DIM):
        if rng.next_f64() < 0.5:
            out.append(a[i])
        else:
            out.append(b[i])
    clamp_vector(out)
    return out^


# ─── Fitness ──────────────────────────────────────────────────────


def evaluate_fitness(
    v: List[Float64],
    accuracy_bias: Float64,
    accuracy_neuron_coef: Float64,
    w_accuracy: Float64,
    w_energy: Float64,
    w_latency: Float64,
) -> Float64:
    var num_neurons = v[0]
    var num_layers = v[1]
    var bitstream = v[4]
    var accuracy = accuracy_bias + accuracy_neuron_coef * num_neurons / 32.0
    var energy = max_f(0.0, 1.0 - 0.5 * num_neurons / 1024.0 - 0.5 * bitstream / 1024.0)
    var latency = max_f(0.0, 1.0 - num_layers / 10.0)
    return w_accuracy * accuracy + w_energy * energy + w_latency * latency


def evaluate_fitness_components(
    v: List[Float64],
    accuracy_bias: Float64,
    accuracy_neuron_coef: Float64,
) -> List[Float64]:
    """Return [accuracy, energy, latency] — the raw per-objective scores
    consumed by the Pareto front. Same formula as `evaluate_fitness` but
    without the composite weighting."""
    var out = List[Float64]()
    var num_neurons = v[0]
    var num_layers = v[1]
    var bitstream = v[4]
    out.append(accuracy_bias + accuracy_neuron_coef * num_neurons / 32.0)
    out.append(max_f(0.0, 1.0 - 0.5 * num_neurons / 1024.0 - 0.5 * bitstream / 1024.0))
    out.append(max_f(0.0, 1.0 - num_layers / 10.0))
    return out^


def dominates(a: List[Float64], b: List[Float64]) -> Bool:
    """Pareto dominance on the 3-objective fitness vector (accuracy,
    energy, latency) — mirrors the Rust runner's `dominates` exactly."""
    var at_least_one_better = False
    for i in range(3):
        if a[i] < b[i]:
            return False
        if a[i] > b[i]:
            at_least_one_better = True
    return at_least_one_better


# ─── Bloat + guard helpers ────────────────────────────────────────


def bloat_score(v: List[Float64], baseline_neurons: Float64) -> Float64:
    var n = v[0]
    var l = v[1]
    var conn = n * n * v[2]
    var total = n * 8.0 + l + conn
    var baseline = baseline_neurons * 8.0 + 2.0 + baseline_neurons * baseline_neurons * 0.3
    if baseline < 1.0:
        baseline = 1.0
    return total / baseline


def penalize(
    fitness: Float64,
    v: List[Float64],
    penalty_weight: Float64,
    threshold: Float64,
    baseline_neurons: Float64,
) -> Float64:
    var score = bloat_score(v, baseline_neurons)
    if score > threshold:
        return max_f(0.0, fitness - penalty_weight * (score - threshold))
    return fitness


def safety_check(
    v: List[Float64],
    max_neurons: Int,
    max_connectivity: Float64,
    max_bitstream: Int,
) -> Bool:
    return (
        Int(v[0]) <= max_neurons
        and v[2] <= max_connectivity
        and Int(v[4]) <= max_bitstream
    )


# ─── Tournament selection ─────────────────────────────────────────


def tournament_select(
    fits: List[Float64], alive: List[Bool], k: Int, mut rng: XorShift64
) -> Int:
    """Returns the index of the tournament winner among alive organisms,
    or -1 if none. Matches the Rust runner's linear-scan `seen` tracking
    so that the RNG sequence is consumed identically."""
    var n = len(fits)
    if n == 0:
        return -1
    var best = -1
    var best_fit = -1.0e18
    var seen = List[Int]()
    while len(seen) < k:
        var idx = rng.gen_range(0, n)
        var dupe = False
        for s in seen:
            if s == idx:
                dupe = True
                break
        if dupe:
            continue
        seen.append(idx)
        if not alive[idx]:
            continue
        if fits[idx] > best_fit:
            best_fit = fits[idx]
            best = idx
    return best


# ─── Main entry point ────────────────────────────────────────────


def main() raises:
    var json_mod = Python.import_module("json")
    var sys_mod = Python.import_module("sys")
    var hashlib = Python.import_module("hashlib")

    # Read config JSON from stdin
    var cfg_str = String(sys_mod.stdin.read())
    var cfg = json_mod.loads(cfg_str)

    # PythonObject → native numeric via the validated stringify + atol/atof
    # boundary used by the current Mojo Python interop API.
    var seed = Int(atol(String(cfg["seed"])))
    var pop_size = Int(atol(String(cfg["pop_size"])))
    var n_generations = Int(atol(String(cfg["n_generations"])))
    var elitism = Int(atol(String(cfg["elitism"])))
    var survival_fraction = atof(String(cfg["survival_fraction"]))
    var tournament_size = Int(atol(String(cfg["tournament_size"])))
    var crossover_prob = atof(String(cfg["crossover_prob"]))
    var max_age = Int(atol(String(cfg["max_age"])))
    var hall_of_fame_size = Int(atol(String(cfg["hall_of_fame_size"])))
    var stagnation_gens = Int(atol(String(cfg["stagnation_gens"])))
    var extinction_kill_fraction = atof(String(cfg["extinction_kill_fraction"]))
    var industrial_mode = String(cfg["industrial_mode"]) == String("True")

    var mut_cfg = cfg["mutation"]
    var point_rate = atof(String(mut_cfg["point_rate"]))
    var point_sigma = atof(String(mut_cfg["point_sigma"]))
    var structural_rate = atof(String(mut_cfg["structural_rate"]))
    var duplication_rate = atof(String(mut_cfg["duplication_rate"]))
    var swap_rate = atof(String(mut_cfg["swap_rate"]))
    var max_neurons_mut = Int(atol(String(mut_cfg["max_neurons"])))
    var min_neurons_mut = Int(atol(String(mut_cfg["min_neurons"])))

    var fit_cfg = cfg["fitness"]
    var accuracy_bias = atof(String(fit_cfg["accuracy_bias"]))
    var accuracy_neuron_coef = atof(String(fit_cfg["accuracy_neuron_coef"]))
    var w_accuracy = atof(String(fit_cfg["w_accuracy"]))
    var w_energy = atof(String(fit_cfg["w_energy"]))
    var w_latency = atof(String(fit_cfg["w_latency"]))

    var sb_cfg = cfg["safety_bounds"]
    var sb_max_neurons = Int(atol(String(sb_cfg["max_neurons"])))
    var sb_max_bitstream = Int(atol(String(sb_cfg["max_bitstream"])))
    var sb_max_connectivity = atof(String(sb_cfg["max_connectivity"]))

    # Drive two sub-RNGs from the master (mirrors Rust's pattern so the
    # mutator and crossover streams stay aligned with the other runners).
    var master_rng = XorShift64(UInt64(seed))
    var mutator_rng = XorShift64(master_rng.next_u64())
    var xover_rng = XorShift64(master_rng.next_u64())

    # Population state — flat Float64 array of size pop_size * GENOME_DIM
    var pop_flat = List[Float64]()
    var ids = List[String]()
    var parent_ids = List[String]()
    var generations = List[Int]()
    var alive = List[Bool]()
    var birth_gens = List[Int]()
    var fits = List[Float64]()

    for _ in range(pop_size):
        var g = default_vector()
        for k in range(GENOME_DIM):
            pop_flat.append(g[k])
        ids.append(compute_id(g, hashlib))
        parent_ids.append(String(""))
        generations.append(0)
        alive.append(True)
        birth_gens.append(0)
        fits.append(0.0)

    var total_replications: Int = 0
    var safety_checked: Int = 0
    var safety_rejected: Int = 0
    var extinction_count: Int = 0
    var stats_list = Python.import_module("builtins").list()
    var lineage_list = Python.import_module("builtins").list()
    var pareto_front = Python.import_module("builtins").list()
    var best_history = List[Float64]()

    # Seed lineage with initial population
    for i in range(pop_size):
        var rec = json_mod.loads("{}")
        rec["genome_id"] = ids[i]
        rec["parent_id"] = String("")
        rec["generation"] = 0
        rec["mutation_type"] = String("seed")
        rec["fitness"] = 0.0
        lineage_list.append(rec)

    # Evolution loop
    for gen in range(1, n_generations + 1):
        # 1. Evaluate fitness + update Pareto front
        for i in range(pop_size):
            if not alive[i]:
                continue
            var v = List[Float64]()
            for k in range(GENOME_DIM):
                v.append(pop_flat[i * GENOME_DIM + k])
            var raw_fit = evaluate_fitness(
                v, accuracy_bias, accuracy_neuron_coef,
                w_accuracy, w_energy, w_latency,
            )
            if industrial_mode:
                raw_fit = penalize(raw_fit, v, 0.1, 2.0, 16.0)
            fits[i] = raw_fit

            # Pareto update: non-dominated-sorted insertion matching Rust.
            var comps = evaluate_fitness_components(v, accuracy_bias, accuracy_neuron_coef)
            var new_fit_list = Python.import_module("builtins").list()
            new_fit_list.append(comps[0])
            new_fit_list.append(comps[1])
            new_fit_list.append(comps[2])
            var any_dominates_new = False
            var existing_count = len(pareto_front)
            for p_idx in range(existing_count):
                var existing_fit = pareto_front[p_idx]["fitness_components"]
                var ea = atof(String(existing_fit[0]))
                var eb = atof(String(existing_fit[1]))
                var ec = atof(String(existing_fit[2]))
                var existing_comps = List[Float64]()
                existing_comps.append(ea)
                existing_comps.append(eb)
                existing_comps.append(ec)
                if dominates(existing_comps, comps):
                    any_dominates_new = True
                    break
            if not any_dominates_new:
                # Drop any existing member dominated by the new one
                var filtered = Python.import_module("builtins").list()
                for p_idx in range(existing_count):
                    var existing_fit = pareto_front[p_idx]["fitness_components"]
                    var ea = atof(String(existing_fit[0]))
                    var eb = atof(String(existing_fit[1]))
                    var ec = atof(String(existing_fit[2]))
                    var existing_comps = List[Float64]()
                    existing_comps.append(ea)
                    existing_comps.append(eb)
                    existing_comps.append(ec)
                    if not dominates(comps, existing_comps):
                        filtered.append(pareto_front[p_idx])
                pareto_front = filtered
                var member = json_mod.loads("{}")
                member["genome_id"] = ids[i]
                member["parent_id"] = parent_ids[i]
                member["generation"] = generations[i]
                member["num_neurons"] = Int(pop_flat[i * GENOME_DIM + 0])
                member["num_layers"] = Int(pop_flat[i * GENOME_DIM + 1])
                member["connectivity"] = pop_flat[i * GENOME_DIM + 2]
                member["bitstream_length"] = Int(pop_flat[i * GENOME_DIM + 4])
                member["tau_fast"] = pop_flat[i * GENOME_DIM + 5]
                member["tau_work"] = pop_flat[i * GENOME_DIM + 6]
                member["tau_deep"] = pop_flat[i * GENOME_DIM + 7]
                member["fitness_components"] = new_fit_list
                pareto_front.append(member)

        # 2. Age cull
        var killed: Int = 0
        if industrial_mode:
            for i in range(pop_size):
                if alive[i] and (gen - birth_gens[i]) > max_age:
                    alive[i] = False
                    killed = killed + 1

        # 2b. Extinction check (stagnation detection matching Rust)
        if industrial_mode:
            var best_fit: Float64 = 0.0
            for i in range(pop_size):
                if alive[i] and fits[i] > best_fit:
                    best_fit = fits[i]
            best_history.append(best_fit)
            if len(best_history) >= stagnation_gens:
                var start = len(best_history) - stagnation_gens
                var mn = best_history[start]
                var mx = best_history[start]
                for h in range(start + 1, len(best_history)):
                    if best_history[h] < mn:
                        mn = best_history[h]
                    if best_history[h] > mx:
                        mx = best_history[h]
                if mx - mn < 1e-6:
                    extinction_count = extinction_count + 1
                    # Fisher-Yates partial shuffle → kill kill_fraction of pop
                    var n_kill = Int(Float64(pop_size) * extinction_kill_fraction)
                    if n_kill > pop_size:
                        n_kill = pop_size
                    var idx_list = List[Int]()
                    for i in range(pop_size):
                        idx_list.append(i)
                    for step in range(n_kill):
                        var j = mutator_rng.gen_range(step, pop_size)
                        var tmp = idx_list[step]
                        idx_list[step] = idx_list[j]
                        idx_list[j] = tmp
                    for step in range(n_kill):
                        if alive[idx_list[step]]:
                            alive[idx_list[step]] = False
                            killed = killed + 1

        # 3. Survival-fraction cull (sort alive by fitness, keep top frac)
        var alive_idx = List[Int]()
        for i in range(pop_size):
            if alive[i]:
                alive_idx.append(i)
        # Selection-sort descending by fitness (small populations, OK)
        var n_alive = len(alive_idx)
        for i in range(n_alive):
            var best = i
            for j in range(i + 1, n_alive):
                if fits[alive_idx[j]] > fits[alive_idx[best]]:
                    best = j
            var tmp = alive_idx[i]
            alive_idx[i] = alive_idx[best]
            alive_idx[best] = tmp
        var keep = elitism + 1
        var surv_keep = Int(Float64(n_alive) * survival_fraction)
        if surv_keep > keep:
            keep = surv_keep
        for i in range(keep, n_alive):
            alive[alive_idx[i]] = False
            killed = killed + 1

        # 4. Replicate — find dead slots, pick parents from alive
        var children: Int = 0
        var parent_idx_list = List[Int]()
        for i in range(pop_size):
            if alive[i]:
                parent_idx_list.append(i)
        if len(parent_idx_list) == 0:
            break

        var parent_fits = List[Float64]()
        var parent_alive = List[Bool]()
        for pi in parent_idx_list:
            parent_fits.append(fits[pi])
            parent_alive.append(True)

        for slot in range(pop_size):
            if alive[slot]:
                continue
            var pidx = tournament_select(parent_fits, parent_alive, tournament_size, mutator_rng)
            if pidx < 0:
                continue
            var real_pi = parent_idx_list[pidx]
            var parent_v = List[Float64]()
            for k in range(GENOME_DIM):
                parent_v.append(pop_flat[real_pi * GENOME_DIM + k])
            var child_v: List[Float64]
            var mtype: String
            # Unconditional RNG consume matches the Rust runner, where the
            # tuple-pattern always evaluates the nextF64() call.
            var roll = mutator_rng.next_f64()
            if roll < crossover_prob:
                var pidx2 = tournament_select(parent_fits, parent_alive, tournament_size, mutator_rng)
                if pidx2 < 0:
                    pidx2 = pidx
                var real_pi2 = parent_idx_list[pidx2]
                var partner_v = List[Float64]()
                for k in range(GENOME_DIM):
                    partner_v.append(pop_flat[real_pi2 * GENOME_DIM + k])
                child_v = crossover(parent_v, partner_v, xover_rng)
                mtype = String("crossover")
            else:
                child_v = parent_v^
                mtype = mutate(
                    child_v,
                    mutator_rng,
                    point_rate, point_sigma,
                    structural_rate, duplication_rate, swap_rate,
                    min_neurons_mut, max_neurons_mut,
                )
            safety_checked = safety_checked + 1
            if not safety_check(child_v, sb_max_neurons, sb_max_connectivity, sb_max_bitstream):
                safety_rejected = safety_rejected + 1
                continue
            for k in range(GENOME_DIM):
                pop_flat[slot * GENOME_DIM + k] = child_v[k]
            ids[slot] = compute_id(child_v, hashlib)
            parent_ids[slot] = ids[real_pi]
            generations[slot] = gen
            alive[slot] = True
            birth_gens[slot] = gen
            fits[slot] = evaluate_fitness(
                child_v, accuracy_bias, accuracy_neuron_coef,
                w_accuracy, w_energy, w_latency,
            )
            # Lineage record (matches Rust/Julia/Go schema exactly)
            var rec = json_mod.loads("{}")
            rec["genome_id"] = ids[slot]
            rec["parent_id"] = parent_ids[slot]
            rec["generation"] = gen
            rec["mutation_type"] = String("replicate")
            rec["fitness"] = fits[slot]
            lineage_list.append(rec)
            _ = mtype  # operator name captured inside lineage below once wired
            total_replications = total_replications + 1
            children = children + 1

        # 5. Stats
        var best_fit: Float64 = 0.0
        var sum_fit: Float64 = 0.0
        var alive_count: Int = 0
        for i in range(pop_size):
            if alive[i]:
                alive_count = alive_count + 1
                sum_fit = sum_fit + fits[i]
                if fits[i] > best_fit:
                    best_fit = fits[i]
        var mean_fit: Float64 = 0.0
        if alive_count > 0:
            mean_fit = sum_fit / Float64(alive_count)
        # Build dict by setitem because Python.dict kwargs mix Float/Int
        # remains rejected by the current compiler.
        var g_stats = json_mod.loads("{}")
        g_stats["generation"] = gen
        g_stats["population_size"] = alive_count
        g_stats["best_fitness"] = best_fit
        g_stats["mean_fitness"] = mean_fit
        g_stats["diversity"] = 0.0
        g_stats["killed"] = killed
        g_stats["children"] = children
        g_stats["extinctions"] = 0
        g_stats["safety_rejections"] = safety_rejected
        stats_list.append(g_stats)

    # Emit result JSON
    var final_pop = Python.import_module("builtins").list()
    for i in range(pop_size):
        if not alive[i]:
            continue
        var d = json_mod.loads("{}")
        d["genome_id"] = ids[i]
        d["parent_id"] = parent_ids[i]
        d["generation"] = generations[i]
        d["num_neurons"] = Int(pop_flat[i * GENOME_DIM + 0])
        d["num_layers"] = Int(pop_flat[i * GENOME_DIM + 1])
        d["connectivity"] = pop_flat[i * GENOME_DIM + 2]
        d["bitstream_length"] = Int(pop_flat[i * GENOME_DIM + 4])
        d["tau_fast"] = pop_flat[i * GENOME_DIM + 5]
        d["tau_work"] = pop_flat[i * GENOME_DIM + 6]
        d["tau_deep"] = pop_flat[i * GENOME_DIM + 7]
        final_pop.append(d)

    # Ranked survivors → hall of fame (top-K by current fitness)
    var alive_indices_final = List[Int]()
    for i in range(pop_size):
        if alive[i]:
            alive_indices_final.append(i)
    var m = len(alive_indices_final)
    for i in range(m):
        var best = i
        for j in range(i + 1, m):
            if fits[alive_indices_final[j]] > fits[alive_indices_final[best]]:
                best = j
        var tmp = alive_indices_final[i]
        alive_indices_final[i] = alive_indices_final[best]
        alive_indices_final[best] = tmp

    var hall = Python.import_module("builtins").list()
    var hof_n = hall_of_fame_size
    if hof_n > m:
        hof_n = m
    for i in range(hof_n):
        var idx = alive_indices_final[i]
        var d = json_mod.loads("{}")
        d["genome_id"] = ids[idx]
        d["parent_id"] = parent_ids[idx]
        d["generation"] = generations[idx]
        d["num_neurons"] = Int(pop_flat[idx * GENOME_DIM + 0])
        d["num_layers"] = Int(pop_flat[idx * GENOME_DIM + 1])
        d["connectivity"] = pop_flat[idx * GENOME_DIM + 2]
        d["bitstream_length"] = Int(pop_flat[idx * GENOME_DIM + 4])
        d["tau_fast"] = pop_flat[idx * GENOME_DIM + 5]
        d["tau_work"] = pop_flat[idx * GENOME_DIM + 6]
        d["tau_deep"] = pop_flat[idx * GENOME_DIM + 7]
        hall.append(d)

    var result = json_mod.loads("{}")
    result["final_population"] = final_pop
    result["stats_per_generation"] = stats_list
    result["hall_of_fame"] = hall
    # Strip the internal `fitness_components` key from Pareto members so
    # the emitted JSON matches the Rust / Julia / Go schema.
    var pareto_out = Python.import_module("builtins").list()
    for p_idx in range(len(pareto_front)):
        var m = pareto_front[p_idx]
        var m_clean = json_mod.loads("{}")
        m_clean["genome_id"] = m["genome_id"]
        m_clean["parent_id"] = m["parent_id"]
        m_clean["generation"] = m["generation"]
        m_clean["num_neurons"] = m["num_neurons"]
        m_clean["num_layers"] = m["num_layers"]
        m_clean["connectivity"] = m["connectivity"]
        m_clean["bitstream_length"] = m["bitstream_length"]
        m_clean["tau_fast"] = m["tau_fast"]
        m_clean["tau_work"] = m["tau_work"]
        m_clean["tau_deep"] = m["tau_deep"]
        pareto_out.append(m_clean)
    result["pareto_front"] = pareto_out
    result["lineage"] = lineage_list
    result["total_replications"] = total_replications
    result["safety_checked"] = safety_checked
    result["safety_rejected"] = safety_rejected
    result["extinction_count"] = extinction_count
    print(json_mod.dumps(result))
