# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for quantum cognition

import std.math
from std.time import perf_counter_ns

# ─── Spin Pool MPS ───

struct QuantumSpinChainMPS:
    """High-performance spin-pool telemetry kernel."""
    var sites: Int
    var bond_dim: Int
    var correlation_length: Float64
    var update_rate: Float64
    var entanglement_map: List[Float64]
    var measurement_count: Int

    fn __init__(out self, sites: Int, bond_dim: Int):
        self.sites = sites
        self.bond_dim = bond_dim
        self.correlation_length = 2.0
        self.update_rate = 0.1
        self.measurement_count = 0
        self.entanglement_map = List[Float64]()
        var uniform_val = 1.0 / Float64(sites)
        for _ in range(sites):
            self.entanglement_map.append(uniform_val)

    fn apply_measurement(mut self, site_idx: Int, intensity: Float64):
        var alpha = self.update_rate
        var one_minus_alpha = 1.0 - alpha
        var total: Float64 = 0.0
        for i in range(self.sites):
            var distance = Float64(abs(i - site_idx))
            var influence = std.math.exp(-distance / self.correlation_length) * intensity
            self.entanglement_map[i] = one_minus_alpha * self.entanglement_map[i] + alpha * influence
            total += self.entanglement_map[i]
        if total > 0.0:
            var inv_total = 1.0 / total
            for i in range(self.sites):
                self.entanglement_map[i] *= inv_total
        self.measurement_count += 1

    fn get_local_atp_telemetry(self, site_idx: Int) -> Float64:
        var value = self.entanglement_map[site_idx]
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    fn get_avg_entanglement(self) -> Float64:
        var total: Float64 = 0.0
        for i in range(self.sites):
            total += self.entanglement_map[i]
        return total / Float64(self.sites)

    fn reset(mut self):
        var uniform = 1.0 / Float64(self.sites)
        for i in range(self.sites):
            self.entanglement_map[i] = uniform
        self.measurement_count = 0


# ─── Population step using SoA (struct-of-arrays) ───
# Mojo 0.26.2 List requires Copyable; we use parallel arrays instead.

fn batch_step_population_soa(
    mut Vm: List[Float64],
    mut atp: List[Float64],
    mut spike_counts: List[Int],
    mut pool: QuantumSpinChainMPS,
    currents: List[Float64],
    n_neurons: Int,
    v_threshold: Float64,
    v_reset: Float64,
    v_rest: Float64,
    tau_m: Float64,
    atp_consumption: Float64,
) -> Int:
    """Step all neurons using SoA layout for SIMD-friendly access."""
    var total_spikes: Int = 0
    for i in range(n_neurons):
        var eff = pool.get_local_atp_telemetry(i)
        atp[i] = min(1.0, atp[i] + eff * 0.01)
        var i_pump = (eff - 0.5) * 2.0 * atp[i]
        var dv = (-(Vm[i] - v_rest) + currents[i] + i_pump) / tau_m
        Vm[i] += dv
        if Vm[i] >= v_threshold:
            if atp[i] >= atp_consumption:
                Vm[i] = v_reset
                atp[i] -= atp_consumption
                spike_counts[i] += 1
                total_spikes += 1
                pool.apply_measurement(i, 1.0)
            else:
                Vm[i] = v_threshold - 1.0
    return total_spikes


fn benchmark_spin_only(sites: Int, n_steps: Int) -> Float64:
    var chain = QuantumSpinChainMPS(sites, 16)
    for step in range(n_steps):
        chain.apply_measurement(step % sites, 1.0)
    return chain.get_avg_entanglement()

fn benchmark_population_soa(n_neurons: Int, n_steps: Int) -> Int:
    var pool = QuantumSpinChainMPS(n_neurons, 16)
    var Vm = List[Float64]()
    var atp = List[Float64]()
    var spk = List[Int]()
    var cur = List[Float64]()
    for i in range(n_neurons):
        Vm.append(-70.0)
        atp.append(1.0)
        spk.append(0)
        cur.append(25.0)
    var total: Int = 0
    for step in range(n_steps):
        for i in range(n_neurons):
            cur[i] = 20.0 + 10.0 * std.math.sin(Float64(step * 7 + i * 3) * 0.01)
        total += batch_step_population_soa(Vm, atp, spk, pool, cur, n_neurons, -50.0, -70.0, -70.0, 20.0, 0.05)
    return total


fn main():
    print("SC-NeuroCore Quantum Cognition — Mojo Benchmark Suite")
    print("=====================================================")

    # Functional test
    var pool = QuantumSpinChainMPS(8, 16)
    pool.apply_measurement(0, 1.0)
    var eff_near = pool.get_local_atp_telemetry(1)
    var eff_far = pool.get_local_atp_telemetry(7)
    print("Non-locality:", "PASS" if eff_near > eff_far else "FAIL")

    # SoA population test
    var Vm = List[Float64]()
    var atp = List[Float64]()
    var spk = List[Int]()
    var cur = List[Float64]()
    for i in range(8):
        Vm.append(-70.0)
        atp.append(1.0)
        spk.append(0)
        cur.append(50.0)
    var pool2 = QuantumSpinChainMPS(8, 16)
    var s = batch_step_population_soa(Vm, atp, spk, pool2, cur, 8, -50.0, -70.0, -70.0, 20.0, 0.05)
    print("Population SoA (8 neurons): spikes =", s)

    # Benchmark 1: Spin pool only
    print("\n--- Benchmark 1: apply_measurement ---")
    var bench_sites = List[Int]()
    bench_sites.append(32)
    bench_sites.append(128)
    bench_sites.append(256)
    for sites in bench_sites:
        var t0 = perf_counter_ns()
        var r = benchmark_spin_only(sites, 10000)
        var ns = perf_counter_ns() - t0
        print("  sites=", sites, " time=", Float64(ns) / 1e6, "ms  per_call=", Float64(ns) / 10000.0 / 1000.0, "µs")

    # Benchmark 2: Population SoA
    print("\n--- Benchmark 2: batch_step_population_soa ---")
    var bench_n = List[Int]()
    bench_n.append(32)
    bench_n.append(128)
    bench_n.append(256)
    for nn in bench_n:
        var t0 = perf_counter_ns()
        var ts = benchmark_population_soa(nn, 1000)
        var ns = perf_counter_ns() - t0
        var total_ns = nn * 1000
        print("  neurons=", nn, " time=", Float64(ns) / 1e6, "ms  per_neuron_step=", Float64(ns) / Float64(total_ns) / 1000.0, "µs  spikes=", ts)

    print("\nMojo kernel: ALL BENCHMARKS COMPLETE")
