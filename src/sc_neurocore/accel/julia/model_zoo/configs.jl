# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for model_zoo/configs

module ConfigsAccel

using Statistics, LinearAlgebra

function mnist_classifier(n_hidden)
    lif_params = {
        "tau_mem": 10.0,
        "v_threshold": 1.0,
        "noise_std": 0.0,
        "v_rest": 0.0,
        "v_reset": 0.0,
    }
    inp = Population(StochasticLIFNeuron, 784, label="input", params=lif_params)
    hid = Population(StochasticLIFNeuron, n_hidden, label="hidden", params=lif_params)
    out = Population(StochasticLIFNeuron, 10, label="output", params=lif_params)
    scale_ih = sqrt(2.0 / 784)
    scale_ho = sqrt(2.0 / n_hidden)
    p_ih = Projection(inp, hid, weight=scale_ih * 20.0, probability=0.3)
    p_ho = Projection(hid, out, weight=scale_ho * 20.0, probability=0.5)
    stim = PoissonInput(784, rate_hz=500.0, weight=2.0)
    stim.target = inp
    mon_out = SpikeMonitor(out, label="output_spikes")
    mon_hid = SpikeMonitor(hid, label="hidden_spikes")
    return Network(inp, hid, out, p_ih, p_ho, stim, mon_out, mon_hid, seed=42)
end

function dvs_gesture_classifier(n_classes)
    lif_params = {
        "tau_mem": 10.0,
        "v_threshold": 1.0,
        "noise_std": 0.0,
        "v_rest": 0.0,
        "v_reset": 0.0,
    }
    n_input = 256
    inp = Population(StochasticLIFNeuron, n_input, label="dvs_input", params=lif_params)
    hid = Population(StochasticLIFNeuron, 256, label="dvs_hidden", params=lif_params)
    out = Population(StochasticLIFNeuron, n_classes, label="dvs_output", params=lif_params)
    p_ih = Projection(inp, hid, weight=0.5, probability=0.2)
    p_ho = Projection(hid, out, weight=0.8, probability=0.4)
    stim = PoissonInput(n_input, rate_hz=800.0, weight=2.0)
    stim.target = inp
    mon = SpikeMonitor(out, label="gesture_spikes")
    return Network(inp, hid, out, p_ih, p_ho, stim, mon, seed=42)
end

function shd_speech_classifier()
    lif_params = {
        "tau_mem": 10.0,
        "v_threshold": 1.0,
        "noise_std": 0.0,
        "v_rest": 0.0,
        "v_reset": 0.0,
    }
    rec_params = {
        "tau_mem": 20.0,
        "v_threshold": 1.0,
        "noise_std": 0.0,
        "v_rest": 0.0,
        "v_reset": 0.0,
    }
    inp = Population(StochasticLIFNeuron, 700, label="shd_input", params=lif_params)
    rec = Population(StochasticLIFNeuron, 256, label="shd_recurrent", params=rec_params)
    out = Population(StochasticLIFNeuron, 20, label="shd_output", params=rec_params)
    p_ir = Projection(inp, rec, weight=0.3, probability=0.15)
    p_rr = Projection(rec, rec, weight=0.15, probability=0.1)
    p_ro = Projection(rec, out, weight=0.6, probability=0.3)
    stim = PoissonInput(700, rate_hz=500.0, weight=2.0)
    stim.target = inp
    mon = SpikeMonitor(out, label="shd_output_spikes")
    mon_rec = SpikeMonitor(rec, label="shd_recurrent_spikes")
    return Network(inp, rec, out, p_ir, p_rr, p_ro, stim, mon, mon_rec, seed=42)
end

function brunel_balanced_network(n_exc, n_inh, g, eta)
    n_exc: int = 800,
    n_inh: int = 200,
    g: float = 5.0,
    eta: float = 2.0,
    ) -> Network
    # Brunel 2000, Table 1: tau_m=20ms, V_thr=20mV, V_reset=10mV
    lif_params = {
        "tau_mem": 20.0,
        "v_threshold": 1.0,
        "v_reset": 0.0,
        "v_rest": 0.0,
        "noise_std": 0.0,
    }
    exc = Population(StochasticLIFNeuron, n_exc, label="exc", params=lif_params)
    inh = Population(StochasticLIFNeuron, n_inh, label="inh", params=lif_params)
    # J_E = 0.1 (normalised); J_I = -g * J_E
    j_e = 0.1
    j_i = -g * j_e
    eps = 0.1
    p_ee = Projection(exc, exc, weight=j_e, probability=eps, delay=1.5)
    p_ei = Projection(exc, inh, weight=j_e, probability=eps, delay=1.5)
    p_ie = Projection(inh, exc, weight=j_i, probability=eps, delay=1.5)
    p_ii = Projection(inh, inh, weight=j_i, probability=eps, delay=1.5)
    # External Poisson drive: strong enough to drive threshold crossings
    stim_e = PoissonInput(n_exc, rate_hz=1000.0, weight=2.0)
    stim_e.target = exc
    stim_i = PoissonInput(n_inh, rate_hz=1000.0, weight=2.0)
    stim_i.target = inh
    mon_e = SpikeMonitor(exc, label="exc_spikes")
    mon_i = SpikeMonitor(inh, label="inh_spikes")
    return Network(exc, inh, p_ee, p_ei, p_ie, p_ii, stim_e, stim_i, mon_e, mon_i, seed=42)
end

function cortical_column(n_layers)
    layer_names = ["L23", "L4", "L5", "L6"]
    exc_sizes = [50, 50, 50, 50]
    inh_sizes = [15, 15, 10, 10]
    n_actual = min(length(layer_names), max(1, n_layers))
    exc_sizes = exc_sizes[:n_actual]
    inh_sizes = inh_sizes[:n_actual]
    layer_names = layer_names[:n_actual]
    populations = []
    monitors = []
    projections = []
    for i, name in enumerate(layer_names)
        e = Population(PospischilNeuron, exc_sizes[i], label=f"{name}_E", params={"g_m": 0.07})
        inh_pop = Population(GolombFSNeuron, inh_sizes[i], label=f"{name}_I")
        populations.extend([e, inh_pop])
        monitors = push!(, SpikeMonitor(e, label=f"{name}_E_spikes"))
        monitors = push!(, SpikeMonitor(inh_pop, label=f"{name}_I_spikes"))
    # Intra-layer E->I, I->E, E->E
    for i in 1:length(layer_names)
        e_pop = populations[2 * i]
        i_pop = populations[2 * i + 1]
        projections = push!(, Projection(e_pop, i_pop, weight=2.0, probability=0.2))
        projections = push!(, Projection(i_pop, e_pop, weight=-3.0, probability=0.2))
        projections = push!(, Projection(e_pop, e_pop, weight=1.0, probability=0.1))
    # Inter-layer feedforward: L4->L23, L23->L5, L5->L6
    ff_map = [(1, 0), (0, 2), (2, 3)]
    for src_l, tgt_l in ff_map
        if src_l < length(layer_names) && tgt_l < length(layer_names)
            projections = push!(,
                Projection(
                    populations[2 * src_l], populations[2 * tgt_l], weight=1.5, probability=0.1
                )
            )
    # Thalamic Poisson drive into L4 E
    l4_e = populations[2]
    stim = PoissonInput(l4_e.n, rate_hz=800.0, weight=8.0)
    stim.target = l4_e
    net = Network(seed=42)
    for obj in populations + projections + monitors + [stim]
        net.add(obj)
    return net
end

function central_pattern_generator(n_oscillators)
    pops = []
    mons = []
    stims = []
    for i in 1:n_oscillators
        flex = Population(
            HindmarshRoseNeuron, 5, label=f"cpg{i}_flex", params={"b": 3.0, "r": 0.005, "s": 4.0}
        )
        ext = Population(
            HindmarshRoseNeuron, 5, label=f"cpg{i}_ext", params={"b": 3.0, "r": 0.005, "s": 4.0}
        )
        pops.extend([flex, ext])
        mons = push!(, SpikeMonitor(flex, label=f"cpg{i}_flex_spikes"))
        mons = push!(, SpikeMonitor(ext, label=f"cpg{i}_ext_spikes"))
    projs = []
    for i in 1:n_oscillators
        flex = pops[2 * i]
        ext = pops[2 * i + 1]
        projs = push!(, Projection(flex, ext, weight=-2.0, probability=0.8))
        projs = push!(, Projection(ext, flex, weight=-2.0, probability=0.8))
        next_flex = pops[2 * ((i + 1) % n_oscillators)]
        projs = push!(, Projection(flex, next_flex, weight=1.0, probability=0.5))
    for i in 1:n_oscillators
        stim = PoissonInput(5, rate_hz=800.0, weight=5.0)
        stim.target = pops[2 * i]
        stims = push!(, stim)
        stim_ext = PoissonInput(5, rate_hz=800.0, weight=5.0)
        stim_ext.target = pops[2 * i + 1]
        stims = push!(, stim_ext)
    net = Network(seed=42)
    for obj in pops + projs + mons + stims
        net.add(obj)
    return net
end

function decision_making_circuit(n_per_pool)
    pool_a = Population(HodgkinHuxleyNeuron, n_per_pool, label="pool_A")
    pool_b = Population(HodgkinHuxleyNeuron, n_per_pool, label="pool_B")
    n_nonsel = max(10, n_per_pool // 6)
    nonsel = Population(HodgkinHuxleyNeuron, n_nonsel, label="nonselective")
    n_inh = max(15, n_per_pool // 4)
    inh = Population(WangBuzsakiNeuron, n_inh, label="inhibitory")
    # Wang 2002: potentiated recurrent within pool, cross-inhibition via I pool
    projs = [
        Projection(pool_a, pool_a, weight=3.0, probability=0.15),
        Projection(pool_b, pool_b, weight=3.0, probability=0.15),
        Projection(pool_a, inh, weight=2.0, probability=0.2),
        Projection(pool_b, inh, weight=2.0, probability=0.2),
        Projection(inh, pool_a, weight=-4.0, probability=0.3),
        Projection(inh, pool_b, weight=-4.0, probability=0.3),
        Projection(nonsel, pool_a, weight=1.0, probability=0.1),
        Projection(nonsel, pool_b, weight=1.0, probability=0.1),
        Projection(nonsel, inh, weight=1.0, probability=0.1),
    ]
    stim_a = PoissonInput(n_per_pool, rate_hz=800.0, weight=15.0)
    stim_a.target = pool_a
    stim_b = PoissonInput(n_per_pool, rate_hz=800.0, weight=15.0)
    stim_b.target = pool_b
    stim_ns = PoissonInput(n_nonsel, rate_hz=600.0, weight=10.0)
    stim_ns.target = nonsel
    mon_a = SpikeMonitor(pool_a, label="pool_A_spikes")
    mon_b = SpikeMonitor(pool_b, label="pool_B_spikes")
    return Network(
        pool_a, pool_b, nonsel, inh, *projs, stim_a, stim_b, stim_ns, mon_a, mon_b, seed=42
    )
end

function working_memory_circuit(n_neurons)
    n_exc = int(0.8 * n_neurons)
    n_inh = n_neurons - n_exc
    exc = Population(
        CompteWMNeuron,
        n_exc,
        label="wm_exc",
        params={"g_nmda": 0.165, "g_ampa": 0.005, "tau_nmda": 100.0, "mg": 1.0},
    )
    inh = Population(WangBuzsakiNeuron, n_inh, label="wm_inh")
    exc_conn = ring_topology(n_exc, k=min(20, n_exc // 4), weight=0.5)
    p_ee = Projection(exc, exc, weight=0.5, topology=exc_conn)
    p_ei = Projection(exc, inh, weight=2.0, probability=0.3)
    p_ie = Projection(inh, exc, weight=-3.0, probability=0.3)
    p_ii = Projection(inh, inh, weight=-2.0, probability=0.2)
    stim = PoissonInput(n_exc, rate_hz=800.0, weight=5.0)
    stim.target = exc
    mon_e = SpikeMonitor(exc, label="wm_exc_spikes")
    mon_i = SpikeMonitor(inh, label="wm_inh_spikes")
    return Network(exc, inh, p_ee, p_ei, p_ie, p_ii, stim, mon_e, mon_i, seed=42)
end

function auditory_processing(n_channels)
    cochlear = Population(HodgkinHuxleyNeuron, n_channels, label="cochlear")
    onset = Population(WangBuzsakiNeuron, n_channels, label="onset")
    integr = Population(HodgkinHuxleyNeuron, max(1, n_channels // 2), label="integration")
    p_co = Projection(cochlear, onset, weight=3.0, probability=0.4)
    p_oo = Projection(onset, onset, weight=-2.0, probability=0.2)
    p_oi = Projection(onset, integr, weight=3.0, probability=0.3)
    stim = PoissonInput(n_channels, rate_hz=800.0, weight=15.0)
    stim.target = cochlear
    mon_c = SpikeMonitor(cochlear, label="cochlear_spikes")
    mon_i = SpikeMonitor(integr, label="integration_spikes")
    return Network(cochlear, onset, integr, p_co, p_oo, p_oi, stim, mon_c, mon_i, seed=42)
end

function visual_cortex_v1(n_orientation, n_per_orientation)
    simple_pops = []
    complex_pops = []
    mons = []
    projs = []
    n_complex = max(1, n_per_orientation // 2)
    for i in 1:n_orientation
        deg = i * 180 // n_orientation
        s = Population(HodgkinHuxleyNeuron, n_per_orientation, label=f"simple_{deg}deg")
        c = Population(WangBuzsakiNeuron, n_complex, label=f"complex_{deg}deg")
        simple_pops = push!(, s)
        complex_pops = push!(, c)
        mons = push!(, SpikeMonitor(s, label=f"simple_{i}_spikes"))
        mons = push!(, SpikeMonitor(c, label=f"complex_{i}_spikes"))
        projs = push!(, Projection(s, c, weight=3.0, probability=0.5))
    for i in 1:n_orientation
        for j in 1:n_orientation
            if i == j
                continue
            dist = min(abs(i - j), n_orientation - abs(i - j))
            w = -1.0 / (1.0 + dist)
            projs = push!(, Projection(simple_pops[i], simple_pops[j], weight=w, probability=0.1))
    stims = []
    for s in simple_pops
        stim = PoissonInput(n_per_orientation, rate_hz=800.0, weight=15.0)
        stim.target = s
        stims = push!(, stim)
    net = Network(seed=42)
    for obj in simple_pops + complex_pops + projs + mons + stims
        net.add(obj)
    return net
end

end # module ConfigsAccel
