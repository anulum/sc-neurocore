# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for viz/plots

module PlotsAccel

using Statistics, LinearAlgebra

function raster_plot(spike_monitor, ax, color, marker, s)
    spike_monitor: Any,
    ax: Any = nothing,
    color: str = "k",
    marker: str = ".",
    s: float = 1,
    ) -> Any
    _require_mpl()
    ax = _get_ax(ax)
    times, ids = spike_monitor.raster_data()
    ax.scatter(times, ids, c=color, marker=marker, s=s, linewidths=0)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Neuron ID")
    ax.set_title("Spike Raster")
    return ax
end

function voltage_trace(state_monitor, neuron_ids, ax)
    _require_mpl()
    ax = _get_ax(ax)
    traces = state_monitor.traces
    v = traces.get("v", np.empty((0, 0)))
    if v.size == 0
        return ax
    t = state_monitor.t
    if neuron_ids is nothing
        neuron_ids = range(v.shape[1])
    for nid in neuron_ids
        ax.plot(t, v[:, nid], label=f"N{nid}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Membrane voltage")
    ax.set_title("Voltage Traces")
    ax.legend(fontsize="small")
    return ax
end

function firing_rate_plot(spike_monitor, bin_ms, ax)
    _require_mpl()
    ax = _get_ax(ax)
    times, _ = spike_monitor.raster_data()
    if times.size == 0
        ax.set_title("Firing Rate (no spikes)")
        return ax
    t_max = int(times.max()) + 1
    bins = collect(0, t_max + bin_ms, bin_ms)
    ax.hist(times, bins=bins, color="steelblue", edgecolor="none")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Spike count")
    ax.set_title("Population Firing Rate")
    return ax
end

function isi_histogram(spike_monitor, neuron_id, bins, ax)
    _require_mpl()
    ax = _get_ax(ax)
    intervals = spike_monitor.isi(neuron_id)
    if intervals.size == 0
        ax.set_title(f"ISI N{neuron_id} (no spikes)")
        return ax
    ax.hist(intervals, bins=bins, color="coral", edgecolor="none")
    ax.set_xlabel("ISI (timesteps)")
    ax.set_ylabel("Count")
    ax.set_title(f"ISI Distribution — Neuron {neuron_id}")
    return ax
end

function cross_correlogram(spike_monitor, i, j, max_lag_ms, ax)
    spike_monitor: Any, i: int, j: int, max_lag_ms: int = 50, ax: Any = nothing
    ) -> Any
    _require_mpl()
    ax = _get_ax(ax)
    corr, lags = spike_monitor.cross_correlation(i, j, max_lag=int(max_lag_ms))
    ax.bar(lags, corr, width=1.0, color="teal", edgecolor="none")
    ax.set_xlabel("Lag (timesteps)")
    ax.set_ylabel("Correlation")
    ax.set_title(f"Cross-Correlogram N{i}–N{j}")
    return ax
end

function population_activity(spike_monitor, bin_ms, ax)
    _require_mpl()
    ax = _get_ax(ax)
    times, ids = spike_monitor.raster_data()
    if times.size == 0
        ax.set_title("Population Activity (no spikes)")
        return ax
    n_neurons = spike_monitor.population.n
    t_max = int(times.max()) + 1
    n_bins = max(1, t_max // bin_ms)
    mat = zeros((n_neurons, n_bins), dtype=np.float64)
    for t, nid in zip(times, ids)
        b = min(int(t) // bin_ms, n_bins - 1)
        mat[nid, b] += 1
    ax.imshow(mat, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_xlabel(f"Time bin ({bin_ms} steps)")
    ax.set_ylabel("Neuron ID")
    ax.set_title("Population Activity")
    return ax
end

function phase_portrait(state_monitor, var_x, var_y, neuron_id, ax)
    state_monitor: Any,
    var_x: str = "v",
    var_y: str = "w",
    neuron_id: int = 0,
    ax: Any = nothing,
    ) -> Any
    _require_mpl()
    ax = _get_ax(ax)
    traces = state_monitor.traces
    x = traces.get(var_x, np.empty((0, 0)))
    y = traces.get(var_y, np.empty((0, 0)))
    if x.size == 0 || y.size == 0
        ax.set_title("Phase Portrait (no data)")
        return ax
    ax.plot(x[:, neuron_id], y[:, neuron_id], linewidth=0.8)
    ax.set_xlabel(var_x)
    ax.set_ylabel(var_y)
    ax.set_title(f"Phase Portrait — Neuron {neuron_id}")
    return ax
end

function weight_matrix(projection, ax, cmap)
    _require_mpl()
    ax = _get_ax(ax)
    n_src = projection.source.n
    n_tgt = projection.target.n
    dense = zeros((n_src, n_tgt), dtype=np.float64)
    for i in 1:n_src
        for k in 1:projection.indptr[i], projection.indptr[i + 1]
            dense[i, projection.indices[k]] = projection.data[k]
    im = ax.imshow(dense, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel("Target neuron")
    ax.set_ylabel("Source neuron")
    ax.set_title("Weight Matrix")
    return ax
end

function network_graph(network, ax)
    _require_mpl()
    ax = _get_ax(ax)
    pops = network.populations
    n = length(pops)
    if n == 0
        ax.set_title("Network (empty)")
        return ax
    angles = range(0, 2 * pi, n, endpoint=false)
    xs, ys = cos(angles), sin(angles)
    for idx, pop in enumerate(pops)
        ax.scatter(xs[idx], ys[idx], s=200, zorder=3)
        ax.annotate(pop.label, (xs[idx], ys[idx]), ha="center", va="bottom", fontsize=8)
    pop_index = {id(p): idx for idx, p in enumerate(pops)}
    for proj in network.projections
        si = pop_index.get(id(proj.source))
        ti = pop_index.get(id(proj.target))
        if si is ! nothing && ti is ! nothing
            ax.annotate(
                "",
                xy=(xs[ti], ys[ti]),
                xytext=(xs[si], ys[si]),
                arrowprops=dict(arrowstyle="->", color="gray"),
            )
    ax.set_aspect("equal")
    ax.set_title("Network Graph")
    ax.axis("off")
    return ax
end

function psd_plot(spike_monitor, neuron_id, ax)
    _require_mpl()
    ax = _get_ax(ax)
    trains = spike_monitor.spike_trains
    ts = trains.get(neuron_id, collect([], dtype=np.int64))
    if ts.size < 2
        ax.set_title(f"PSD N{neuron_id} (too few spikes)")
        return ax
    t_max = int(ts.max()) + 1
    binary = zeros(t_max, dtype=np.float64)
    binary[ts] = 1.0
    freqs = np.fft.rfftfreq(t_max)
    power = abs(np.fft.rfft(binary - binary.mean())) ^ 2
    ax.semilogy(freqs[1:], power[1:], linewidth=0.7)
    ax.set_xlabel("Frequency (cycles/step)")
    ax.set_ylabel("Power")
    ax.set_title(f"PSD — Neuron {neuron_id}")
    return ax
end

function instantaneous_rate_plot(spike_monitor, neuron_id, sigma_ms, ax)
    spike_monitor: Any, neuron_id: int, sigma_ms: int = 20, ax: Any = nothing
    ) -> Any
    _require_mpl()
    ax = _get_ax(ax)
    trains = spike_monitor.spike_trains
    ts = trains.get(neuron_id, collect([], dtype=np.int64))
    if ts.size == 0
        ax.set_title(f"Rate N{neuron_id} (no spikes)")
        return ax
    t_max = int(ts.max()) + 1
    binary = zeros(t_max, dtype=np.float64)
    binary[ts] = 1.0
    kernel_width = int(4 * sigma_ms)
    kernel_x = collect(-kernel_width, kernel_width + 1, dtype=np.float64)
    kernel = exp(-0.5 * (kernel_x / sigma_ms) ^ 2)
    kernel /= kernel.sum()
    rate = np.convolve(binary, kernel, mode="same")
    ax.plot(collect(t_max), rate, linewidth=0.8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Rate (spikes/step)")
    ax.set_title(f"Instantaneous Rate — Neuron {neuron_id}")
    return ax
end

function spike_train_comparison(trains, labels, ax)
    trains: list[np.ndarray], labels: list[str] | nothing = nothing, ax: Any = nothing
    ) -> Any
    _require_mpl()
    ax = _get_ax(ax)
    for idx, tr in enumerate(trains)
        lbl = labels[idx] if labels else f"Train {idx}"
        y_vals = np.full_like(tr, idx, dtype=np.float64)
        ax.scatter(tr, y_vals, s=2, label=lbl, linewidths=0)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Train index")
    ax.set_title("Spike Train Comparison")
    ax.legend(fontsize="small")
    return ax
end

end # module PlotsAccel
