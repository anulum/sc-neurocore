# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for plots

fn _require_mpl() -> Int:
    var __require_mpl_line = 'if not HAS_MPL:'
    var __require_mpl_line = 'raise ImportError("matplotlib is required for sc_neurocore.v'
    return 0

fn _get_ax(ax: Int) -> Int:
    var __get_ax_line = 'if ax is not 0:'
    return 0  # return ax
    var __get_ax_line = '_, ax = plt.subplots()'
    return 0  # return ax

fn raster_plot(spike_monitor: Int, ax: Int, color: Int, marker: Int, s: Int) -> Int:
    var _raster_plot_line = 'spike_monitor: Any,'
    var _raster_plot_line = 'ax: Any = 0,'
    var _raster_plot_line = 'color: str = "k",'
    var _raster_plot_line = 'marker: str = ".",'
    var _raster_plot_line = 's: float = 1,'
    var _raster_plot_line = ') -> Any:'
    var _raster_plot_line = '_require_mpl()'
    var _raster_plot_line = 'ax = _get_ax(ax)'
    var _raster_plot_line = 'times, ids = spike_monitor.raster_data()'
    var _raster_plot_line = 'ax.scatter(times, ids, c=color, marker=marker, s=s, linewidt'
    var _raster_plot_line = 'ax.set_xlabel("Timestep")'
    var _raster_plot_line = 'ax.set_ylabel("Neuron ID")'
    var _raster_plot_line = 'ax.set_title("Spike Raster")'
    return 0  # return ax

fn voltage_trace(state_monitor: Int, neuron_ids: Int, ax: Int) -> Int:
    var _voltage_trace_line = '_require_mpl()'
    var _voltage_trace_line = 'ax = _get_ax(ax)'
    var _voltage_trace_line = 'traces = state_monitor.traces'
    var _voltage_trace_line = 'v = traces.get("v", empty((0, 0)))'
    var _voltage_trace_line = 'if v.size == 0:'
    return 0  # return ax
    var _voltage_trace_line = 't = state_monitor.t'
    var _voltage_trace_line = 'if neuron_ids is 0:'
    var _voltage_trace_line = 'neuron_ids = range(v.shape[1])'
    var _voltage_trace_line = 'for nid in neuron_ids:'
    var _voltage_trace_line = 'ax.plot(t, v[:, nid], label=f"N{nid}")'
    var _voltage_trace_line = 'ax.set_xlabel("Timestep")'
    var _voltage_trace_line = 'ax.set_ylabel("Membrane voltage")'
    var _voltage_trace_line = 'ax.set_title("Voltage Traces")'
    var _voltage_trace_line = 'ax.legend(fontsize="small")'
    return 0  # return ax

fn firing_rate_plot(spike_monitor: Int, bin_ms: Int, ax: Int) -> Int:
    var _firing_rate_plot_line = '_require_mpl()'
    var _firing_rate_plot_line = 'ax = _get_ax(ax)'
    var _firing_rate_plot_line = 'times, _ = spike_monitor.raster_data()'
    var _firing_rate_plot_line = 'if times.size == 0:'
    var _firing_rate_plot_line = 'ax.set_title("Firing Rate (no spikes)")'
    return 0  # return ax
    var _firing_rate_plot_line = 't_max = int(times.max()) + 1'
    var _firing_rate_plot_line = 'bins = arange(0, t_max + bin_ms, bin_ms)'
    var _firing_rate_plot_line = 'ax.hist(times, bins=bins, color="steelblue", edgecolor="none'
    var _firing_rate_plot_line = 'ax.set_xlabel("Timestep")'
    var _firing_rate_plot_line = 'ax.set_ylabel("Spike count")'
    var _firing_rate_plot_line = 'ax.set_title("Population Firing Rate")'
    return 0  # return ax

fn isi_histogram(spike_monitor: Int, neuron_id: Int, bins: Int, ax: Int) -> Int:
    var _isi_histogram_line = '_require_mpl()'
    var _isi_histogram_line = 'ax = _get_ax(ax)'
    var _isi_histogram_line = 'intervals = spike_monitor.isi(neuron_id)'
    var _isi_histogram_line = 'if intervals.size == 0:'
    var _isi_histogram_line = 'ax.set_title(f"ISI N{neuron_id} (no spikes)")'
    return 0  # return ax
    var _isi_histogram_line = 'ax.hist(intervals, bins=bins, color="coral", edgecolor="none'
    var _isi_histogram_line = 'ax.set_xlabel("ISI (timesteps)")'
    var _isi_histogram_line = 'ax.set_ylabel("Count")'
    var _isi_histogram_line = 'ax.set_title(f"ISI Distribution — Neuron {neuron_id}")'
    return 0  # return ax

fn cross_correlogram(spike_monitor: Int, i: Int, j: Int, max_lag_ms: Int, ax: Int) -> Int:
    var _cross_correlogram_line = 'spike_monitor: Any, i: int, j: int, max_lag_ms: int = 50, ax'
    var _cross_correlogram_line = ') -> Any:'
    var _cross_correlogram_line = '_require_mpl()'
    var _cross_correlogram_line = 'ax = _get_ax(ax)'
    var _cross_correlogram_line = 'corr, lags = spike_monitor.cross_correlation(i, j, max_lag=i'
    var _cross_correlogram_line = 'ax.bar(lags, corr, width=1.0, color="teal", edgecolor="none"'
    var _cross_correlogram_line = 'ax.set_xlabel("Lag (timesteps)")'
    var _cross_correlogram_line = 'ax.set_ylabel("Correlation")'
    var _cross_correlogram_line = 'ax.set_title(f"Cross-Correlogram N{i}–N{j}")'
    return 0  # return ax

fn population_activity(spike_monitor: Int, bin_ms: Int, ax: Int) -> Int:
    var _population_activity_line = '_require_mpl()'
    var _population_activity_line = 'ax = _get_ax(ax)'
    var _population_activity_line = 'times, ids = spike_monitor.raster_data()'
    var _population_activity_line = 'if times.size == 0:'
    var _population_activity_line = 'ax.set_title("Population Activity (no spikes)")'
    return 0  # return ax
    var _population_activity_line = 'n_neurons = spike_monitor.population.n'
    var _population_activity_line = 't_max = int(times.max()) + 1'
    var _population_activity_line = 'n_bins = max(1, t_max // bin_ms)'
    var _population_activity_line = 'mat = zeros((n_neurons, n_bins), dtype=float64)'
    var _population_activity_line = 'for t, nid in zip(times, ids):'
    var _population_activity_line = 'b = min(int(t) // bin_ms, n_bins - 1)'
    var _population_activity_line = 'mat[nid, b] += 1'
    var _population_activity_line = 'ax.imshow(mat, aspect="auto", origin="lower", interpolation='
    var _population_activity_line = 'ax.set_xlabel(f"Time bin ({bin_ms} steps)")'
    var _population_activity_line = 'ax.set_ylabel("Neuron ID")'
    var _population_activity_line = 'ax.set_title("Population Activity")'
    return 0  # return ax

fn phase_portrait(state_monitor: Int, var_x: Int, var_y: Int, neuron_id: Int, ax: Int) -> Int:
    var _phase_portrait_line = 'state_monitor: Any,'
    var _phase_portrait_line = 'var_x: str = "v",'
    var _phase_portrait_line = 'var_y: str = "w",'
    var _phase_portrait_line = 'neuron_id: int = 0,'
    var _phase_portrait_line = 'ax: Any = 0,'
    var _phase_portrait_line = ') -> Any:'
    var _phase_portrait_line = '_require_mpl()'
    var _phase_portrait_line = 'ax = _get_ax(ax)'
    var _phase_portrait_line = 'traces = state_monitor.traces'
    var _phase_portrait_line = 'x = traces.get(var_x, empty((0, 0)))'
    var _phase_portrait_line = 'y = traces.get(var_y, empty((0, 0)))'
    var _phase_portrait_line = 'if x.size == 0 or y.size == 0:'
    var _phase_portrait_line = 'ax.set_title("Phase Portrait (no data)")'
    return 0  # return ax
    var _phase_portrait_line = 'ax.plot(x[:, neuron_id], y[:, neuron_id], linewidth=0.8)'
    var _phase_portrait_line = 'ax.set_xlabel(var_x)'
    var _phase_portrait_line = 'ax.set_ylabel(var_y)'
    var _phase_portrait_line = 'ax.set_title(f"Phase Portrait — Neuron {neuron_id}")'
    return 0  # return ax

fn weight_matrix(projection: Int, ax: Int, cmap: Int) -> Int:
    var _weight_matrix_line = '_require_mpl()'
    var _weight_matrix_line = 'ax = _get_ax(ax)'
    var _weight_matrix_line = 'n_src = projection.source.n'
    var _weight_matrix_line = 'n_tgt = projection.target.n'
    var _weight_matrix_line = 'dense = zeros((n_src, n_tgt), dtype=float64)'
    var _weight_matrix_line = 'for i in range(n_src):'
    var _weight_matrix_line = 'for k in range(projection.indptr[i], projection.indptr[i + 1'
    var _weight_matrix_line = 'dense[i, projection.indices[k]] = projection.data[k]'
    var _weight_matrix_line = 'im = ax.imshow(dense, aspect="auto", cmap=cmap, interpolatio'
    var _weight_matrix_line = 'ax.figure.colorbar(im, ax=ax, fraction=0.046)'
    var _weight_matrix_line = 'ax.set_xlabel("Target neuron")'
    var _weight_matrix_line = 'ax.set_ylabel("Source neuron")'
    var _weight_matrix_line = 'ax.set_title("Weight Matrix")'
    return 0  # return ax

fn network_graph(network: Int, ax: Int) -> Int:
    var _network_graph_line = '_require_mpl()'
    var _network_graph_line = 'ax = _get_ax(ax)'
    var _network_graph_line = 'pops = network.populations'
    var _network_graph_line = 'n = len(pops)'
    var _network_graph_line = 'if n == 0:'
    var _network_graph_line = 'ax.set_title("Network (empty)")'
    return 0  # return ax
    var _network_graph_line = 'angles = linspace(0, 2 * pi, n, endpoint=False)'
    var _network_graph_line = 'xs, ys = cos(angles), sin(angles)'
    var _network_graph_line = 'for idx, pop in enumerate(pops):'
    var _network_graph_line = 'ax.scatter(xs[idx], ys[idx], s=200, zorder=3)'
    var _network_graph_line = 'ax.annotate(pop.label, (xs[idx], ys[idx]), ha="center", va="'
    var _network_graph_line = 'pop_index = {id(p): idx for idx, p in enumerate(pops)}'
    var _network_graph_line = 'for proj in network.projections:'
    var _network_graph_line = 'si = pop_index.get(id(proj.source))'
    var _network_graph_line = 'ti = pop_index.get(id(proj.target))'
    var _network_graph_line = 'if si is not 0 and ti is not 0:'
    var _network_graph_line = 'ax.annotate('
    var _network_graph_line = '"",'
    var _network_graph_line = 'xy=(xs[ti], ys[ti]),'
    var _network_graph_line = 'xytext=(xs[si], ys[si]),'
    var _network_graph_line = 'arrowprops=dict(arrowstyle="->", color="gray"),'
    var _network_graph_line = ')'
    var _network_graph_line = 'ax.set_aspect("equal")'
    var _network_graph_line = 'ax.set_title("Network Graph")'
    var _network_graph_line = 'ax.axis("off")'
    return 0  # return ax

fn psd_plot(spike_monitor: Int, neuron_id: Int, ax: Int) -> Int:
    var _psd_plot_line = '_require_mpl()'
    var _psd_plot_line = 'ax = _get_ax(ax)'
    var _psd_plot_line = 'trains = spike_monitor.spike_trains'
    var _psd_plot_line = 'ts = trains.get(neuron_id, array([], dtype=int64))'
    var _psd_plot_line = 'if ts.size < 2:'
    var _psd_plot_line = 'ax.set_title(f"PSD N{neuron_id} (too few spikes)")'
    return 0  # return ax
    var _psd_plot_line = 't_max = int(ts.max()) + 1'
    var _psd_plot_line = 'binary = zeros(t_max, dtype=float64)'
    var _psd_plot_line = 'binary[ts] = 1.0'
    var _psd_plot_line = 'freqs = fft.rfftfreq(t_max)'
    var _psd_plot_line = 'power = abs(fft.rfft(binary - binary.mean())) ** 2'
    var _psd_plot_line = 'ax.semilogy(freqs[1:], power[1:], linewidth=0.7)'
    var _psd_plot_line = 'ax.set_xlabel("Frequency (cycles/step)")'
    var _psd_plot_line = 'ax.set_ylabel("Power")'
    var _psd_plot_line = 'ax.set_title(f"PSD — Neuron {neuron_id}")'
    return 0  # return ax

fn instantaneous_rate_plot(spike_monitor: Int, neuron_id: Int, sigma_ms: Int, ax: Int) -> Int:
    var _instantaneous_rate_plot_line = 'spike_monitor: Any, neuron_id: int, sigma_ms: int = 20, ax: '
    var _instantaneous_rate_plot_line = ') -> Any:'
    var _instantaneous_rate_plot_line = '_require_mpl()'
    var _instantaneous_rate_plot_line = 'ax = _get_ax(ax)'
    var _instantaneous_rate_plot_line = 'trains = spike_monitor.spike_trains'
    var _instantaneous_rate_plot_line = 'ts = trains.get(neuron_id, array([], dtype=int64))'
    var _instantaneous_rate_plot_line = 'if ts.size == 0:'
    var _instantaneous_rate_plot_line = 'ax.set_title(f"Rate N{neuron_id} (no spikes)")'
    return 0  # return ax
    var _instantaneous_rate_plot_line = 't_max = int(ts.max()) + 1'
    var _instantaneous_rate_plot_line = 'binary = zeros(t_max, dtype=float64)'
    var _instantaneous_rate_plot_line = 'binary[ts] = 1.0'
    var _instantaneous_rate_plot_line = 'kernel_width = int(4 * sigma_ms)'
    var _instantaneous_rate_plot_line = 'kernel_x = arange(-kernel_width, kernel_width + 1, dtype=flo'
    var _instantaneous_rate_plot_line = 'kernel = exp(-0.5 * (kernel_x / sigma_ms) ** 2)'
    var _instantaneous_rate_plot_line = 'kernel /= kernel.sum()'
    var _instantaneous_rate_plot_line = 'rate = convolve(binary, kernel, mode="same")'
    var _instantaneous_rate_plot_line = 'ax.plot(arange(t_max), rate, linewidth=0.8)'
    var _instantaneous_rate_plot_line = 'ax.set_xlabel("Timestep")'
    var _instantaneous_rate_plot_line = 'ax.set_ylabel("Rate (spikes/step)")'
    var _instantaneous_rate_plot_line = 'ax.set_title(f"Instantaneous Rate — Neuron {neuron_id}")'
    return 0  # return ax

fn spike_train_comparison(trains: Int, labels: Int, ax: Int) -> Int:
    var _spike_train_comparison_line = 'trains: list[ndarray], labels: list[str] | 0 = 0, ax: Any = '
    var _spike_train_comparison_line = ') -> Any:'
    var _spike_train_comparison_line = '_require_mpl()'
    var _spike_train_comparison_line = 'ax = _get_ax(ax)'
    var _spike_train_comparison_line = 'for idx, tr in enumerate(trains):'
    var _spike_train_comparison_line = 'lbl = labels[idx] if labels else f"Train {idx}"'
    var _spike_train_comparison_line = 'y_vals = full_like(tr, idx, dtype=float64)'
    var _spike_train_comparison_line = 'ax.scatter(tr, y_vals, s=2, label=lbl, linewidths=0)'
    var _spike_train_comparison_line = 'ax.set_xlabel("Timestep")'
    var _spike_train_comparison_line = 'ax.set_ylabel("Train index")'
    var _spike_train_comparison_line = 'ax.set_title("Spike Train Comparison")'
    var _spike_train_comparison_line = 'ax.legend(fontsize="small")'
    return 0  # return ax
