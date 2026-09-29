// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — every analysis view in words and as a table

/**
 * What each analysis view says to a reader who cannot see the canvas.
 *
 * A canvas is pixels. The trace view was the only one described in numbers;
 * every other view named itself and pointed at exports that hold only the
 * trace. Each view here gets one sentence and one table, both read from the
 * result the view draws, so a screen reader gets what the eye gets. A
 * sentence states only what the result carries: a point that failed, an
 * elasticity that is undefined, a sweep that is not a continuation are said
 * as such, never smoothed over.
 */

import type {
  BifurcationResponse,
  CharacterizeResponse,
  CompareResponse,
  FICurveResponse,
  FreqResponse,
  HeatmapResponse,
  NetworkResult,
  NullclineResponse,
  PrecisionResponse,
  SensitivityResponse,
  SimulateResponse,
} from "./api/types";
import { saturatedPopulations, saturationWarning } from "./networkSaturation";
import { formatReading, multiModelDescription, runName, traceDataRows, traceDescription } from "./plotAccessibility";
import type { SpikeTriggeredAverage } from "./plots/analysisViews";

/** A view's numbers, as the Data table shows them. */
export interface ViewTable {
  caption: string;
  columns: string[];
  rows: string[][];
}

/** A view in words and as a table. */
export interface ViewDescription {
  sentence: string;
  table: ViewTable;
}

/** Most rows a view's table lists before it says how many it left out. */
export const VIEW_TABLE_ROW_LIMIT = 200;

const r = formatReading;

/**
 * Cap a table's rows, saying how many were left out.
 *
 * @param rows - Every row.
 * @param width - How many columns a row has.
 * @returns At most {@link VIEW_TABLE_ROW_LIMIT} rows, plus one that counts the rest.
 */
function capped(rows: string[][], width: number): string[][] {
  if (rows.length <= VIEW_TABLE_ROW_LIMIT) return rows;
  const rest = rows.length - VIEW_TABLE_ROW_LIMIT;
  return [...rows.slice(0, VIEW_TABLE_ROW_LIMIT), [`${rest} more rows in the JSON export of this analysis`, ...Array<string>(width - 1).fill("")]];
}

/**
 * Minimum and maximum of the finite values.
 *
 * @param values - The values.
 * @returns The extremes, or `null` when there is no finite value.
 */
function extent(values: readonly number[]): { min: number; max: number } | null {
  const finite = values.filter((v) => Number.isFinite(v));
  if (finite.length === 0) return null;
  return { min: Math.min(...finite), max: Math.max(...finite) };
}

/**
 * "1 spike", "3 spikes".
 *
 * @param n - The count.
 * @param noun - The singular noun.
 * @returns The phrase.
 */
function count(n: number, noun: string): string {
  return `${String(n)} ${n === 1 ? noun : `${noun}s`}`;
}

/**
 * The trace view.
 *
 * @param result - The run.
 * @returns The description.
 */
export function describeTrace(result: SimulateResponse): ViewDescription {
  return {
    sentence: traceDescription(result),
    table: {
      caption: `Trace data: ${String(result.n_steps)} steps of ${r(result.dt)} ms, ${count(result.spike_count, "spike")}`,
      columns: ["Variable", "Samples shown", "Minimum", "Maximum", "Final"],
      rows: traceDataRows(result).map((row) => [
        row.variable, String(row.samples), r(row.minimum), r(row.maximum), r(row.final),
      ]),
    },
  };
}

/**
 * The f-I curve.
 *
 * The first current with a non-zero rate is the first *measured* firing
 * point; the true threshold lies between it and the point before, and the
 * sentence says so rather than naming it the threshold.
 *
 * @param fi - The sweep.
 * @returns The description.
 */
export function describeFICurve(fi: FICurveResponse): ViewDescription {
  const n = Math.min(fi.currents.length, fi.rates.length);
  const rows = Array.from({ length: n }, (_, i) => [r(fi.currents[i] ?? null), r(fi.rates[i] ?? null)]);
  const span = extent(fi.currents);
  const range = span === null ? "no currents" : `I from ${r(span.min)} to ${r(span.max)} in ${count(n, "point")}`;
  const firing = fi.rates.findIndex((rate) => rate > 0);
  let finding: string;
  if (n === 0) {
    finding = "nothing was measured";
  } else if (firing < 0) {
    finding = "no spikes at any current in the range";
  } else {
    let peak = firing;
    for (let i = 0; i < n; i += 1) if ((fi.rates[i] ?? -Infinity) > (fi.rates[peak] ?? -Infinity)) peak = i;
    const before = firing > 0 ? `silent up to I = ${r(fi.currents[firing - 1] ?? null)}; ` : "";
    finding = `${before}first firing measured at I = ${r(fi.currents[firing] ?? null)} (${r(fi.rates[firing] ?? null)} Hz); ` +
      `highest rate ${r(fi.rates[peak] ?? null)} Hz at I = ${r(fi.currents[peak] ?? null)}`;
  }
  return {
    sentence: `f-I curve over ${range}: ${finding}. Currents are in the model's own units.`,
    table: { caption: "f-I curve: firing rate at each constant current", columns: ["Current", "Rate (Hz)"], rows: capped(rows, 2) },
  };
}

/**
 * The bifurcation (extrema) sweep.
 *
 * @param bif - The sweep.
 * @returns The description.
 */
export function describeBifurcation(bif: BifurcationResponse): ViewDescription {
  const kinds = bif.attractor_kinds ?? [];
  const points = bif.param_values.length;
  const tally = { fixed: 0, oscillating: 0, insufficient: 0 };
  const rows = bif.param_values.map((value, i) => {
    const extrema = bif.attractors[i] ?? [];
    // The server classifies each point; a point it did not classify is not
    // guessed at from the number of extrema.
    const kind = kinds[i];
    if (kind === "fixed_point") tally.fixed += 1;
    else if (kind === "insufficient_samples") tally.insufficient += 1;
    else if (kind === "extrema") tally.oscillating += 1;
    const span = extent(extrema);
    const label = kind === "fixed_point" ? "fixed point" : kind === "insufficient_samples" ? "too few samples"
      : kind === "extrema" ? "extrema" : "not classified";
    return [r(value), label, String(extrema.length), r(span?.min ?? null), r(span?.max ?? null)];
  });
  const span = extent(bif.param_values);
  const variable = bif.variable ?? "the first state";
  const parts = [
    tally.fixed > 0 ? `${count(tally.fixed, "point")} ${tally.fixed === 1 ? "settles" : "settle"} to a fixed point` : "",
    tally.oscillating > 0
      ? `${count(tally.oscillating, "point")} ${tally.oscillating === 1 ? "shows" : "show"} oscillation extrema`
      : "",
    tally.insufficient > 0 ? `${count(tally.insufficient, "point")} had too few samples to tell` : "",
  ].filter((part) => part !== "");
  return {
    sentence: `Extrema of ${variable} swept over ${bif.param_name}` +
      (span === null ? "" : ` from ${r(span.min)} to ${r(span.max)}`) +
      ` in ${count(points, "point")}${bif.protocol ? ` under a ${bif.protocol} drive` : ""}: ` +
      `${parts.length > 0 ? parts.join("; ") : "no point was classified"}. ` +
      "This is a sampled sweep, not a numerical continuation.",
    table: {
      caption: `Extrema of ${variable} at each value of ${bif.param_name}`,
      columns: [bif.param_name, "Kind", "Extrema", "Lowest", "Highest"],
      rows: capped(rows, 5),
    },
  };
}

/**
 * The 2-D sweep.
 *
 * @param map - The sweep.
 * @returns The description.
 */
export function describeHeatmap(map: HeatmapResponse): ViewDescription {
  let best = { rate: -Infinity, x: NaN, y: NaN };
  let silent = 0;
  map.rates.forEach((row, j) => {
    row.forEach((rate, i) => {
      if (rate === 0) silent += 1;
      if (rate > best.rate) best = { rate, x: map.x_values[i] ?? NaN, y: map.y_values[j] ?? NaN };
    });
  });
  const cells = map.x_values.length * map.y_values.length;
  const rows = map.y_values.map((y, j) => [r(y), ...map.x_values.map((_, i) => r(map.rates[j]?.[i] ?? null))]);
  return {
    sentence: `Firing rate over ${map.param_x} × ${map.param_y}, ${String(map.x_values.length)} × ` +
      `${String(map.y_values.length)} points: from ${r(map.rate_min)} to ${r(map.rate_max)} Hz` +
      (Number.isFinite(best.rate) ? `, highest at ${map.param_x} = ${r(best.x)}, ${map.param_y} = ${r(best.y)}` : "") +
      `; ${count(silent, "point")} of ${String(cells)} silent.`,
    table: {
      caption: `Firing rate (Hz): rows ${map.param_y}, columns ${map.param_x}`,
      columns: [`${map.param_y} \\ ${map.param_x}`, ...map.x_values.map((x) => r(x))],
      rows: capped(rows, map.x_values.length + 1),
    },
  };
}

/**
 * The sensitivity ranking.
 *
 * @param sens - The analysis.
 * @returns The description.
 */
export function describeSensitivity(sens: SensitivityResponse): ViewDescription {
  const defined = sens.sensitivities.filter((row) => row.sensitivity !== null);
  const undefinedCount = sens.sensitivities.length - defined.length;
  const top = [...defined]
    .sort((a, b) => Math.abs(b.sensitivity ?? 0) - Math.abs(a.sensitivity ?? 0))
    .slice(0, 3)
    .map((row) => `${row.param} (${r(row.sensitivity)})`);
  return {
    sentence: `Rate elasticity of ${count(sens.sensitivities.length, "parameter")} around a base rate of ` +
      `${r(sens.base_rate)} Hz: ${top.length > 0 ? `largest ${top.join(", ")}` : "none is defined"}` +
      `${undefinedCount > 0 ? `; ${String(undefinedCount)} undefined` : ""}.`,
    table: {
      caption: "Rate elasticity per parameter; an undefined one says why",
      columns: ["Parameter", "Elasticity", "Rate − (Hz)", "Rate + (Hz)", "Why undefined"],
      rows: capped(sens.sensitivities.map((row) => [
        row.param, r(row.sensitivity), r(row.rate_minus ?? null), r(row.rate_plus ?? null), row.reason ?? "",
      ]), 5),
    },
  };
}

/**
 * The spike-aligned average.
 *
 * It averages the run's first state around each spike, not the injected
 * drive; the sentence says which.
 *
 * @param sta - The average.
 * @param variable - The state it averages.
 * @returns The description.
 */
export function describeSta(sta: SpikeTriggeredAverage, variable: string): ViewDescription {
  const span = extent(sta.average);
  let peak = 0;
  sta.average.forEach((value, i) => { if (value > (sta.average[peak] ?? -Infinity)) peak = i; });
  const first = sta.time_ms[0];
  const last = sta.time_ms.at(-1);
  return {
    sentence: `Average of ${variable} around ${count(sta.n_spikes, "spike")}, from ${r(first ?? null)} to ` +
      `${r(last ?? null)} ms relative to the spike: ${span === null ? "no finite value" :
        `from ${r(span.min)} to ${r(span.max)}, highest at ${r(sta.time_ms[peak] ?? null)} ms`}.`,
    table: {
      caption: `Average of ${variable} by time relative to the spike`,
      columns: ["Time (ms)", `Mean ${variable}`],
      rows: capped(sta.time_ms.map((t, i) => [r(t), r(sta.average[i] ?? null)]), 2),
    },
  };
}

/**
 * The frequency response.
 *
 * @param freq - The sweep.
 * @returns The description.
 */
export function describeFrequency(freq: FreqResponse): ViewDescription {
  const n = Math.min(freq.frequencies_hz.length, freq.rates.length);
  let hi = 0, lo = 0;
  for (let i = 0; i < n; i += 1) {
    if ((freq.rates[i] ?? -Infinity) > (freq.rates[hi] ?? -Infinity)) hi = i;
    if ((freq.rates[i] ?? Infinity) < (freq.rates[lo] ?? Infinity)) lo = i;
  }
  const span = extent(freq.frequencies_hz);
  return {
    sentence: `Firing rate under a sine drive of amplitude ${r(freq.amplitude)}` +
      (span === null ? "" : ` from ${r(span.min)} to ${r(span.max)} Hz`) + ` in ${count(n, "point")}` +
      (n === 0 ? "." : `: highest ${r(freq.rates[hi] ?? null)} Hz at ${r(freq.frequencies_hz[hi] ?? null)} Hz, ` +
        `lowest ${r(freq.rates[lo] ?? null)} Hz at ${r(freq.frequencies_hz[lo] ?? null)} Hz.`),
    table: {
      caption: "Firing rate at each drive frequency",
      columns: ["Drive frequency (Hz)", "Rate (Hz)"],
      rows: capped(Array.from({ length: n }, (_, i) => [r(freq.frequencies_hz[i] ?? null), r(freq.rates[i] ?? null)]), 2),
    },
  };
}

/**
 * The characterisation summary.
 *
 * @param char - The characterisation.
 * @returns The description.
 */
export function describeCharacterization(char: CharacterizeResponse): ViewDescription {
  const threshold = char.threshold_current === null
    ? "no firing threshold within the swept currents"
    : `firing threshold near I = ${r(char.threshold_current)}`;
  const sensitive = char.top_sensitivities[0];
  const rows: string[][] = [
    ["Firing pattern", char.pattern.description],
    ["Threshold current", char.threshold_current === null ? "none in range" : r(char.threshold_current)],
    ["Highest rate (Hz)", r(char.max_rate)],
    ["Spikes in the base run", String(char.spike_count)],
    ...Object.entries(char.state_ranges).map(([name, range]) => [
      `${name} range`, `${r(range.min)} to ${r(range.max)} (mean ${r(range.mean)})`,
    ]),
    ...char.top_sensitivities.map((row) => [`Rate change for ${row.param}`, r(row.rate_change)]),
  ];
  return {
    sentence: `${char.pattern.description}; ${threshold}; highest rate ${r(char.max_rate)} Hz; ` +
      `${count(char.spike_count, "spike")} in the base run` +
      (sensitive ? `; most rate-sensitive parameter ${sensitive.param}.` : "."),
    table: { caption: "Characterisation", columns: ["Quantity", "Value"], rows },
  };
}

/**
 * One run of a comparison, in a phrase.
 *
 * @param label - "A" or "B".
 * @param run - The run.
 * @param index - Its position.
 * @returns The phrase and its table row.
 */
function comparedRun(label: string, run: SimulateResponse, index: number): { phrase: string; row: string[] } {
  const first = traceDataRows(run)[0];
  const name = runName(run, index);
  const range = first ? `${first.variable} from ${r(first.minimum)} to ${r(first.maximum)}` : "no recorded state";
  return {
    phrase: `${label}: ${name}, ${count(run.spike_count, "spike")} (${r(run.stats.rate_hz)} Hz), ${range}`,
    row: [label, name, String(run.spike_count), r(run.stats.rate_hz), first?.variable ?? "", r(first?.minimum ?? null), r(first?.maximum ?? null)],
  };
}

/**
 * The A/B comparison.
 *
 * @param cmp - The comparison.
 * @returns The description.
 */
export function describeCompare(cmp: CompareResponse): ViewDescription {
  const a = comparedRun("A", cmp.a, 0);
  const b = comparedRun("B", cmp.b, 1);
  return {
    sentence: `A/B comparison on the same drive. ${a.phrase}; ${b.phrase}.`,
    table: {
      caption: "A/B comparison",
      columns: ["Run", "Model", "Spikes", "Rate (Hz)", "State", "Lowest", "Highest"],
      rows: [a.row, b.row],
    },
  };
}

/**
 * The E-I network.
 *
 * @param net - The run.
 * @returns The description.
 */
export function describeNetwork(net: NetworkResult): ViewDescription {
  const populations = [
    { label: "excitatory", mean_rate_hz: net.mean_exc_rate, neurons: net.n_exc },
    { label: "inhibitory", mean_rate_hz: net.mean_inh_rate, neurons: net.n_inh },
  ];
  const warning = saturationWarning(saturatedPopulations(populations, net.dt));
  return {
    sentence: `E-I network of ${String(net.n_exc)} excitatory and ${String(net.n_inh)} inhibitory neurons over ` +
      `${r(net.duration)} ms at dt ${r(net.dt)} ms: ${count(net.n_spikes, "spike")}; mean rate ` +
      `${r(net.mean_exc_rate)} Hz excitatory, ${r(net.mean_inh_rate)} Hz inhibitory.${warning === null ? "" : ` ${warning}`}`,
    table: {
      caption: "E-I network: mean rate per population",
      columns: ["Population", "Neurons", "Mean rate (Hz)"],
      rows: populations.map((p) => [p.label, String(p.neurons), r(p.mean_rate_hz)]),
    },
  };
}

/**
 * The interspike-interval histogram.
 *
 * @param result - The run whose intervals it bins.
 * @returns The description, or `null` without a histogram.
 */
export function describeIsi(result: SimulateResponse): ViewDescription | null {
  const histogram = result.stats.isi_histogram;
  if (histogram === null) return null;
  const total = histogram.counts.reduce((a, b) => a + b, 0);
  let mode = 0;
  histogram.counts.forEach((c, i) => { if (c > (histogram.counts[mode] ?? -Infinity)) mode = i; });
  return {
    sentence: `Interspike-interval histogram of ${count(total, "interval")}` +
      (result.stats.isi_mean_ms === null ? "" : `: mean ${r(result.stats.isi_mean_ms)} ms`) +
      (result.stats.isi_cv === null ? "" : `, CV ${r(result.stats.isi_cv)}`) +
      (total === 0 ? "." : `; most intervals (${String(histogram.counts[mode] ?? 0)}) between ` +
        `${r(histogram.edges[mode] ?? null)} and ${r(histogram.edges[mode + 1] ?? null)} ms.`),
    table: {
      caption: "Interspike intervals per bin",
      columns: ["From (ms)", "To (ms)", "Intervals"],
      rows: capped(histogram.counts.map((c, i) => [r(histogram.edges[i] ?? null), r(histogram.edges[i + 1] ?? null), String(c)]), 3),
    },
  };
}

/**
 * The phase portrait.
 *
 * @param result - The run it draws.
 * @param nullclines - The nullclines, when computed.
 * @returns The description, or `null` with fewer than two states.
 */
export function describePhase(result: SimulateResponse, nullclines: NullclineResponse | null): ViewDescription | null {
  const rows = traceDataRows(result);
  const x = rows[0];
  const y = rows[1];
  if (x === undefined || y === undefined) return null;
  const nullclineText = nullclines === null
    ? "nullclines not computed"
    : `${nullclines.nullcline_0.variable}-nullcline ${count(nullclines.nullcline_0.points.length, "point")}, ` +
      `${nullclines.nullcline_1.variable}-nullcline ${count(nullclines.nullcline_1.points.length, "point")}`;
  return {
    sentence: `Phase portrait of ${y.variable} against ${x.variable}: ${x.variable} from ${r(x.minimum)} to ` +
      `${r(x.maximum)}, ${y.variable} from ${r(y.minimum)} to ${r(y.maximum)}, ending at ` +
      `(${r(x.final)}, ${r(y.final)}); ${nullclineText}.`,
    table: {
      caption: "Phase portrait: the two states drawn",
      columns: ["State", "Minimum", "Maximum", "Final"],
      rows: [x, y].map((row) => [row.variable, r(row.minimum), r(row.maximum), r(row.final)]),
    },
  };
}

/**
 * The float-against-fixed-point comparison.
 *
 * @param prec - The comparison.
 * @returns The description.
 */
export function describePrecision(prec: PrecisionResponse): ViewDescription {
  const format = prec.arithmetic?.q_format ?? prec.encoding?.q_format ?? "fixed-point";
  const divergence = prec.error.first_divergence_step;
  const where = divergence === null || divergence === undefined
    ? "the spike trains never diverge"
    : `the spike trains first diverge at step ${String(divergence)}`;
  return {
    sentence: `Float64 reference against the ${format} candidate for ${prec.error.variable}: largest error ` +
      `${r(prec.error.max_error)}, mean ${r(prec.error.mean_error)}, RMS ${r(prec.error.rms_error)}; ${where}; ` +
      `${count(prec.float_result.spike_count, "spike")} in float, ${count(prec.fixed_result.spike_count, "spike")} in ${format}.`,
    table: {
      caption: `Error of ${prec.error.variable}, ${format} against float64`,
      columns: ["Measure", "Value"],
      rows: [
        ["Largest error", r(prec.error.max_error)],
        ["Mean error", r(prec.error.mean_error)],
        ["RMS error", r(prec.error.rms_error)],
        ["First divergence step", divergence === null || divergence === undefined ? "none" : String(divergence)],
        ["Spikes, float64", String(prec.float_result.spike_count)],
        [`Spikes, ${format}`, String(prec.fixed_result.spike_count)],
      ],
    },
  };
}

/**
 * The multi-model overlay.
 *
 * @param runs - The overlaid runs.
 * @returns The description.
 */
export function describeMulti(runs: readonly SimulateResponse[]): ViewDescription {
  return {
    sentence: multiModelDescription(runs),
    table: {
      caption: "Overlaid runs",
      columns: ["Model", "State", "Lowest", "Highest", "Spikes", "Rate (Hz)", "Step (ms)"],
      rows: runs.map((run, i) => {
        const first = traceDataRows(run)[0];
        return [runName(run, i), first?.variable ?? "", r(first?.minimum ?? null), r(first?.maximum ?? null),
          String(run.spike_count), r(run.stats.rate_hz), r(run.dt)];
      }),
    },
  };
}

/** The results a view can be described from. */
export interface ViewResults {
  result: SimulateResponse | null;
  fiResult: FICurveResponse | null;
  bifResult: BifurcationResponse | null;
  heatmapResult: HeatmapResponse | null;
  sensResult: SensitivityResponse | null;
  staResult: SpikeTriggeredAverage | null;
  freqResult: FreqResponse | null;
  charResult: CharacterizeResponse | null;
  compareResult: CompareResponse | null;
  networkResult: NetworkResult | null;
  nullclineResult: NullclineResponse | null;
  precResult: PrecisionResponse | null;
  multiResults: SimulateResponse[] | null;
}

/**
 * Describe whatever the active view draws.
 *
 * @param tab - The active view.
 * @param results - Every result the Studio holds.
 * @param showsTrace - Whether the canvas fell back to drawing the trace.
 * @returns The description, or `null` when the view has nothing to describe.
 */
export function describeView(tab: string, results: ViewResults, showsTrace: boolean): ViewDescription | null {
  const { result } = results;
  if (showsTrace) return result === null ? null : describeTrace(result);
  switch (tab) {
    case "fi-curve": return results.fiResult === null ? null : describeFICurve(results.fiResult);
    case "bifurcation": return results.bifResult === null ? null : describeBifurcation(results.bifResult);
    case "heatmap": return results.heatmapResult === null ? null : describeHeatmap(results.heatmapResult);
    case "sensitivity": return results.sensResult === null ? null : describeSensitivity(results.sensResult);
    case "sta":
      return results.staResult === null || result === null
        ? null
        : describeSta(results.staResult, Object.keys(result.states)[0] ?? "the first state");
    case "freq": return results.freqResult === null ? null : describeFrequency(results.freqResult);
    case "characterize": return results.charResult === null ? null : describeCharacterization(results.charResult);
    case "compare": return results.compareResult === null ? null : describeCompare(results.compareResult);
    case "network": return results.networkResult === null ? null : describeNetwork(results.networkResult);
    case "isi": return result === null ? null : describeIsi(result);
    case "phase": return result === null ? null : describePhase(result, results.nullclineResult);
    case "precision": return results.precResult === null ? null : describePrecision(results.precResult);
    case "multi":
      return results.multiResults === null || results.multiResults.length === 0 ? null : describeMulti(results.multiResults);
    default: return result === null ? null : describeTrace(result);
  }
}
