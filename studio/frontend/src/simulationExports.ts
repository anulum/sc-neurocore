// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio simulation export builders

/**
 * Turning a run into something a reader can keep.
 *
 * Three formats, and a plan in front of each. The plan exists because two of
 * them can be unavailable — an SVG needs a trace with samples in it, and the
 * PNG path needs a canvas the browser will hand over — and a button that
 * silently does nothing is worse than one that says why it cannot.
 *
 * Filenames are built from the model's name, which is user text: it is
 * restricted to a safe set and falls back rather than being trusted, because
 * this string becomes a path on the reader's disk.
 */

import type { SimulateResponse } from "./api/client";
import type { ReplayPack } from "./api/types";
import { downloadBrowserArtefact } from "./browserArtefactDownload";
import { downloadCanvasPng } from "./browserCanvasExport";
import { fullDriveTrace, fullSampleTimes, fullStateNames, fullStateTrace } from "./simulationRaw";
import { at } from "./arrayAt";
import { PLOT_AXIS } from "./simulationPlotCanvas";

/** The trace colours an exported SVG uses, in order. */
const SVG_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373", "#ce93d8"] as const;

/** Which format a run is being exported to. */
export type SimulationExportKind = "csv" | "json" | "svg";

/** How an exported file reaches the reader's disk. */
export type SimulationExportDownloader = (payload: Blob, filename: string) => void;

/** The PNG path, which the caller supplies because it needs a live canvas. */
export type SimulationExportCanvasFallback = () => boolean;

/** One export: its bytes and the filename to offer it under. */
export interface SimulationExportArtefact {
  blob: Blob;
  filename: string;
}

/** An export that can proceed. */
export interface SimulationExportReadyPlan {
  available: true;
  artefact: SimulationExportArtefact;
  writeArtefact: (downloader?: SimulationExportDownloader) => void;
}

/** An export that cannot, and why. */
export interface SimulationExportUnavailablePlan {
  available: false;
  runFallback: (fallback?: SimulationExportCanvasFallback) => boolean;
}

/** An export that can proceed, or a refusal that says why not. */
export type SimulationExportPlan = SimulationExportReadyPlan | SimulationExportUnavailablePlan;

/**
 * A filename stem from a model's name, safe to write to disk.
 *
 * This string becomes a path, and the model name is user text. Anything
 * outside a conservative set is replaced, and a name that reduces to
 * nothing falls back rather than producing a dotfile or an empty name.
 *
 * @param modelName - The model's name, if there is one.
 * @param fallback - The stem to use when the name yields none.
 * @returns A stem safe to use in a filename.
 */
function safeSimulationStem(modelName: string | undefined, fallback: string): string {
  // `||` and not `??`: a name that is only whitespace trims to the empty
  // string, which is not a filename.
  // eslint-disable-next-line @typescript-eslint/prefer-nullish-coalescing
  const rawName = modelName?.trim() || fallback;
  const safeName = rawName.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^_+|_+$/g, "");
  return safeName.length > 0 ? safeName : fallback;
}

/**
 * Escape text for an SVG text node.
 *
 * @param value - The text.
 * @returns The text with SVG's special characters escaped.
 */
function escapeSvgText(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

/**
 * The filename a JSON export is offered under.
 *
 * @param result - The run.
 * @returns The filename.
 */
export function simulationJsonFilename(result: SimulateResponse): string {
  return `simulation_${safeSimulationStem(result.model_name, "custom")}.json`;
}

/**
 * The filename a CSV export is offered under.
 *
 * @param result - The run.
 * @returns The filename.
 */
export function simulationCsvFilename(result: SimulateResponse): string {
  return `simulation_${safeSimulationStem(result.model_name, "custom")}.csv`;
}

/**
 * The filename an SVG export is offered under.
 *
 * @param result - The run.
 * @returns The filename.
 */
export function simulationSvgFilename(result: SimulateResponse): string {
  return `sc_neurocore_${safeSimulationStem(result.model_name, "custom")}.svg`;
}

/**
 * A run as JSON bytes.
 *
 * @param result - The run.
 * @returns The blob.
 */
export function simulationJsonBlob(result: SimulateResponse): Blob {
  return new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
}

/**
 * A run as a named JSON file.
 *
 * @param result - The run.
 * @returns The artefact.
 */
export function simulationJsonExport(result: SimulateResponse): SimulationExportArtefact {
  return {
    blob: simulationJsonBlob(result),
    filename: simulationJsonFilename(result),
  };
}

/**
 * A run as CSV: one row per sample, one column per state variable.
 *
 * @param result - The run.
 * @returns The CSV text.
 */
export function simulationCsvText(result: SimulateResponse): string {
  // Export the full-resolution raw traces (post-step sample times); a legacy
  // result without raw custody exports the arrays it carries.
  const variables = fullStateNames(result);
  const traces = variables.map((variable) => fullStateTrace(result, variable) ?? []);
  const drive = fullDriveTrace(result);
  const times = fullSampleTimes(result);
  const header = ["time", ...variables, "current"].join(",");
  const rows = times.map((time, index) => {
    const values = traces.map((trace) => trace[index]?.toFixed(6) ?? "");
    return [
      time.toFixed(4),
      ...values,
      drive[index]?.toFixed(4) ?? "",
    ].join(",");
  });
  return [header, ...rows].join("\n");
}

/**
 * A run as CSV bytes.
 *
 * @param result - The run.
 * @returns The blob.
 */
export function simulationCsvBlob(result: SimulateResponse): Blob {
  return new Blob([simulationCsvText(result)], { type: "text/csv" });
}

/**
 * A run as a named CSV file.
 *
 * @param result - The run.
 * @returns The artefact.
 */
export function simulationCsvExport(result: SimulateResponse): SimulationExportArtefact {
  return {
    blob: simulationCsvBlob(result),
    filename: simulationCsvFilename(result),
  };
}

/**
 * A run as a standalone SVG plot.
 *
 * Self-contained: it carries its own axes, colours and labels, so it opens
 * in anything without the Studio's stylesheet.
 *
 * @param result - The run.
 * @returns The SVG document.
 */
export function simulationSvgText(result: SimulateResponse): string {
  const width = 800;
  const height = 400;
  const padding = { top: 20, right: 20, bottom: 40, left: 60 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const variables = Object.keys(result.states);
  const values = variables.flatMap((variable) => result.states[variable] ?? []);
  const yMin = values.length > 0 ? Math.min(...values) : 0;
  const yMax = values.length > 0 ? Math.max(...values) : 1;
  const yRange = yMax - yMin || 1;
  const xMin = result.time[0] ?? 0;
  const xMax = result.time[result.time.length - 1] ?? xMin + result.dt;
  const xRange = xMax - xMin || 1;
  const toX = (time: number): number => padding.left + ((time - xMin) / xRange) * plotWidth;
  const toY = (value: number): number =>
    padding.top + (1 - (value - yMin) / yRange) * plotHeight;

  let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">\n`;
  svg += `<rect width="${width}" height="${height}" fill="#0d1117"/>\n`;
  for (let index = 0; index <= 4; index++) {
    const y = padding.top + (plotHeight * index) / 4;
    svg += `<line x1="${padding.left}" y1="${y}" x2="${padding.left + plotWidth}" y2="${y}" stroke="#1a1f2a" stroke-width="0.5"/>\n`;
  }
  const stride = Math.max(1, Math.floor(result.time.length / 2000));
  for (const [variableIndex, variable] of variables.entries()) {
    const variableValues = result.states[variable] ?? [];
    const points: string[] = [];
    for (let index = 0; index < result.time.length; index += stride) {
      points.push(
        `${toX(at(result.time, index)).toFixed(1)},${toY(at(variableValues, index)).toFixed(1)}`,
      );
    }
    svg += `<polyline points="${points.join(" ")}" fill="none" stroke="${SVG_COLORS[variableIndex % SVG_COLORS.length]}" stroke-width="1.5"/>\n`;
  }
  for (const spikeIndex of result.spikes.slice(0, 200)) {
    const x = toX(result.time[spikeIndex] ?? spikeIndex * result.dt);
    svg += `<line x1="${x.toFixed(1)}" y1="${padding.top}" x2="${x.toFixed(1)}" y2="${padding.top + 8}" stroke="#ff5252" stroke-width="1.5"/>\n`;
  }
  svg += `<line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${padding.top + plotHeight}" stroke="${PLOT_AXIS}"/>\n`;
  svg += `<line x1="${padding.left}" y1="${padding.top + plotHeight}" x2="${padding.left + plotWidth}" y2="${padding.top + plotHeight}" stroke="${PLOT_AXIS}"/>\n`;
  svg += `<text x="${padding.left + plotWidth / 2}" y="${height - 5}" text-anchor="middle" fill="#8b949e" font-size="11" font-family="sans-serif">time (ms)</text>\n`;
  svg += `<text x="12" y="${padding.top + plotHeight / 2}" text-anchor="middle" fill="#8b949e" font-size="11" font-family="sans-serif" transform="rotate(-90,12,${padding.top + plotHeight / 2})">mV</text>\n`;
  for (let index = 0; index <= 4; index++) {
    const value = yMin + (yRange * index) / 4;
    svg += `<text x="${padding.left - 5}" y="${toY(value) + 3}" text-anchor="end" fill="#8b949e" font-size="9" font-family="monospace">${value.toFixed(1)}</text>\n`;
  }
  for (const [variableIndex, variable] of variables.entries()) {
    const x = padding.left + variableIndex * 80;
    const stroke = at(SVG_COLORS as readonly string[], variableIndex % SVG_COLORS.length);
    svg += `<line x1="${x}" y1="10" x2="${x + 15}" y2="10" stroke="${stroke}" stroke-width="2"/><text x="${x + 18}" y="13" fill="#8b949e" font-size="10">${escapeSvgText(variable)}</text>\n`;
  }
  if (result.model_name) {
    svg += `<text x="${width - padding.right}" y="13" text-anchor="end" fill="${PLOT_AXIS}" font-size="9" font-family="monospace">${escapeSvgText(result.model_name)}</text>\n`;
  }
  svg += `</svg>`;
  return svg;
}

/**
 * A run as SVG bytes.
 *
 * @param result - The run.
 * @returns The blob.
 */
export function simulationSvgBlob(result: SimulateResponse): Blob {
  return new Blob([simulationSvgText(result)], { type: "image/svg+xml" });
}

/**
 * A run as a named SVG file.
 *
 * @param result - The run.
 * @returns The artefact.
 */
export function simulationSvgExport(result: SimulateResponse): SimulationExportArtefact {
  return {
    blob: simulationSvgBlob(result),
    filename: simulationSvgFilename(result),
  };
}

/**
 * The filename a replay pack is offered under.
 *
 * Named with the experiment's identity digest, so two packs from the same
 * model but different experiments cannot overwrite one another.
 *
 * @param pack - The sealed pack.
 * @returns The filename.
 */
export function replayPackFilename(pack: ReplayPack): string {
  const name =
    typeof pack.request.name === "string"
      ? pack.request.name
      : pack.source === "ode"
        ? "equations"
        : "custom";
  return `replay_${safeSimulationStem(name, "custom")}_${pack.experiment_identity_sha256.slice(0, 12)}.json`;
}

/**
 * A sealed replay pack as bytes.
 *
 * @param pack - The pack.
 * @returns The blob.
 */
export function replayPackBlob(pack: ReplayPack): Blob {
  return new Blob([JSON.stringify(pack, null, 2)], { type: "application/json" });
}

/**
 * A saved replay pack: the experiment, its identity digest and the complete
 * expectation another installation compares against. The filename carries the
 * identity digest so two packs of the same model but different experiments do
 * not overwrite each other.
 *
 * @param pack - The sealed pack.
 * @returns The artefact.
 */
export function replayPackExport(pack: ReplayPack): SimulationExportArtefact {
  return { blob: replayPackBlob(pack), filename: replayPackFilename(pack) };
}

/**
 * The artefact for one export format.
 *
 * @param kind - The format.
 * @param result - The run.
 * @returns The artefact.
 */
export function simulationExportArtefact(
  kind: SimulationExportKind,
  result: SimulateResponse,
): SimulationExportArtefact {
  if (kind === "json") {
    return simulationJsonExport(result);
  }
  if (kind === "csv") {
    return simulationCsvExport(result);
  }
  return simulationSvgExport(result);
}

/**
 * Decide whether a run can be exported in one format.
 *
 * @param kind - The format.
 * @param result - The run, if there is one.
 * @returns A plan that can proceed, or a refusal that says why not.
 */
export function simulationExportPlan(
  kind: SimulationExportKind,
  result: SimulateResponse | null,
): SimulationExportPlan {
  if (result === null) {
    return {
      available: false,
      runFallback: (fallback = kind === "svg" ? downloadCanvasPng : undefined) =>
        fallback?.() ?? false,
    };
  }
  const artefact = simulationExportArtefact(kind, result);
  return {
    available: true,
    artefact,
    writeArtefact: (downloader = downloadBrowserArtefact) => {
      downloader(artefact.blob, artefact.filename);
    },
  };
}
