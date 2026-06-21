// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio simulation export builders

import type { SimulateResponse } from "./api/client";
import { downloadBrowserArtefact } from "./browserArtefactDownload";
import { downloadCanvasPng } from "./browserCanvasExport";

const SVG_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373", "#ce93d8"] as const;

export type SimulationExportKind = "csv" | "json" | "svg";

export type SimulationExportDownloader = (payload: Blob, filename: string) => void;

export type SimulationExportCanvasFallback = () => boolean;

export interface SimulationExportArtefact {
  blob: Blob;
  filename: string;
}

export interface SimulationExportReadyPlan {
  available: true;
  artefact: SimulationExportArtefact;
  writeArtefact: (downloader?: SimulationExportDownloader) => void;
}

export interface SimulationExportUnavailablePlan {
  available: false;
  runFallback: (fallback?: SimulationExportCanvasFallback) => boolean;
}

export type SimulationExportPlan = SimulationExportReadyPlan | SimulationExportUnavailablePlan;

function safeSimulationStem(modelName: string | undefined, fallback: string): string {
  const rawName = modelName?.trim() || fallback;
  const safeName = rawName.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^_+|_+$/g, "");
  return safeName.length > 0 ? safeName : fallback;
}

function escapeSvgText(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

export function simulationJsonFilename(result: SimulateResponse): string {
  return `simulation_${safeSimulationStem(result.model_name, "custom")}.json`;
}

export function simulationCsvFilename(result: SimulateResponse): string {
  return `simulation_${safeSimulationStem(result.model_name, "custom")}.csv`;
}

export function simulationSvgFilename(result: SimulateResponse): string {
  return `sc_neurocore_${safeSimulationStem(result.model_name, "custom")}.svg`;
}

export function simulationJsonBlob(result: SimulateResponse): Blob {
  return new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
}

export function simulationJsonExport(result: SimulateResponse): SimulationExportArtefact {
  return {
    blob: simulationJsonBlob(result),
    filename: simulationJsonFilename(result),
  };
}

export function simulationCsvText(result: SimulateResponse): string {
  const variables = Object.keys(result.states);
  const header = ["time", ...variables, "current"].join(",");
  const rows = result.time.map((time, index) => {
    const values = variables.map((variable) => result.states[variable][index]?.toFixed(6) ?? "");
    return [
      time.toFixed(4),
      ...values,
      result.current_trace[index]?.toFixed(4) ?? "",
    ].join(",");
  });
  return [header, ...rows].join("\n");
}

export function simulationCsvBlob(result: SimulateResponse): Blob {
  return new Blob([simulationCsvText(result)], { type: "text/csv" });
}

export function simulationCsvExport(result: SimulateResponse): SimulationExportArtefact {
  return {
    blob: simulationCsvBlob(result),
    filename: simulationCsvFilename(result),
  };
}

export function simulationSvgText(result: SimulateResponse): string {
  const width = 800;
  const height = 400;
  const padding = { top: 20, right: 20, bottom: 40, left: 60 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const variables = Object.keys(result.states);
  const values = variables.flatMap((variable) => result.states[variable]);
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
  for (let variableIndex = 0; variableIndex < variables.length; variableIndex++) {
    const variableValues = result.states[variables[variableIndex]];
    const points: string[] = [];
    for (let index = 0; index < result.time.length; index += stride) {
      points.push(
        `${toX(result.time[index]).toFixed(1)},${toY(variableValues[index]).toFixed(1)}`,
      );
    }
    svg += `<polyline points="${points.join(" ")}" fill="none" stroke="${SVG_COLORS[variableIndex % SVG_COLORS.length]}" stroke-width="1.5"/>\n`;
  }
  for (const spikeIndex of result.spikes.slice(0, 200)) {
    const x = toX(result.time[spikeIndex] ?? spikeIndex * result.dt);
    svg += `<line x1="${x.toFixed(1)}" y1="${padding.top}" x2="${x.toFixed(1)}" y2="${padding.top + 8}" stroke="#ff5252" stroke-width="1.5"/>\n`;
  }
  svg += `<line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${padding.top + plotHeight}" stroke="#484f58"/>\n`;
  svg += `<line x1="${padding.left}" y1="${padding.top + plotHeight}" x2="${padding.left + plotWidth}" y2="${padding.top + plotHeight}" stroke="#484f58"/>\n`;
  svg += `<text x="${padding.left + plotWidth / 2}" y="${height - 5}" text-anchor="middle" fill="#8b949e" font-size="11" font-family="sans-serif">time (ms)</text>\n`;
  svg += `<text x="12" y="${padding.top + plotHeight / 2}" text-anchor="middle" fill="#8b949e" font-size="11" font-family="sans-serif" transform="rotate(-90,12,${padding.top + plotHeight / 2})">mV</text>\n`;
  for (let index = 0; index <= 4; index++) {
    const value = yMin + (yRange * index) / 4;
    svg += `<text x="${padding.left - 5}" y="${toY(value) + 3}" text-anchor="end" fill="#8b949e" font-size="9" font-family="monospace">${value.toFixed(1)}</text>\n`;
  }
  for (let variableIndex = 0; variableIndex < variables.length; variableIndex++) {
    const x = padding.left + variableIndex * 80;
    svg += `<line x1="${x}" y1="10" x2="${x + 15}" y2="10" stroke="${SVG_COLORS[variableIndex % SVG_COLORS.length]}" stroke-width="2"/><text x="${x + 18}" y="13" fill="#8b949e" font-size="10">${escapeSvgText(variables[variableIndex])}</text>\n`;
  }
  if (result.model_name) {
    svg += `<text x="${width - padding.right}" y="13" text-anchor="end" fill="#484f58" font-size="9" font-family="monospace">${escapeSvgText(result.model_name)}</text>\n`;
  }
  svg += `</svg>`;
  return svg;
}

export function simulationSvgBlob(result: SimulateResponse): Blob {
  return new Blob([simulationSvgText(result)], { type: "image/svg+xml" });
}

export function simulationSvgExport(result: SimulateResponse): SimulationExportArtefact {
  return {
    blob: simulationSvgBlob(result),
    filename: simulationSvgFilename(result),
  };
}

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
