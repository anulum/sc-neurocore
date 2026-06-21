// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio trace import request builder

export const MIN_TRACE_IMPORT_SAMPLES = 10;

export interface StudioTraceImportRequest {
  voltage: number[];
  dt: number;
}

export function parseStudioTraceVoltageValues(csv: string): number[] {
  const values: number[] = [];
  for (const line of csv.trim().split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const parts = trimmed.split(/[,\t\s]+/);
    const value = Number.parseFloat(parts[parts.length - 1] ?? "");
    if (Number.isFinite(value)) values.push(value);
  }
  return values;
}

export function studioTraceImportRequest(csv: string, dt: number): StudioTraceImportRequest {
  const voltage = parseStudioTraceVoltageValues(csv);
  if (voltage.length < MIN_TRACE_IMPORT_SAMPLES) {
    throw new Error(`Need at least ${MIN_TRACE_IMPORT_SAMPLES} data points`);
  }
  return { voltage, dt };
}
