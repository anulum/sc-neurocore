// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/**
 * What each Studio panel may do, according to the backend's capability
 * registry.
 *
 * A panel is available, unavailable, or **unregistered**, and the third is not
 * a failure. Some panels have no backend contract at all -- `delays` is one --
 * and those are shown and usable; only a panel whose contract exists and is
 * unhealthy is closed. Treating "no contract" as "not available" would hide
 * working panels, so the two are separate states with separate messages.
 *
 * Only *unmet* requirements are projected. A panel that is closed should say
 * what is missing, not list everything it needs and leave the reader to find
 * the one that failed.
 */

import type { StudioCapability } from "./api/client";

/** The states the backend registry gives a capability. */
export type CapabilityStatus =
  | "stable"
  | "experimental"
  | "degraded"
  | "unavailable";

/**
 * What the shell shows for a panel: a capability's status, or `unregistered`
 * for a panel the registry has no contract for.
 */
export type ShellStatus = CapabilityStatus | "unregistered";

/** Every panel the shell can show, named by a stable key. */
export type PanelKey =
  | "trace"
  | "phase"
  | "isi"
  | "fi-curve"
  | "bifurcation"
  | "sensitivity"
  | "precision"
  | "heatmap"
  | "verilog"
  | "code"
  | "compare"
  | "freq"
  | "sta"
  | "characterize"
  | "multi"
  | "network"
  | "ir"
  | "synth"
  | "train"
  | "canvas"
  | "delays"
  | "admin"
  | "candidate";

/** The registry counted by status, with the worst one named. */
export interface CapabilitySummary {
  total: number;
  healthy: number;
  unavailable: number;
  degraded: number;
  experimental: number;
  stable: number;
  headline: string;
  worstStatus: CapabilityStatus | "none";
}

/** What one panel may do, and what to say when it may not. */
export interface PanelCapabilityState {
  panelKey: PanelKey;
  capabilityId: string | null;
  title: string;
  available: boolean;
  status: ShellStatus;
  message: string;
  requirements: string[];
  evidence: string[];
  docsPath: string | null;
}

/**
 * A patch from the capability request. The fields are optional because the
 * loading patch deliberately leaves the previous list on screen rather than
 * blanking the shell while a refresh is in flight.
 */
export interface CapabilityLoadStatePatch {
  capabilities?: StudioCapability[];
  capabilitiesError?: string | null;
  capabilitiesLoading: boolean;
}

/**
 * Which backend capability governs each panel. Several panels share one
 * contract because they are views of the same backend surface. A panel absent
 * here is unregistered, not unavailable.
 */
const PANEL_CAPABILITY_IDS: Partial<Record<PanelKey, string>> = {
  trace: "studio.simulation_workbench",
  phase: "studio.simulation_workbench",
  isi: "studio.simulation_workbench",
  "fi-curve": "studio.analysis_suite",
  bifurcation: "studio.analysis_suite",
  sensitivity: "studio.analysis_suite",
  precision: "studio.analysis_suite",
  heatmap: "studio.analysis_suite",
  compare: "studio.analysis_suite",
  freq: "studio.analysis_suite",
  sta: "studio.analysis_suite",
  characterize: "studio.analysis_suite",
  multi: "studio.analysis_suite",
  network: "studio.simulation_workbench",
  verilog: "studio.compiler_inspector",
  code: "studio.export_tools",
  ir: "studio.compiler_inspector",
  canvas: "studio.network_canvas",
  synth: "studio.synthesis_dashboard",
  train: "studio.training_monitor",
  admin: "studio.capability_registry",
};

/** The name each panel is shown under before its contract is consulted. */
const PANEL_TITLES: Record<PanelKey, string> = {
  trace: "Trace",
  phase: "Phase",
  isi: "ISI",
  "fi-curve": "f-I",
  bifurcation: "Bifurcation",
  sensitivity: "Sensitivity",
  precision: "Q8.8",
  heatmap: "2D Sweep",
  verilog: "RTL",
  code: "Code",
  compare: "A/B",
  freq: "Frequency",
  sta: "STA",
  characterize: "Characterize",
  multi: "Multi-model",
  network: "E-I Network",
  ir: "IR",
  synth: "FPGA",
  train: "Training",
  canvas: "Canvas",
  delays: "Learnable Delays (DCLS)",
  admin: "Admin",
  candidate: "Candidate model",
};

/**
 * The name a panel is shown under.
 *
 * @param panelKey - The panel.
 * @returns Its title.
 */
export function panelTitle(panelKey: PanelKey): string {
  return PANEL_TITLES[panelKey];
}

/**
 * Find one capability by its backend identifier.
 *
 * @param capabilities - The registry as it was last read.
 * @param capabilityId - The identifier to look for.
 * @returns The capability, or `null` when the registry has no such contract.
 */
export function capabilityById(
  capabilities: readonly StudioCapability[],
  capabilityId: string,
): StudioCapability | null {
  return capabilities.find((capability) => capability.capability_id === capabilityId) ?? null;
}

/**
 * Start a capability request, keeping the list already on screen.
 *
 * @returns The patch.
 */
export function capabilityLoadingState(): CapabilityLoadStatePatch {
  return {
    capabilitiesError: null,
    capabilitiesLoading: true,
  };
}

/**
 * Record a capability list that arrived.
 *
 * @param capabilities - The list.
 * @returns The patch.
 */
export function capabilityLoadedState(
  capabilities: StudioCapability[],
): CapabilityLoadStatePatch {
  return {
    capabilities,
    capabilitiesError: null,
    capabilitiesLoading: false,
  };
}

/**
 * Record a capability request that failed.
 *
 * @param error - What was thrown, which need not be an `Error`.
 * @returns The patch.
 */
export function capabilityFailureState(error: unknown): CapabilityLoadStatePatch {
  return {
    capabilitiesError: error instanceof Error && error.message.length > 0
      ? error.message
      : "Capability check failed",
    capabilitiesLoading: false,
  };
}

/**
 * Count the registry by status, for the shell's health line.
 *
 * `unavailable` counts a capability the registry reports as unhealthy *or* as
 * unavailable, so a contract that disagrees with itself is counted as broken
 * rather than overlooked.
 *
 * @param capabilities - The registry as it was last read.
 * @returns The counts, a headline, and the worst status present.
 */
export function summarizeCapabilities(
  capabilities: readonly StudioCapability[],
): CapabilitySummary {
  const healthy = capabilities.filter((capability) => capability.healthy).length;
  const unavailable = capabilities.filter(
    (capability) => !capability.healthy || capability.status === "unavailable",
  ).length;
  const degraded = capabilities.filter((capability) => capability.status === "degraded").length;
  const experimental = capabilities.filter(
    (capability) => capability.status === "experimental",
  ).length;
  const stable = capabilities.filter((capability) => capability.status === "stable").length;
  const worstStatus = selectWorstStatus(capabilities);

  return {
    total: capabilities.length,
    healthy,
    unavailable,
    degraded,
    experimental,
    stable,
    headline: `${healthy}/${capabilities.length} ready`,
    worstStatus,
  };
}

/**
 * Decide what one panel may do.
 *
 * @param capabilities - The registry as it was last read.
 * @param panelKey - The panel.
 * @returns Its state: available and unregistered when no contract governs it,
 *   unavailable when its contract is missing from the registry, and otherwise
 *   whatever its contract reports, with the unmet requirements listed.
 */
export function panelCapabilityState(
  capabilities: readonly StudioCapability[],
  panelKey: PanelKey,
): PanelCapabilityState {
  const capabilityId = PANEL_CAPABILITY_IDS[panelKey] ?? null;
  if (capabilityId === null) {
    return {
      panelKey,
      capabilityId,
      title: PANEL_TITLES[panelKey],
      available: true,
      status: "unregistered",
      message: "No backend capability contract is registered for this panel.",
      requirements: [],
      evidence: [],
      docsPath: null,
    };
  }

  const capability = capabilityById(capabilities, capabilityId);
  if (capability === null) {
    return {
      panelKey,
      capabilityId,
      title: PANEL_TITLES[panelKey],
      available: false,
      status: "unavailable",
      message: "Backend capability contract is missing from the registry.",
      requirements: [],
      evidence: [],
      docsPath: null,
    };
  }

  return {
    panelKey,
    capabilityId,
    title: capability.title,
    available: capability.healthy && capability.status !== "unavailable",
    status: capability.status,
    message: capability.message,
    requirements: capability.requirements
      .filter((requirement) => !requirement.available)
      .map((requirement) => `${requirement.name}: ${requirement.detail}`),
    evidence: capability.evidence,
    docsPath: capability.docs_path,
  };
}

/**
 * Name the worst status in a list, for the shell's single indicator.
 *
 * Unhealthy and `unavailable` are both worst: a capability the registry
 * calls healthy while reporting `unavailable` is still a panel that will
 * not work.
 *
 * @param capabilities - The list.
 * @returns The worst status, or `none` for an empty list.
 */
function selectWorstStatus(
  capabilities: readonly StudioCapability[],
): CapabilityStatus | "none" {
  if (capabilities.length === 0) return "none";
  if (capabilities.some((capability) => !capability.healthy || capability.status === "unavailable")) {
    return "unavailable";
  }
  if (capabilities.some((capability) => capability.status === "degraded")) return "degraded";
  if (capabilities.some((capability) => capability.status === "experimental")) {
    return "experimental";
  }
  return "stable";
}
