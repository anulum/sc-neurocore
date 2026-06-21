import type { StudioCapability } from "./api/client";

export type CapabilityStatus =
  | "stable"
  | "experimental"
  | "degraded"
  | "unavailable";

export type ShellStatus = CapabilityStatus | "unregistered";

/** Stable Studio panel identifiers used by the frontend shell. */
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
  | "admin";

/** Aggregate health projection for the backend capability registry. */
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

/** Frontend-ready availability state for one Studio panel. */
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

export interface CapabilityLoadStatePatch {
  capabilities?: StudioCapability[];
  capabilitiesError?: string | null;
  capabilitiesLoading: boolean;
}

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
  admin: "Admin",
};

/** Return one capability by its stable backend identifier. */
export function capabilityById(
  capabilities: readonly StudioCapability[],
  capabilityId: string,
): StudioCapability | null {
  return capabilities.find((capability) => capability.capability_id === capabilityId) ?? null;
}

export function capabilityLoadingState(): CapabilityLoadStatePatch {
  return {
    capabilitiesError: null,
    capabilitiesLoading: true,
  };
}

export function capabilityLoadedState(
  capabilities: StudioCapability[],
): CapabilityLoadStatePatch {
  return {
    capabilities,
    capabilitiesError: null,
    capabilitiesLoading: false,
  };
}

export function capabilityFailureState(error: unknown): CapabilityLoadStatePatch {
  return {
    capabilitiesError: error instanceof Error && error.message.length > 0
      ? error.message
      : "Capability check failed",
    capabilitiesLoading: false,
  };
}

/** Summarize backend capability health without storing duplicate UI state. */
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

/** Project backend capability health into frontend panel availability. */
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
