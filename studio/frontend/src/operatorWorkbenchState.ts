// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio operator workbench state
import type { SimulateResponse, StudioOperatorStatus } from "./api/client";
import type { GuidedFlowState, GuidedFlowStepKey, GuidedFlowStepStatus } from "./guidedFlowState";

export type OperatorWorkbenchCardKey =
  | "workspace"
  | "model"
  | "simulation"
  | "evidence"
  | "compile"
  | "export";

export type OperatorWorkbenchCardStatus = "ready" | "active" | "warning" | "blocked";

export interface OperatorWorkbenchInputs {
  sourceMode: "model" | "ode";
  selectedModelName: string;
  modelCount: number;
  projectName: string | null;
  serverProjectCount: number;
  savedSessionCount: number;
  simulationResult: SimulateResponse | null;
  isSimulating: boolean;
  progressMessage: string;
  operatorStatus: StudioOperatorStatus | null;
  guidedFlow: GuidedFlowState;
  compileComplete: boolean;
  synthesisComplete: boolean;
  evidenceBundleExported: boolean;
}

export interface OperatorWorkbenchCard {
  action: string;
  detail: string;
  key: OperatorWorkbenchCardKey;
  status: OperatorWorkbenchCardStatus;
  title: string;
  value: string;
}

export interface OperatorWorkbenchState {
  cards: OperatorWorkbenchCard[];
  evidenceActionEnabled: boolean;
  headline: string;
  subhead: string;
}

const STEP_STATUS_LABELS: Record<GuidedFlowStepStatus, string> = {
  available: "ready",
  blocked: "blocked",
  completed: "done",
  current: "next",
};

/**
 * Build the first-screen operator summary from existing Studio state.
 *
 * The workbench does not introduce new authority or hidden checks. It only
 * aggregates already-loaded store values, operator status, and guided-flow
 * evidence into compact cards that can be rendered before the detailed panels.
 */
export function buildOperatorWorkbenchState(
  inputs: OperatorWorkbenchInputs,
): OperatorWorkbenchState {
  const currentStep = currentGuidedStep(inputs.guidedFlow);
  return {
    cards: [
      workspaceCard(inputs),
      modelCard(inputs),
      simulationCard(inputs),
      evidenceCard(inputs),
      compileCard(inputs),
      exportCard(inputs),
    ],
    evidenceActionEnabled: inputs.projectName !== null && !inputs.evidenceBundleExported,
    headline: currentStep === null ? "Workflow complete" : `Next: ${currentStep.title}`,
    subhead: `${inputs.guidedFlow.completedCount}/${inputs.guidedFlow.totalCount} lifecycle steps complete`,
  };
}

function workspaceCard(inputs: OperatorWorkbenchInputs): OperatorWorkbenchCard {
  const projectName = inputs.projectName;
  const hasProject = projectName !== null;
  return {
    action: hasProject ? "Open projects" : "Save project",
    detail: `${inputs.serverProjectCount} server projects, ${inputs.savedSessionCount} local sessions`,
    key: "workspace",
    status: hasProject ? "ready" : "warning",
    title: "Workspace",
    value: hasProject ? projectName : "Unsaved",
  };
}

function modelCard(inputs: OperatorWorkbenchInputs): OperatorWorkbenchCard {
  if (inputs.sourceMode === "ode") {
    return {
      action: "Edit equations",
      detail: `${inputs.modelCount} catalogue models remain available`,
      key: "model",
      status: "ready",
      title: "Design source",
      value: "ODE mode",
    };
  }
  return {
    action: "Browse models",
    detail: `${inputs.modelCount} catalogue models loaded`,
    key: "model",
    status: inputs.selectedModelName.length > 0 ? "ready" : "blocked",
    title: "Design source",
    value: inputs.selectedModelName.length > 0 ? inputs.selectedModelName : "No model selected",
  };
}

function simulationCard(inputs: OperatorWorkbenchInputs): OperatorWorkbenchCard {
  if (inputs.isSimulating) {
    return {
      action: "Inspect progress",
      detail: inputs.progressMessage.length > 0 ? inputs.progressMessage : "Simulation is running",
      key: "simulation",
      status: "active",
      title: "Simulation",
      value: "Running",
    };
  }
  if (inputs.simulationResult === null) {
    return {
      action: "Run simulation",
      detail: "No path-free simulation metadata is loaded",
      key: "simulation",
      status: "blocked",
      title: "Simulation",
      value: "Not run",
    };
  }
  return {
    action: "Inspect trace",
    detail: `${inputs.simulationResult.spike_count} spikes, ${inputs.simulationResult.stats.rate_hz} Hz`,
    key: "simulation",
    status: "ready",
    title: "Simulation",
    value: inputs.simulationResult.run_metadata.status,
  };
}

function evidenceCard(inputs: OperatorWorkbenchInputs): OperatorWorkbenchCard {
  if (inputs.operatorStatus === null) {
    return {
      action: "Load status",
      detail: "Operator status has not loaded yet",
      key: "evidence",
      status: "warning",
      title: "Evidence health",
      value: "Unknown",
    };
  }
  if (!inputs.operatorStatus.audit.healthy || inputs.operatorStatus.capabilities.unavailable_count > 0) {
    return {
      action: "Open admin",
      detail: `${inputs.operatorStatus.capabilities.unavailable_count} unavailable capabilities, audit healthy=${inputs.operatorStatus.audit.healthy}`,
      key: "evidence",
      status: "warning",
      title: "Evidence health",
      value: inputs.operatorStatus.deployment_profile,
    };
  }
  return {
    action: "Open admin",
    detail: `${inputs.operatorStatus.capabilities.healthy_count}/${inputs.operatorStatus.capabilities.total_count} capabilities healthy`,
    key: "evidence",
    status: "ready",
    title: "Evidence health",
    value: inputs.operatorStatus.deployment_profile,
  };
}

function compileCard(inputs: OperatorWorkbenchInputs): OperatorWorkbenchCard {
  const compileStep = guidedStep(inputs.guidedFlow, "compile");
  const synthStep = guidedStep(inputs.guidedFlow, "synthesise");
  if (inputs.synthesisComplete) {
    return {
      action: "Open synthesis",
      detail: "Compile and synthesis evidence are available",
      key: "compile",
      status: "ready",
      title: "Hardware path",
      value: "Synthesised",
    };
  }
  if (inputs.compileComplete) {
    return {
      action: "Open synthesis",
      detail: stepDetail(synthStep),
      key: "compile",
      status: statusFromStep(synthStep),
      title: "Hardware path",
      value: "Compiled",
    };
  }
  return {
    action: "Open compiler",
    detail: stepDetail(compileStep),
    key: "compile",
    status: statusFromStep(compileStep),
    title: "Hardware path",
    value: STEP_STATUS_LABELS[compileStep?.status ?? "blocked"],
  };
}

function exportCard(inputs: OperatorWorkbenchInputs): OperatorWorkbenchCard {
  if (inputs.evidenceBundleExported) {
    return {
      action: "Download bundle",
      detail: "An evidence bundle is ready for artifact download",
      key: "export",
      status: "ready",
      title: "Export",
      value: "Bundle ready",
    };
  }
  const exportStep = guidedStep(inputs.guidedFlow, "export");
  return {
    action: inputs.projectName === null ? "Save project first" : "Export bundle",
    detail: stepDetail(exportStep),
    key: "export",
    status: inputs.projectName === null ? "blocked" : statusFromStep(exportStep),
    title: "Export",
    value: STEP_STATUS_LABELS[exportStep?.status ?? "blocked"],
  };
}

function currentGuidedStep(state: GuidedFlowState): { key: GuidedFlowStepKey; title: string } | null {
  const step = state.steps.find((candidate) => candidate.status === "current");
  return step === undefined ? null : { key: step.key, title: step.title };
}

function guidedStep(
  state: GuidedFlowState,
  key: GuidedFlowStepKey,
): GuidedFlowState["steps"][number] | null {
  return state.steps.find((candidate) => candidate.key === key) ?? null;
}

function statusFromStep(
  step: GuidedFlowState["steps"][number] | null,
): OperatorWorkbenchCardStatus {
  if (step === null) {
    return "blocked";
  }
  if (step.status === "completed" || step.status === "available" || step.status === "current") {
    return "ready";
  }
  return "blocked";
}

function stepDetail(step: GuidedFlowState["steps"][number] | null): string {
  if (step === null) {
    return "Workflow step is not registered";
  }
  return step.blockedReason ?? `${step.title} is ${STEP_STATUS_LABELS[step.status]}`;
}
