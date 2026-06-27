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
export type OperatorWorkbenchEvidenceTarget = "project" | "compile" | "synthesis";

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
  compileBundleExported: boolean;
  synthesisComplete: boolean;
  synthesisBundleExported: boolean;
  projectBundleExported: boolean;
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
  evidenceExportTarget: OperatorWorkbenchEvidenceTarget | null;
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
  const exportTarget = selectedEvidenceTarget(inputs);
  return {
    cards: [
      workspaceCard(inputs),
      modelCard(inputs),
      simulationCard(inputs),
      evidenceCard(inputs),
      compileCard(inputs),
      exportCard(inputs),
    ],
    evidenceActionEnabled: exportTarget !== null,
    evidenceExportTarget: exportTarget,
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
  const exportTarget = selectedEvidenceTarget(inputs);
  if (exportTarget === null) {
    return {
      action: "Save project first",
      detail: "Save a project before exporting project evidence",
      key: "export",
      status: "blocked",
      title: "Export",
      value: "Not available",
    };
  }
  const exported = bundleExported(inputs, exportTarget);
  const label = evidenceTargetLabel(exportTarget);
  if (exported) {
    return {
      action: `Open ${label} bundle`,
      detail: `${label} evidence bundle is ready for artifact download`,
      key: "export",
      status: "ready",
      title: "Export",
      value: "Bundle ready",
    };
  }
  return {
    action: `Export ${label} bundle`,
    detail: evidenceTargetDetail(exportTarget, guidedStep(inputs.guidedFlow, "export")),
    key: "export",
    status: "ready",
    title: "Export",
    value: `${label} scope`,
  };
}

function selectedEvidenceTarget(
  inputs: OperatorWorkbenchInputs,
): OperatorWorkbenchEvidenceTarget | null {
  if (inputs.synthesisComplete) {
    return "synthesis";
  }
  if (inputs.compileComplete) {
    return "compile";
  }
  if (inputs.projectName !== null) {
    return "project";
  }
  return null;
}

function bundleExported(
  inputs: OperatorWorkbenchInputs,
  target: OperatorWorkbenchEvidenceTarget,
): boolean {
  switch (target) {
    case "compile":
      return inputs.compileBundleExported;
    case "project":
      return inputs.projectBundleExported;
    case "synthesis":
      return inputs.synthesisBundleExported;
  }
}

function evidenceTargetLabel(target: OperatorWorkbenchEvidenceTarget): string {
  switch (target) {
    case "compile":
      return "compile";
    case "project":
      return "project";
    case "synthesis":
      return "synthesis";
  }
}

function evidenceTargetDetail(
  target: OperatorWorkbenchEvidenceTarget,
  exportStep: GuidedFlowState["steps"][number] | null,
): string {
  switch (target) {
    case "compile":
      return "Bundle compile traceability, audit excerpt, and RTL provenance";
    case "project":
      return stepDetail(exportStep);
    case "synthesis":
      return "Bundle the latest synthesis job, audit excerpt, and hardware artifacts";
  }
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
