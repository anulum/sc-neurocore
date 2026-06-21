// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio guided default-flow state machine

export type GuidedFlowStepKey =
  | "design"
  | "simulate"
  | "analyse"
  | "train"
  | "compile"
  | "synthesise"
  | "export";

export type GuidedFlowStepStatus = "completed" | "current" | "available" | "blocked";

/** Accomplished-evidence facts that drive the guided flow, derived from the store. */
export interface GuidedFlowInputs {
  modelSelected: boolean;
  simulationComplete: boolean;
  analysisComplete: boolean;
  trainingComplete: boolean;
  trainingSkipped: boolean;
  compileComplete: boolean;
  synthesisComplete: boolean;
  evidenceExported: boolean;
}

/** Per-step capability availability from the Studio capability registry. */
export type GuidedFlowCapabilityMap = Record<GuidedFlowStepKey, boolean>;

export interface GuidedFlowStep {
  key: GuidedFlowStepKey;
  title: string;
  optional: boolean;
  status: GuidedFlowStepStatus;
  blockedReason: string | null;
}

export interface GuidedFlowState {
  steps: GuidedFlowStep[];
  currentStepKey: GuidedFlowStepKey | null;
  completedCount: number;
  totalCount: number;
}

interface GuidedFlowStepDefinition {
  key: GuidedFlowStepKey;
  title: string;
  optional: boolean;
  /** The step whose completion unblocks this one, or null for the entry step. */
  requires: GuidedFlowStepKey | null;
}

const GUIDED_FLOW_STEPS: readonly GuidedFlowStepDefinition[] = [
  { key: "design", title: "Design", optional: false, requires: null },
  { key: "simulate", title: "Simulate", optional: false, requires: "design" },
  { key: "analyse", title: "Analyse", optional: false, requires: "simulate" },
  { key: "train", title: "Train", optional: true, requires: "analyse" },
  { key: "compile", title: "Compile", optional: false, requires: "analyse" },
  { key: "synthesise", title: "Synthesise", optional: false, requires: "compile" },
  { key: "export", title: "Export evidence", optional: false, requires: "synthesise" },
];

function allCapabilitiesAvailable(): GuidedFlowCapabilityMap {
  return {
    design: true,
    simulate: true,
    analyse: true,
    train: true,
    compile: true,
    synthesise: true,
    export: true,
  };
}

function isStepComplete(key: GuidedFlowStepKey, inputs: GuidedFlowInputs): boolean {
  switch (key) {
    case "design":
      return inputs.modelSelected;
    case "simulate":
      return inputs.simulationComplete;
    case "analyse":
      return inputs.analysisComplete;
    case "train":
      return inputs.trainingComplete || inputs.trainingSkipped;
    case "compile":
      return inputs.compileComplete;
    case "synthesise":
      return inputs.synthesisComplete;
    case "export":
      return inputs.evidenceExported;
  }
}

function titleOf(key: GuidedFlowStepKey): string {
  const definition = GUIDED_FLOW_STEPS.find((step) => step.key === key);
  return definition ? definition.title : key;
}

/**
 * Compute the guided default-flow state from accomplished evidence and
 * per-step capability availability.
 *
 * A step is `completed` when its evidence exists (`train` also counts as
 * complete when explicitly skipped). A step is `blocked` when its capability is
 * unavailable or its required predecessor is not yet complete, with a concrete
 * `blockedReason`. The earliest actionable step is `current`; any later
 * actionable step (reachable because the optional `train` step can be skipped)
 * is `available`.
 */
export function computeGuidedFlowState(
  inputs: GuidedFlowInputs,
  capabilities: GuidedFlowCapabilityMap = allCapabilitiesAvailable(),
): GuidedFlowState {
  let currentAssigned = false;
  const steps: GuidedFlowStep[] = GUIDED_FLOW_STEPS.map((definition) => {
    const completed = isStepComplete(definition.key, inputs);
    if (completed) {
      return { ...stepBase(definition), status: "completed", blockedReason: null };
    }
    if (!capabilities[definition.key]) {
      return {
        ...stepBase(definition),
        status: "blocked",
        blockedReason: `${definition.title} capability is unavailable`,
      };
    }
    if (definition.requires !== null && !isStepComplete(definition.requires, inputs)) {
      return {
        ...stepBase(definition),
        status: "blocked",
        blockedReason: `Requires ${titleOf(definition.requires)}`,
      };
    }
    if (!currentAssigned) {
      currentAssigned = true;
      return { ...stepBase(definition), status: "current", blockedReason: null };
    }
    return { ...stepBase(definition), status: "available", blockedReason: null };
  });

  const currentStep = steps.find((step) => step.status === "current");
  return {
    steps,
    currentStepKey: currentStep ? currentStep.key : null,
    completedCount: steps.filter((step) => step.status === "completed").length,
    totalCount: steps.length,
  };
}

function stepBase(
  definition: GuidedFlowStepDefinition,
): Pick<GuidedFlowStep, "key" | "title" | "optional"> {
  return { key: definition.key, title: definition.title, optional: definition.optional };
}
