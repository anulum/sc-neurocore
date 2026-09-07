// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio guided default-flow state machine

/**
 * The guided path from a model to exported evidence, as a state machine.
 *
 * Eight steps, each unblocked by the one before it. The order is the argument
 * the workflow makes: you cannot analyse a run you have not done, cannot claim
 * parity for RTL you have not compiled, and cannot export evidence of a
 * synthesis that has not happened. A step is `blocked` with a stated reason
 * rather than merely disabled, because "why can I not click this" is the
 * question the reader actually has.
 *
 * Two steps bend the chain. `train` is optional and may be skipped, which is
 * why a later step can be `available` while an earlier one is not complete.
 * `cosim` is dropped entirely when co-simulation does not apply to the run --
 * removed from the flow, not shown as permanently blocked, because a step that
 * can never complete would make the count of remaining work a lie.
 */

/** The eight steps of the guided workflow, in order. */
export type GuidedFlowStepKey =
  | "design"
  | "simulate"
  | "analyse"
  | "train"
  | "compile"
  | "cosim"
  | "synthesise"
  | "export";

/**
 * Where a step stands. `current` is the one to do next; `available` is a later
 * step that is reachable anyway, which happens when an optional step is
 * skipped.
 */
export type GuidedFlowStepStatus = "completed" | "current" | "available" | "blocked";

/** Accomplished-evidence facts that drive the guided flow, derived from the store. */
export interface GuidedFlowInputs {
  modelSelected: boolean;
  simulationComplete: boolean;
  analysisComplete: boolean;
  trainingComplete: boolean;
  trainingSkipped: boolean;
  compileComplete: boolean;
  cosimApplicable: boolean;
  cosimComplete: boolean;
  synthesisComplete: boolean;
  evidenceExported: boolean;
}

/** Per-step capability availability from the Studio capability registry. */
export type GuidedFlowCapabilityMap = Record<GuidedFlowStepKey, boolean>;

/** One step: what it is, where it stands, and why it is blocked. */
export interface GuidedFlowStep {
  key: GuidedFlowStepKey;
  title: string;
  optional: boolean;
  status: GuidedFlowStepStatus;
  blockedReason: string | null;
}

/** The whole flow: its steps, the current one, and how far it has got. */
export interface GuidedFlowState {
  steps: GuidedFlowStep[];
  currentStepKey: GuidedFlowStepKey | null;
  completedCount: number;
  totalCount: number;
}

/** A step's fixed properties: its name, whether it is optional, what it needs. */
interface GuidedFlowStepDefinition {
  key: GuidedFlowStepKey;
  title: string;
  optional: boolean;
  /** The step whose completion unblocks this one, or null for the entry step. */
  requires: GuidedFlowStepKey | null;
}

/** The workflow itself, in the order the steps must happen. */
const GUIDED_FLOW_STEPS: readonly GuidedFlowStepDefinition[] = [
  { key: "design", title: "Design", optional: false, requires: null },
  { key: "simulate", title: "Simulate", optional: false, requires: "design" },
  { key: "analyse", title: "Analyse", optional: false, requires: "simulate" },
  { key: "train", title: "Train", optional: true, requires: "analyse" },
  { key: "compile", title: "Compile", optional: false, requires: "analyse" },
  { key: "cosim", title: "Co-sim parity", optional: false, requires: "compile" },
  { key: "synthesise", title: "Synthesise", optional: false, requires: "cosim" },
  { key: "export", title: "Export evidence", optional: false, requires: "synthesise" },
];

/**
 * Assume every step's capability is available.
 *
 * The default for callers that do not consult the registry, so a caller
 * that has not asked is not silently locked out of every step.
 *
 * @returns A map marking every step available.
 */
function allCapabilitiesAvailable(): GuidedFlowCapabilityMap {
  return {
    design: true,
    simulate: true,
    analyse: true,
    train: true,
    compile: true,
    cosim: true,
    synthesise: true,
    export: true,
  };
}

/**
 * Whether one step's work has been done.
 *
 * @param key - The step.
 * @param inputs - What the store says has been accomplished.
 * @returns Whether it counts as complete. `train` counts as complete when
 *   it was skipped: the reader made that decision and the flow should not
 *   keep asking.
 */
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
    case "cosim":
      return inputs.cosimComplete;
    case "synthesise":
      return inputs.synthesisComplete;
    case "export":
      return inputs.evidenceExported;
  }
}

/**
 * The name a step is shown under.
 *
 * @param key - The step.
 * @returns Its title.
 */
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
 *
 * @param inputs - What the store says has been accomplished.
 * @param capabilities - Which steps the deployment can actually perform;
 *   every step is assumed available when this is not given.
 * @returns The flow: its steps with their statuses, the current one, and how
 *   many of them are done.
 */
export function computeGuidedFlowState(
  inputs: GuidedFlowInputs,
  capabilities: GuidedFlowCapabilityMap = allCapabilitiesAvailable(),
): GuidedFlowState {
  let currentAssigned = false;
  const definitions = GUIDED_FLOW_STEPS.filter(
    (definition) => definition.key !== "cosim" || inputs.cosimApplicable,
  );
  const steps: GuidedFlowStep[] = definitions.map((definition) => {
    const requiredStep = definition.key === "synthesise" && !inputs.cosimApplicable
      ? "compile"
      : definition.requires;
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
    if (requiredStep !== null && !isStepComplete(requiredStep, inputs)) {
      return {
        ...stepBase(definition),
        status: "blocked",
        blockedReason: `Requires ${titleOf(requiredStep)}`,
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

/**
 * The fields a step carries whatever its status.
 *
 * @param definition - The step's definition.
 * @returns Its key, title and optionality.
 */
function stepBase(
  definition: GuidedFlowStepDefinition,
): Pick<GuidedFlowStep, "key" | "title" | "optional"> {
  return { key: definition.key, title: definition.title, optional: definition.optional };
}
