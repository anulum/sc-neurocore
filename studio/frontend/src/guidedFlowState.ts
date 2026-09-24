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
 * synthesis that has not happened.
 *
 * Every status says something different and none borrows another's meaning.
 * `completed` is evidence for the current inputs; nothing else counts as done.
 * `skipped` is the reader's decision about the optional `train` step: it lets
 * the flow continue but is not evidence. `not_applicable` is a step the run
 * cannot have -- co-simulation of an equation typed by hand -- shown with its
 * reason and left out of the count, because a step that can never complete
 * would make the count of remaining work a lie. `unsupported` is a step this
 * deployment cannot perform, named with the capability's own message.
 * `failed` is a step whose latest attempt for the current inputs failed; it
 * stays the thing to do next, as a retry. `blocked` waits on an earlier step,
 * `current` is what to do next, and `available` is a later step reachable
 * anyway because the optional step before it was left undone.
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

/** Where a step stands; see the module description for what each one means. */
export type GuidedFlowStepStatus =
  | "completed"
  | "skipped"
  | "not_applicable"
  | "unsupported"
  | "failed"
  | "current"
  | "available"
  | "blocked";

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
  /**
   * Why a step's latest attempt for the current inputs failed. A step absent
   * here has not failed for these inputs, even if an earlier experiment did.
   */
  failures?: Partial<Record<GuidedFlowStepKey, string>>;
}

/** Per-step capability availability from the Studio capability registry. */
export type GuidedFlowCapabilityMap = Record<GuidedFlowStepKey, boolean>;

/** One step: what it is, where it stands, and why when it is not simply done or next. */
export interface GuidedFlowStep {
  key: GuidedFlowStepKey;
  title: string;
  optional: boolean;
  status: GuidedFlowStepStatus;
  /** Why the step is blocked, unsupported, failed, skipped or not applicable. */
  reason: string | null;
}

/** The whole flow: its steps, what to do next, and how far it has got. */
export interface GuidedFlowState {
  steps: GuidedFlowStep[];
  /** The step to act on next -- a `current` step or a `failed` one to retry. */
  currentStepKey: GuidedFlowStepKey | null;
  /** Steps with evidence for the current inputs; skipped steps are not counted. */
  completedCount: number;
  /** Optional steps the reader chose to skip. */
  skippedCount: number;
  /** Steps that apply to this run; `not_applicable` steps are excluded. */
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

/** Why co-simulation is left out of a run it cannot apply to. */
export const COSIM_NOT_APPLICABLE_REASON = "Co-simulation applies to catalogue models only";

/** Why the optional training step shows as skipped. */
export const TRAINING_SKIPPED_REASON = "by choice; not training evidence";

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
 * Whether a step has evidence for the current inputs.
 *
 * @param key - The step.
 * @param inputs - What the store says has been accomplished.
 * @returns Whether it is complete. A skipped `train` step is not.
 */
function hasEvidence(key: GuidedFlowStepKey, inputs: GuidedFlowInputs): boolean {
  switch (key) {
    case "design":
      return inputs.modelSelected;
    case "simulate":
      return inputs.simulationComplete;
    case "analyse":
      return inputs.analysisComplete;
    case "train":
      return inputs.trainingComplete;
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
 * Whether a later step may proceed past this one.
 *
 * @param key - The step a later one requires.
 * @param inputs - What the store says has been accomplished.
 * @returns Whether it has evidence, or is the optional step the reader skipped.
 */
function satisfies(key: GuidedFlowStepKey, inputs: GuidedFlowInputs): boolean {
  return hasEvidence(key, inputs) || (key === "train" && inputs.trainingSkipped);
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
 * Decide one step's status before `current` and `available` are told apart.
 *
 * @param definition - The step.
 * @param inputs - What the store says has been accomplished.
 * @param capabilities - Which steps the deployment can perform.
 * @param unsupportedReasons - The capability registry's own message per step.
 * @returns The status and its reason; `current` stands for any actionable step.
 */
function classify(
  definition: GuidedFlowStepDefinition,
  inputs: GuidedFlowInputs,
  capabilities: GuidedFlowCapabilityMap,
  unsupportedReasons: Partial<Record<GuidedFlowStepKey, string>>,
): { status: GuidedFlowStepStatus; reason: string | null } {
  const { key } = definition;
  if (key === "cosim" && !inputs.cosimApplicable) {
    return { status: "not_applicable", reason: COSIM_NOT_APPLICABLE_REASON };
  }
  if (hasEvidence(key, inputs)) {
    return { status: "completed", reason: null };
  }
  if (key === "train" && inputs.trainingSkipped) {
    return { status: "skipped", reason: TRAINING_SKIPPED_REASON };
  }
  if (!capabilities[key]) {
    return {
      status: "unsupported",
      reason: unsupportedReasons[key] ?? `${definition.title} capability is unavailable`,
    };
  }
  const required = key === "synthesise" && !inputs.cosimApplicable ? "compile" : definition.requires;
  if (required !== null && !satisfies(required, inputs)) {
    return { status: "blocked", reason: `Requires ${titleOf(required)}` };
  }
  const failure = inputs.failures?.[key];
  if (failure !== undefined) {
    return { status: "failed", reason: failure };
  }
  return { status: "current", reason: null };
}

/**
 * Compute the guided default-flow state from accomplished evidence and
 * per-step capability availability.
 *
 * The earliest actionable step -- one that is `current` or `failed` -- is what
 * to do next; any later actionable step, reachable because the optional
 * `train` step before it was left undone, is `available`. A failed step keeps
 * its status so the reader sees why a retry is offered.
 *
 * @param inputs - What the store says has been accomplished and what failed.
 * @param capabilities - Which steps the deployment can actually perform;
 *   every step is assumed available when this is not given.
 * @param unsupportedReasons - The capability registry's message for a step
 *   the deployment cannot perform, used in place of a generic reason.
 * @returns The flow: its steps with their statuses, the step to act on next,
 *   and how many of them are done or skipped.
 */
export function computeGuidedFlowState(
  inputs: GuidedFlowInputs,
  capabilities: GuidedFlowCapabilityMap = allCapabilitiesAvailable(),
  unsupportedReasons: Partial<Record<GuidedFlowStepKey, string>> = {},
): GuidedFlowState {
  let currentStepKey: GuidedFlowStepKey | null = null;
  const steps: GuidedFlowStep[] = GUIDED_FLOW_STEPS.map((definition) => {
    const { status, reason } = classify(definition, inputs, capabilities, unsupportedReasons);
    const base = { key: definition.key, title: definition.title, optional: definition.optional };
    if (status !== "current" && status !== "failed") {
      return { ...base, status, reason };
    }
    if (currentStepKey === null) {
      currentStepKey = definition.key;
      return { ...base, status, reason };
    }
    return { ...base, status: status === "failed" ? "failed" : "available", reason };
  });

  const applicable = steps.filter((step) => step.status !== "not_applicable");
  return {
    steps,
    currentStepKey,
    completedCount: applicable.filter((step) => step.status === "completed").length,
    skippedCount: applicable.filter((step) => step.status === "skipped").length,
    totalCount: applicable.length,
  };
}
