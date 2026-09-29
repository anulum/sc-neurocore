// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — what a plot view says when it has nothing of its own to show

/**
 * The empty state of each plot view.
 *
 * A plot view with no result of its own used to draw the membrane trace
 * instead, under its own tab: the 2-D sweep, bifurcation, STA, E-I network and
 * the other views showed the voltage trace while their tab was selected, with
 * nothing on screen saying it was not their result, and a screen reader was
 * given the trace's description. Each view now says that it is empty and what
 * produces its result.
 */

import { panelTitle } from "./capabilityShell";
import type { ViewTab } from "./stores/studio";

/** A store action the empty state can offer as its button. */
export type EmptyViewAction = "characterize" | "freq" | "sta" | "network" | "precision";

/** What an empty view shows. */
export interface EmptyViewGuidance {
  /** A heading such as "No bifurcation diagram yet". */
  title: string;
  /** What produces the result, naming the control to use. */
  detail: string;
  /** The action that produces the result directly, when there is one. */
  action: EmptyViewAction | null;
  /** The action button's label, matching the header's. */
  actionLabel: string | null;
}

/** Which results the plot views currently hold. */
export interface PlotViewFacts {
  hasRun: boolean;
  stateCount: number;
  hasIsiHistogram: boolean;
  spikeCount: number;
  sweepX: string;
  sweepY: string;
  hasFi: boolean;
  hasBifurcation: boolean;
  hasHeatmap: boolean;
  hasSensitivity: boolean;
  hasPrecision: boolean;
  hasCompare: boolean;
  hasFrequency: boolean;
  hasSta: boolean;
  hasCharacterization: boolean;
  hasMulti: boolean;
  hasNetwork: boolean;
}

const RUN_ANALYSIS = "click Run async analysis in the action row";

/**
 * The guidance a plot view shows instead of a plot, or `null` when it has a
 * result to draw (or is the trace, which draws the run itself).
 *
 * @param view - The selected view.
 * @param facts - What the views currently hold.
 * @returns The empty state, or `null`.
 */
export function emptyViewGuidance(view: ViewTab, facts: PlotViewFacts): EmptyViewGuidance | null {
  const empty = (
    title: string,
    detail: string,
    action: EmptyViewAction | null = null,
    actionLabel: string | null = null,
  ): EmptyViewGuidance => ({ title, detail, action, actionLabel });
  switch (view) {
    case "phase":
      return facts.hasRun && facts.stateCount < 2
        ? empty("No phase plane for this run", "A phase plane needs at least two state variables; this run has one.")
        : null;
    case "isi":
      return facts.hasRun && !facts.hasIsiHistogram
        ? empty("No inter-spike intervals", "The current run has fewer than two spikes, so there is no interval to histogram.")
        : null;
    case "fi-curve":
      return facts.hasFi
        ? null
        : empty("No f-I curve yet", `To measure the firing rate over a range of input currents, ${RUN_ANALYSIS}.`);
    case "bifurcation":
      if (facts.hasBifurcation) return null;
      return facts.sweepX === ""
        ? empty("No bifurcation diagram yet", `Choose a parameter in Sweep X, then ${RUN_ANALYSIS}.`)
        : empty("No bifurcation diagram yet", `To sweep ${facts.sweepX}, ${RUN_ANALYSIS}.`);
    case "heatmap":
      if (facts.hasHeatmap) return null;
      return facts.sweepX === "" || facts.sweepY === ""
        ? empty("No 2-D sweep yet", `Choose two different parameters in Sweep X and Sweep Y, then ${RUN_ANALYSIS}.`)
        : empty("No 2-D sweep yet", `To sweep ${facts.sweepX} against ${facts.sweepY}, ${RUN_ANALYSIS}.`);
    case "sensitivity":
      return facts.hasSensitivity
        ? null
        : empty("No sensitivity analysis yet", `To see how the firing rate responds to each parameter, ${RUN_ANALYSIS}.`);
    case "precision":
      return facts.hasPrecision
        ? null
        : empty("No precision comparison yet", "Compares the float run with its Q8.8 fixed-point version.", "precision", "Check Q8.8 precision");
    case "compare":
      return facts.hasCompare
        ? null
        : empty("No A/B comparison yet", "Choose model B in the A/B compare section of the left panel.");
    case "freq":
      return facts.hasFrequency
        ? null
        : empty("No frequency response yet", "Drives the model with sinusoidal input over a range of frequencies.", "freq", "Measure frequency response");
    case "sta":
      if (facts.hasSta) return null;
      return facts.spikeCount < 3
        ? empty("No spike-triggered average", `The current run has ${String(facts.spikeCount)} ${facts.spikeCount === 1 ? "spike" : "spikes"}; the average needs at least three. Increase the input current and run again.`)
        : empty("No spike-triggered average yet", "Averages the input over a window before each spike.", "sta", "Compute STA");
    case "characterize":
      return facts.hasCharacterization
        ? null
        : empty("No characterization yet", "Firing pattern, f-I curve and the most sensitive parameters in one run.", "characterize", "Characterize");
    case "multi":
      return facts.hasMulti
        ? null
        : empty("No models to overlay yet", "Pick up to four models in the Multi-model section of the left panel.");
    case "network":
      return facts.hasNetwork
        ? null
        : empty("No E-I network run yet", "Simulates a balanced excitatory-inhibitory network and shows its raster and rates.", "network", "Run E-I network");
    default:
      return null;
  }
}

/**
 * What a screen reader hears for an empty view, in place of a plot description.
 *
 * @param view - The selected view.
 * @param guidance - Its empty state.
 * @returns The sentence.
 */
export function emptyViewDescription(view: ViewTab, guidance: EmptyViewGuidance): string {
  return `${panelTitle(view)}: ${guidance.title}. ${guidance.detail}`;
}
