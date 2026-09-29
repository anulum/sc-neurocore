// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — which workbench views exist, what they are called, and how the keyboard moves between them

/**
 * The workbench's views as one ordered, grouped list.
 *
 * The header used to carry twenty-odd abbreviated buttons ("Bif", "Sens",
 * "2D", "Char") with nothing saying which was showing, and repeated four of
 * them ("Canvas", "Train", "Admin", "Freq") as separate buttons with the same
 * names. The list here is the one source for the view switcher: every view
 * has one full name, belongs to one named group, and is shown only when it
 * can hold something for the current source.
 */

import type { ViewTab } from "./stores/studio";

/** One view in the switcher. */
export interface ViewTabDefinition {
  /** The store's key for the view. */
  view: ViewTab;
  /** The name a reader sees and a screen reader announces. */
  label: string;
}

/** A named group of views. */
export interface ViewTabGroup {
  /** The group's name, announced before its first view. */
  label: string;
  views: ViewTabDefinition[];
}

/** What decides whether a view can show anything right now. */
export interface ViewTabContext {
  sourceMode: "model" | "ode";
  /** The current result has at least two state variables. */
  hasPhase: boolean;
  /** The current result carries an inter-spike-interval histogram. */
  hasIsi: boolean;
}

/**
 * The groups and views for the current source, in display order.
 *
 * Phase plane and ISI need a result that can fill them; characterization
 * works on catalogue models only; Q8.8 precision and the IR inspector work on
 * typed equations only. RTL and FPGA synthesis are listed in both modes,
 * because both sources can be compiled and the workbench opens them from
 * either.
 *
 * @param context - The source mode and what the current result holds.
 * @returns The groups, each with at least one view.
 */
export function viewTabGroups(context: ViewTabContext): ViewTabGroup[] {
  const model = context.sourceMode === "model";
  const groups: ViewTabGroup[] = [
    {
      label: "Neuron",
      views: [
        { view: "trace", label: "Trace" },
        ...(context.hasPhase ? [{ view: "phase" as const, label: "Phase plane" }] : []),
        ...(context.hasIsi ? [{ view: "isi" as const, label: "ISI" }] : []),
        { view: "fi-curve", label: "f-I curve" },
        { view: "bifurcation", label: "Bifurcation" },
        { view: "heatmap", label: "2-D sweep" },
        { view: "sensitivity", label: "Sensitivity" },
        { view: "sta", label: "STA" },
        { view: "freq", label: "Frequency response" },
        ...(model ? [{ view: "characterize" as const, label: "Characterization" }] : []),
        { view: "multi", label: "Multi-model" },
        { view: "compare", label: "A/B compare" },
      ],
    },
    {
      label: "Network",
      views: [
        { view: "network", label: "E-I network" },
        { view: "delays", label: "Synaptic delays" },
        { view: "canvas", label: "Network canvas" },
      ],
    },
    {
      label: "Code and hardware",
      views: [
        { view: "code", label: "Python code" },
        ...(model ? [] : [{ view: "precision" as const, label: "Q8.8 precision" }]),
        ...(model ? [] : [{ view: "ir" as const, label: "IR" }]),
        { view: "verilog", label: "RTL" },
        { view: "synth", label: "FPGA synthesis" },
      ],
    },
    {
      label: "Research",
      views: [
        { view: "candidate", label: "Candidate model" },
        { view: "fit", label: "Fitting" },
        { view: "train", label: "Training" },
        { view: "review", label: "Review" },
      ],
    },
    {
      label: "Operator",
      views: [{ view: "admin", label: "Admin" }],
    },
  ];
  return groups;
}

/**
 * The element id of a view's tab, so the panel can name the tab that labels it.
 *
 * @param view - The view.
 * @returns A stable, document-unique id.
 */
export function viewTabId(view: ViewTab): string {
  return `studio-view-tab-${view}`;
}

/** The element id of the one panel every tab controls. */
export const VIEW_PANEL_ID = "studio-view-panel";

/**
 * Where the arrow keys move focus in the view switcher.
 *
 * The WAI-ARIA tabs pattern: Left/Right move to the previous/next enabled
 * tab and wrap around, Home and End go to the first and last enabled tab.
 * Disabled tabs are skipped, because they cannot be activated.
 *
 * @param key - The `KeyboardEvent.key` pressed.
 * @param current - The index of the focused tab.
 * @param enabled - Whether each tab, in order, can be activated.
 * @returns The index to focus, or `null` when the key is not a navigation key
 *   or no tab is enabled.
 */
export function nextViewTabIndex(
  key: string,
  current: number,
  enabled: readonly boolean[],
): number | null {
  const count = enabled.length;
  if (count === 0 || !enabled.some(Boolean)) return null;
  const step = (from: number, direction: 1 | -1): number => {
    let index = from;
    for (let visited = 0; visited < count; visited += 1) {
      index = (index + direction + count) % count;
      if (enabled[index]) return index;
    }
    return from;
  };
  switch (key) {
    case "ArrowRight":
      return step(current, 1);
    case "ArrowLeft":
      return step(current, -1);
    case "Home":
      return step(count - 1, 1);
    case "End":
      return step(0, -1);
    default:
      return null;
  }
}
