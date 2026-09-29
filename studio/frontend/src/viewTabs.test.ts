// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — tests for the view switcher's list and keyboard movement

import { describe, expect, it } from "vitest";

import { nextViewTabIndex, viewTabGroups, viewTabId, VIEW_PANEL_ID } from "./viewTabs";
import type { ViewTabContext } from "./viewTabs";

const MODEL: ViewTabContext = { sourceMode: "model", hasPhase: true, hasIsi: true };
const ODE: ViewTabContext = { sourceMode: "ode", hasPhase: false, hasIsi: false };

/** The views listed for one context, in order. */
function views(context: ViewTabContext): string[] {
  return viewTabGroups(context).flatMap((group) => group.views.map((entry) => entry.view));
}

/** The names shown for one context, in order. */
function labels(context: ViewTabContext): string[] {
  return viewTabGroups(context).flatMap((group) => group.views.map((entry) => entry.label));
}

describe("view switcher list", () => {
  it("names every view once, in full, with no repeated name", () => {
    for (const context of [MODEL, ODE]) {
      const names = labels(context);
      expect(new Set(names).size).toBe(names.length);
      expect(new Set(views(context)).size).toBe(names.length);
      for (const abbreviation of ["Bif", "Sens", "2D", "Char", "Char.", "Train", "Freq", "E-I", "Multi"]) {
        expect(names).not.toContain(abbreviation);
      }
    }
  });

  it("shows phase plane and ISI only when the result can fill them", () => {
    expect(views(MODEL)).toEqual(expect.arrayContaining(["phase", "isi"]));
    expect(views({ ...MODEL, hasPhase: false, hasIsi: false })).not.toEqual(
      expect.arrayContaining(["phase"]),
    );
    expect(views({ ...MODEL, hasPhase: false, hasIsi: false })).not.toContain("isi");
  });

  it("keeps each source's own views and lists RTL and synthesis for both", () => {
    expect(views(MODEL)).toContain("characterize");
    expect(views(MODEL)).not.toContain("precision");
    expect(views(MODEL)).not.toContain("ir");
    expect(views(ODE)).not.toContain("characterize");
    expect(views(ODE)).toEqual(expect.arrayContaining(["precision", "ir"]));
    for (const context of [MODEL, ODE]) {
      expect(views(context)).toEqual(expect.arrayContaining(["verilog", "synth"]));
    }
  });

  it("groups the views under names and leaves no group empty", () => {
    const groups = viewTabGroups(MODEL);
    expect(groups.map((group) => group.label)).toEqual([
      "Neuron",
      "Network",
      "Code and hardware",
      "Research",
      "Operator",
    ]);
    for (const context of [MODEL, ODE]) {
      for (const group of viewTabGroups(context)) expect(group.views.length).toBeGreaterThan(0);
    }
  });

  it("gives each tab an id the panel can point back to", () => {
    expect(viewTabId("fi-curve")).toBe("studio-view-tab-fi-curve");
    expect(VIEW_PANEL_ID).toBe("studio-view-panel");
  });
});

describe("arrow-key movement between views", () => {
  const all = [true, true, true, true];

  it("moves to the next and previous view and wraps at the ends", () => {
    expect(nextViewTabIndex("ArrowRight", 0, all)).toBe(1);
    expect(nextViewTabIndex("ArrowRight", 3, all)).toBe(0);
    expect(nextViewTabIndex("ArrowLeft", 0, all)).toBe(3);
    expect(nextViewTabIndex("ArrowLeft", 2, all)).toBe(1);
  });

  it("goes to the first and last enabled view on Home and End", () => {
    expect(nextViewTabIndex("Home", 2, [false, true, true, false])).toBe(1);
    expect(nextViewTabIndex("End", 1, [false, true, true, false])).toBe(2);
  });

  it("skips views that cannot be opened", () => {
    expect(nextViewTabIndex("ArrowRight", 0, [true, false, false, true])).toBe(3);
    expect(nextViewTabIndex("ArrowLeft", 3, [true, false, false, true])).toBe(0);
  });

  it("ignores other keys and a switcher with nothing enabled", () => {
    expect(nextViewTabIndex("Enter", 0, all)).toBeNull();
    expect(nextViewTabIndex("ArrowRight", 0, [])).toBeNull();
    expect(nextViewTabIndex("ArrowRight", 0, [false, false])).toBeNull();
  });
});
