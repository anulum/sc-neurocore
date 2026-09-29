// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — tests for the plot views' empty states

import { describe, expect, it } from "vitest";

import type { ViewTab } from "./stores/studio";
import { emptyViewDescription, emptyViewGuidance } from "./viewEmptyState";
import type { PlotViewFacts } from "./viewEmptyState";

const NOTHING: PlotViewFacts = {
  hasRun: true,
  stateCount: 3,
  hasIsiHistogram: true,
  spikeCount: 12,
  sweepX: "",
  sweepY: "",
  hasFi: false,
  hasBifurcation: false,
  hasHeatmap: false,
  hasSensitivity: false,
  hasPrecision: false,
  hasCompare: false,
  hasFrequency: false,
  hasSta: false,
  hasCharacterization: false,
  hasMulti: false,
  hasNetwork: false,
};

const EVERYTHING: PlotViewFacts = {
  ...NOTHING,
  hasFi: true,
  hasBifurcation: true,
  hasHeatmap: true,
  hasSensitivity: true,
  hasPrecision: true,
  hasCompare: true,
  hasFrequency: true,
  hasSta: true,
  hasCharacterization: true,
  hasMulti: true,
  hasNetwork: true,
};

const PLOT_VIEWS: ViewTab[] = [
  "fi-curve", "bifurcation", "heatmap", "sensitivity", "precision", "compare",
  "freq", "sta", "characterize", "multi", "network",
];

describe("plot view empty states", () => {
  it("says every result-backed view is empty instead of drawing the trace", () => {
    for (const view of PLOT_VIEWS) {
      const guidance = emptyViewGuidance(view, NOTHING);
      expect(guidance, view).not.toBeNull();
      expect(guidance?.title, view).toMatch(/^No /);
      expect(guidance?.detail.length, view).toBeGreaterThan(20);
    }
  });

  it("gets out of the way once the view has its own result", () => {
    for (const view of PLOT_VIEWS) expect(emptyViewGuidance(view, EVERYTHING), view).toBeNull();
    expect(emptyViewGuidance("trace", NOTHING)).toBeNull();
    expect(emptyViewGuidance("phase", NOTHING)).toBeNull();
    expect(emptyViewGuidance("isi", NOTHING)).toBeNull();
  });

  it("offers the action that produces the result, named as in the header", () => {
    expect(emptyViewGuidance("characterize", NOTHING)).toMatchObject({ action: "characterize", actionLabel: "Characterize" });
    expect(emptyViewGuidance("freq", NOTHING)).toMatchObject({ action: "freq", actionLabel: "Measure frequency response" });
    expect(emptyViewGuidance("sta", NOTHING)).toMatchObject({ action: "sta", actionLabel: "Compute STA" });
    expect(emptyViewGuidance("network", NOTHING)).toMatchObject({ action: "network", actionLabel: "Run E-I network" });
    expect(emptyViewGuidance("precision", NOTHING)).toMatchObject({ action: "precision", actionLabel: "Check Q8.8 precision" });
    for (const view of ["fi-curve", "bifurcation", "heatmap", "sensitivity"] as const) {
      expect(emptyViewGuidance(view, NOTHING)?.detail, view).toContain("Run async analysis");
    }
  });

  it("asks for the sweep parameters a sweep still lacks", () => {
    expect(emptyViewGuidance("bifurcation", NOTHING)?.detail).toContain("Choose a parameter in Sweep X");
    expect(emptyViewGuidance("bifurcation", { ...NOTHING, sweepX: "tau_m" })?.detail).toContain("To sweep tau_m");
    expect(emptyViewGuidance("heatmap", { ...NOTHING, sweepX: "tau_m" })?.detail).toContain("Sweep X and Sweep Y");
    expect(emptyViewGuidance("heatmap", { ...NOTHING, sweepX: "a", sweepY: "b" })?.detail).toContain("a against b");
  });

  it("explains an STA that cannot be computed from too few spikes, with no button", () => {
    const guidance = emptyViewGuidance("sta", { ...NOTHING, spikeCount: 1 });
    expect(guidance?.detail).toContain("has 1 spike; the average needs at least three");
    expect(guidance?.action).toBeNull();
  });

  it("explains a phase plane or ISI that this run cannot fill", () => {
    expect(emptyViewGuidance("phase", { ...NOTHING, stateCount: 1 })?.detail).toContain("two state variables");
    expect(emptyViewGuidance("isi", { ...NOTHING, hasIsiHistogram: false })?.detail).toContain("fewer than two spikes");
  });

  it("gives a screen reader the view's name, the state and what to do", () => {
    const guidance = emptyViewGuidance("network", NOTHING);
    expect(guidance).not.toBeNull();
    if (guidance === null) return;
    expect(emptyViewDescription("network", guidance)).toBe(
      "E-I network: No E-I network run yet. Simulates a balanced excitatory-inhibitory network and shows its raster and rates.",
    );
  });
});
