// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — the view switcher as mounted: its tabs, its keys and its render timing mark
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ViewTab } from "../stores/studio";
import { viewTabGroups, viewTabId, VIEW_PANEL_ID } from "../viewTabs";
import ViewTabs from "./ViewTabs";
import type { ViewTabAvailability } from "./ViewTabs";

const GROUPS = viewTabGroups({ sourceMode: "model", hasPhase: true, hasIsi: true });
const ALL_AVAILABLE = (): ViewTabAvailability => ({ available: true, message: "" });

let root: Root | undefined;
let container: HTMLDivElement;

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root?.unmount(); await Promise.resolve(); });
  document.body.replaceChildren();
  performance.clearMarks();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

/**
 * Render the switcher with one view active.
 *
 * @param active - The active view.
 * @param onSelect - What a choice does.
 * @param availability - Each view's availability.
 */
async function show(
  active: ViewTab,
  onSelect: (view: ViewTab) => void = () => undefined,
  availability: (view: ViewTab) => ViewTabAvailability = ALL_AVAILABLE,
): Promise<void> {
  await act(async () => {
    root?.render(<ViewTabs active={active} availability={availability} groups={GROUPS} onSelect={onSelect} />);
    await Promise.resolve();
  });
}

/** The view-drawn marks recorded so far. */
function viewMarks(): number {
  return performance.getEntriesByName("studio:view-drawn", "mark").length;
}

describe("ViewTabs", () => {
  it("is one tab list whose selected tab alone is in the Tab order and controls the view panel", async () => {
    await show("trace");
    const tabs = [...container.querySelectorAll<HTMLButtonElement>("[role=tab]")];
    expect(container.querySelectorAll("[role=tablist]")).toHaveLength(1);
    expect(tabs.length).toBe(GROUPS.flatMap((group) => group.views).length);
    const selected = tabs.filter((tab) => tab.getAttribute("aria-selected") === "true");
    expect(selected.map((tab) => tab.id)).toEqual([viewTabId("trace")]);
    expect(tabs.filter((tab) => tab.tabIndex === 0)).toEqual(selected);
    expect(tabs.every((tab) => tab.getAttribute("aria-controls") === VIEW_PANEL_ID)).toBe(true);
  });

  it("chooses a view by click and by arrow key, moving the focus with it", async () => {
    const onSelect = vi.fn();
    await show("trace", onSelect);
    const tabs = [...container.querySelectorAll<HTMLButtonElement>("[role=tab]")];
    const second = tabs[1];
    if (!second) throw new Error("the switcher has one tab only");
    await act(async () => { second.click(); await Promise.resolve(); });
    expect(onSelect).toHaveBeenLastCalledWith(GROUPS.flatMap((group) => group.views)[1]?.view);
    await act(async () => {
      tabs[0]?.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
      await Promise.resolve();
    });
    expect(document.activeElement).toBe(second);
    expect(onSelect).toHaveBeenCalledTimes(2);
  });

  it("keeps an unavailable view out of reach and says why", async () => {
    await show("trace", undefined, (view) => (view === "trace" ? ALL_AVAILABLE() : { available: false, message: "Run first" }));
    const disabled = [...container.querySelectorAll<HTMLButtonElement>("[role=tab]")].filter((tab) => tab.disabled);
    expect(disabled.length).toBeGreaterThan(0);
    expect(disabled.every((tab) => tab.title === "Run first")).toBe(true);
  });

  it("marks a drawn view once per change of the active view, and not when first shown", async () => {
    await show("trace");
    expect(viewMarks()).toBe(0);
    await show("trace");
    expect(viewMarks()).toBe(0);
    await show("phase");
    expect(viewMarks()).toBe(1);
    await show("phase");
    await show("isi");
    expect(viewMarks()).toBe(2);
  });

  it("draws the same markup whether or not the timeline takes marks", async () => {
    await show("trace");
    await show("phase");
    const marked = container.innerHTML;
    await act(async () => { root?.unmount(); await Promise.resolve(); });
    root = createRoot(container);
    vi.spyOn(performance, "mark").mockImplementation(() => {
      throw new Error("no timeline");
    });
    await show("trace");
    await show("phase");
    expect(container.innerHTML).toBe(marked);
  });
});
