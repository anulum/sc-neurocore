// @vitest-environment happy-dom
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio side-by-side comparison names itself and its limit

import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import type { ModelSummary } from "../api/client";
import { useStudioStore } from "../stores/studio";
import ModelComparison, { SIDE_BY_SIDE_LIMIT } from "./ModelComparison";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const NAMES = ["AdExNeuron", "HodgkinHuxleyNeuron", "IzhikevichNeuron", "LIFNeuron", "WangBuzsakiNeuron"];

describe("ModelComparison", () => {
  const initial = useStudioStore.getState();
  afterEach(() => {
    useStudioStore.setState(initial, true);
  });

  it("is a named section whose checkboxes form a named group", async () => {
    useStudioStore.setState({ models: NAMES.map((name) => ({ name }) as ModelSummary), selectedModelName: "LIFNeuron" });
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => { root.render(<ModelComparison />); });

    const heading = container.querySelector("h2");
    expect(heading?.textContent).toBe(`Side-by-side (1 of ${String(SIDE_BY_SIDE_LIMIT)})`);
    const group = container.querySelector('[role="group"]');
    expect(group?.getAttribute("aria-label")).toBe("Models to compare side by side");
    expect(group?.querySelectorAll('input[type="checkbox"]')).toHaveLength(NAMES.length);

    await act(async () => { root.unmount(); });
    container.remove();
  });

  it("disables the remaining boxes once the table is full instead of ignoring a fifth click", async () => {
    useStudioStore.setState({ models: NAMES.map((name) => ({ name }) as ModelSummary), selectedModelName: "LIFNeuron" });
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => { root.render(<ModelComparison />); });

    const boxes = [...container.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')];
    for (const box of boxes.slice(0, SIDE_BY_SIDE_LIMIT)) {
      await act(async () => { box.click(); });
    }

    const after = [...container.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')];
    expect(after.filter((box) => box.checked)).toHaveLength(SIDE_BY_SIDE_LIMIT);
    expect(after[SIDE_BY_SIDE_LIMIT]?.disabled).toBe(true);
    expect(after[0]?.disabled).toBe(false);
    expect(container.querySelector("h2")?.textContent).toBe(`Side-by-side (4 of ${String(SIDE_BY_SIDE_LIMIT)})`);

    await act(async () => { root.unmount(); });
    container.remove();
  });
});
