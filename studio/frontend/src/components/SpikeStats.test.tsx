// @vitest-environment happy-dom
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio spike statistics state numbers, not a second verdict

import { act } from "react";
import type React from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import type { SimulateResponse } from "../api/client";
import { useStudioStore } from "../stores/studio";
import SpikeStats from "./SpikeStats";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/**
 * Render into the document with the live store and return the markup.
 *
 * Server rendering reads a zustand store's initial state, not the state a
 * case sets, so these cases mount into a real document instead.
 *
 * @param element - What to render.
 * @returns The rendered markup.
 */
async function mountedMarkup(element: React.ReactElement): Promise<string> {
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  await act(async () => { root.render(element); });
  const html = container.innerHTML;
  await act(async () => { root.unmount(); });
  container.remove();
  return html;
}

describe("SpikeStats", () => {
  const initial = useStudioStore.getState();
  afterEach(() => {
    useStudioStore.setState(initial, true);
  });

  it("shows the CV without naming a pattern of its own", async () => {
    // The server classified this run as chaotic; the panel used to call the
    // same CV "bursting" from thresholds of its own.
    useStudioStore.setState({
      result: {
        spike_count: 12,
        stats: { rate_hz: 120, isi_mean_ms: 8.682, isi_cv: 1.9246, isi_histogram: null },
        pattern: { pattern: "chaotic", description: "Highly irregular/chaotic (CV=1.925)" },
      } as unknown as SimulateResponse,
    });

    const html = await mountedMarkup(<SpikeStats />);

    expect(html).toContain(">1.9246<");
    for (const verdict of ["bursting", "regular", "irregular", "chaotic"]) {
      expect(html).not.toContain(verdict);
    }
    expect(html).not.toContain("var(--error)");
  });

  it("renders nothing before a run", async () => {
    useStudioStore.setState({ result: null });
    expect(await mountedMarkup(<SpikeStats />)).toBe("");
  });
});
