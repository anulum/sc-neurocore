// @vitest-environment happy-dom
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio status bar names the run in progress

import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import { useStudioStore } from "../stores/studio";
import StatusBar from "./StatusBar";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/**
 * Render the status bar with the live store and return its text.
 *
 * @returns The rendered text.
 */
async function statusText(): Promise<string> {
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  await act(async () => { root.render(<StatusBar />); });
  const text = container.textContent;
  await act(async () => { root.unmount(); });
  container.remove();
  return text;
}

describe("StatusBar", () => {
  const initial = useStudioStore.getState();
  afterEach(() => {
    useStudioStore.setState(initial, true);
  });

  it("names the run that holds the Studio instead of calling every run a simulation", async () => {
    useStudioStore.setState({ isSimulating: true, busyWith: "Network pipeline" });
    const pipeline = await statusText();
    expect(pipeline).toContain("Network pipeline running…");
    expect(pipeline).not.toContain("simulating");

    useStudioStore.setState({ isSimulating: true, busyWith: "Simulation" });
    expect(await statusText()).toContain("Simulation running…");
  });
});
