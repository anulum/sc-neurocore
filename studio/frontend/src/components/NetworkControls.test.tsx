// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — the balanced-network controls, mounted with their store
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import NetworkControls from "./NetworkControls";

const store = vi.hoisted(() => ({
  activeTab: "network",
  isSimulating: false,
  networkParams: { n_exc: 80, n_inh: 20, w_ee: 0.1, w_ie: 0.12, w_ei: 0.5, w_ii: 0.4, p_conn: 0.1, ext_rate: 9.5 },
  setNetworkParam: vi.fn(),
  runNetwork: vi.fn(async () => undefined),
}));

vi.mock("../stores/studio", () => ({
  useStudioStore: () => store,
}));

let root: Root | undefined;
let container: HTMLDivElement;

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  store.activeTab = "network";
  store.isSimulating = false;
  store.setNetworkParam.mockClear();
  store.runNetwork.mockClear();
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root?.unmount(); await Promise.resolve(); });
  document.body.replaceChildren();
  vi.unstubAllGlobals();
});

/** Mount the controls. */
async function mount(): Promise<void> {
  await act(async () => { root?.render(<NetworkControls />); await Promise.resolve(); });
}

/**
 * The slider with an accessible name.
 *
 * @param name - The slider's aria-label.
 * @returns The input.
 */
function slider(name: string): HTMLInputElement {
  const input = container.querySelector<HTMLInputElement>(`input[aria-label="${name}"]`);
  if (input === null) throw new Error(`no slider is named ${name}`);
  return input;
}

describe("NetworkControls", () => {
  it("shows itself only in the E-I network view", async () => {
    store.activeTab = "trace";
    await mount();
    expect(container.innerHTML).toBe("");
    store.activeTab = "network";
    await mount();
    expect(container.querySelector(".panel-header")?.textContent).toBe("E-I network");
  });

  it("names every slider and says its value with its unit, at the precision of its step", async () => {
    await mount();
    const read = [...container.querySelectorAll<HTMLInputElement>("input[type=range]")].map((input) => [
      input.getAttribute("aria-label"),
      input.getAttribute("aria-valuetext"),
    ]);
    expect(read).toEqual([
      ["Excitatory neurons", "80"],
      ["Inhibitory neurons", "20"],
      ["Weight excitatory to excitatory", "0.10 mV"],
      ["Weight excitatory to inhibitory", "0.12 mV"],
      ["Weight inhibitory to excitatory", "0.50 mV"],
      ["Weight inhibitory to inhibitory", "0.40 mV"],
      ["Connection probability", "0.10"],
      ["Rate of each external input", "9.5 Hz"],
    ]);
    expect(container.querySelectorAll(".slider-label[aria-hidden=true]")).toHaveLength(8);
  });

  it("binds each weight to the parameter named post-pre, as the model does", async () => {
    await mount();
    for (const [name, param] of [
      ["Weight excitatory to inhibitory", "w_ie"],
      ["Weight inhibitory to excitatory", "w_ei"],
      ["Rate of each external input", "ext_rate"],
    ] as const) {
      store.setNetworkParam.mockClear();
      await act(async () => {
        const input = slider(name);
        // The native setter, as the browser sets a dragged value; React tracks direct assignments and ignores them.
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set?.call(
          input,
          name.startsWith("Rate") ? "12.5" : "0.75",
        );
        input.dispatchEvent(new Event("input", { bubbles: true }));
        await Promise.resolve();
      });
      expect(store.setNetworkParam).toHaveBeenCalledWith(param, name.startsWith("Rate") ? 12.5 : 0.75);
    }
  });

  it("runs the network, and while it runs says so and refuses another run", async () => {
    await mount();
    const run = container.querySelector<HTMLButtonElement>("button");
    expect(run?.textContent).toBe("Run E-I network");
    await act(async () => { run?.click(); await Promise.resolve(); });
    expect(store.runNetwork).toHaveBeenCalledTimes(1);
    store.isSimulating = true;
    await mount();
    const running = container.querySelector<HTMLButtonElement>("button");
    expect(running?.textContent).toBe("Running…");
    expect(running?.disabled).toBe(true);
  });
});
