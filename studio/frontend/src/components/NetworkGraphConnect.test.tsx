// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

// @vitest-environment happy-dom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import type { PopulationNode } from "../api/client";
import NetworkGraphConnect from "./NetworkGraphConnect";

/** A minimal population, distinguishable by label in the rendered options. */
function population(id: string, label: string): PopulationNode {
  return {
    id,
    type: "population",
    label,
    model: "LIFNeuron",
    count: 10,
    neuron_type: "excitatory",
    position: { x: 0, y: 0 },
    params: {},
  };
}

describe("NetworkGraphConnect", () => {
  it("offers every population as both source and target", () => {
    const markup = renderToStaticMarkup(
      <NetworkGraphConnect
        populations={[population("p1", "Excitatory"), population("p2", "Inhibitory")]}
        onConnect={() => undefined}
      />,
    );

    expect(markup).toContain('aria-label="Projection source population"');
    expect(markup).toContain('aria-label="Projection target population"');
    expect(markup.match(/Excitatory/g)).toHaveLength(2);
    expect(markup.match(/Inhibitory/g)).toHaveLength(2);
  });

  it("refuses to submit before both ends are chosen", () => {
    const markup = renderToStaticMarkup(
      <NetworkGraphConnect
        populations={[population("p1", "A"), population("p2", "B")]}
        onConnect={() => undefined}
      />,
    );

    expect(markup).toContain("disabled");
  });

  it("says what is missing when there is nothing to connect", () => {
    const markup = renderToStaticMarkup(
      <NetworkGraphConnect populations={[population("p1", "Only")]} onConnect={() => undefined} />,
    );

    expect(markup).toContain("Add a second population to connect one.");
    expect(markup).not.toContain("<select");
  });

  it("names the control for a screen reader rather than relying on its glyph", () => {
    const markup = renderToStaticMarkup(
      <NetworkGraphConnect
        populations={[population("p1", "A"), population("p2", "B")]}
        onConnect={() => undefined}
      />,
    );

    expect(markup).toContain(
      'aria-label="Connect the chosen source population to the chosen target population"',
    );
  });
  it("hands the chosen identities to the same action the canvas drag calls", async () => {
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    const onConnect = vi.fn();

    await act(async () => {
      root.render(
        <NetworkGraphConnect
          populations={[population("p1", "Excitatory"), population("p2", "Inhibitory")]}
          onConnect={onConnect}
        />,
      );
    });

    const [source, target] = Array.from(container.querySelectorAll("select"));
    const button = container.querySelector<HTMLButtonElement>(
      '[data-testid="graph-connect-submit"]',
    );
    if (!source || !target || !button) throw new Error("connect controls did not render");

    expect(button.disabled).toBe(true);

    await act(async () => {
      source.value = "p1";
      source.dispatchEvent(new Event("change", { bubbles: true }));
      target.value = "p2";
      target.dispatchEvent(new Event("change", { bubbles: true }));
    });

    expect(button.disabled).toBe(false);

    await act(async () => { button.click(); });

    expect(onConnect).toHaveBeenCalledWith("p1", "p2");

    await act(async () => { root.unmount(); });
    container.remove();
  });
});
