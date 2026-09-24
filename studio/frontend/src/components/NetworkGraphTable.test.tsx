// @vitest-environment happy-dom
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Table markup a screen reader can navigate

/**
 * The markup, not only the model, has to carry the topology.
 *
 * A grid of divs holding the right strings is still unusable: without a
 * caption, column scopes and a row header, a screen reader reads cells with no
 * idea which population they belong to. These cases assert the semantics
 * rather than the styling, because the semantics are what a reader navigates.
 */

import { at } from "../arrayAt";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import type { PopulationNode, ProjectionEdge } from "../api/client";
import NetworkGraphTable from "./NetworkGraphTable";

const populations: PopulationNode[] = [
  {
    count: 100,
    drive: { current: 1.2, kind: "constant" },
    id: "input",
    label: "Input",
    model: "LIF",
    neuron_type: "excitatory",
    params: {},
    position: { x: 0, y: 0 },
    type: "population",
  },
  {
    count: 20,
    id: "output",
    label: "Output",
    model: "AdEx",
    neuron_type: "inhibitory",
    params: {},
    position: { x: 200, y: 0 },
    type: "population",
  },
];

const projections: ProjectionEdge[] = [
  {
    delay: 0,
    id: "p1",
    probability: 0.1,
    rule: "random",
    source: "input",
    target: "output",
    weight: 0.5,
  },
];

/** Handlers for the controls a case does not exercise. */
const unused = {
  onEditPopulation: () => undefined,
  onEditProjection: () => undefined,
  onRemoveProjection: () => undefined,
};

/** Render the table to static markup, so the assertions are on semantics. */
function render(
  nodes: PopulationNode[] = populations,
  edges: ProjectionEdge[] = projections,
): string {
  return renderToStaticMarkup(
    <NetworkGraphTable
      populations={nodes}
      projections={edges}
      onRemovePopulation={() => undefined}
      {...unused}
    />,
  );
}

describe("NetworkGraphTable", () => {
  it("captions the table with the size of the topology", () => {
    expect(render()).toContain(
      "<caption" +
        ' style="caption-side:top;font-size:11px;padding:4px 0;text-align:left">' +
        "Network topology: 2 populations holding 120 neurons, connected by 1 projection.",
    );
  });

  it("scopes the column headers so a reader knows what a cell holds", () => {
    const html = render();

    for (const column of ["Population", "Model", "Neurons", "Type", "Input", "Incoming", "Outgoing", "Actions"]) {
      expect(html).toContain(`scope="col"`);
      expect(html).toContain(`>${column}</th>`);
    }
  });

  it("makes the population the row header and attaches its description", () => {
    const html = render();

    expect(html).toContain(`scope="row"`);
    expect(html).toContain("<span>Input</span>");
    expect(html).toContain(
      "Input: 100 excitatory LIF neurons, I = 1.2; no incoming projections; " +
        "1 outgoing to Output (w=0.5 p=0.1).",
    );
  });

  it("gives every delete control a name that says what leaves with the population", () => {
    const html = render();

    expect(html).toContain('aria-label="Delete population Input and its 1 projection"');
    expect(html).toContain('aria-label="Delete population Output and its 1 projection"');
  });

  it("names an unconnected population's delete control without a projection count", () => {
    const html = render([at(populations, 1)], []);

    expect(html).toContain('aria-label="Delete population Output"');
  });

  it("writes the population's own fields into the row", () => {
    const html = render();

    expect(html).toContain(">AdEx</td>");
    expect(html).toContain(">20</td>");
    expect(html).toContain(">inhibitory</td>");
    expect(html).toContain(">I = 1.2</td>");
  });

  it("says none rather than leaving a connection cell empty", () => {
    const html = render();

    // Two empty connection cells and two clean Problems cells.
    expect(html.match(/>none</g)).toHaveLength(4);
  });

  it("lists each connection with the population at the other end", () => {
    const html = render();

    expect(html).toContain("Output <span");
    expect(html).toContain(">w=0.5 p=0.1</span>");
  });

  it("renders a caption and no rows for an empty graph", () => {
    const html = render([], []);

    expect(html).toContain("Network topology: no populations yet.");
    expect(html).toContain("<tbody></tbody>");
  });

  it("shows what validation refused about a population on that row", () => {
    const html = renderToStaticMarkup(
      <NetworkGraphTable
        populations={populations}
        projections={projections}
        issues={[
          {
            attribute: "count",
            field: "populations[1].count",
            id: "output",
            kind: "population",
            message: "Population Output count must be a positive integer",
            subject: "Output",
          },
          {
            attribute: "weight",
            field: "projections[0].weight",
            id: "p1",
            kind: "projection",
            message: "Projection p1 weight conflicts with its excitatory source",
            subject: "Input → Output",
          },
        ]}
        onRemovePopulation={() => undefined}
        {...unused}
      />,
    );

    expect(html).toContain("Population Output count must be a positive integer");
    expect(html).toContain("Projection p1 weight conflicts with its excitatory source");
    // The row's own sentence says it too, for a reader who does not walk cells.
    expect(html).toContain(
      "Validation refused it: Population Output count must be a positive integer",
    );
  });

  it("acts on the object each control names, and on nothing else", async () => {
    const container = document.createElement("div");
    const root = createRoot(container);
    const handlers = {
      onEditPopulation: vi.fn(),
      onEditProjection: vi.fn(),
      onRemovePopulation: vi.fn(),
      onRemoveProjection: vi.fn(),
    };

    await act(async () => {
      root.render(
        <NetworkGraphTable populations={populations} projections={projections} {...handlers} />,
      );
    });
    const named = (name: string): HTMLButtonElement => {
      const found = container.querySelector<HTMLButtonElement>(`button[aria-label="${name}"]`);
      if (found === null) throw new Error(`no control named ${name}`);
      return found;
    };
    named("Delete population Output and its 1 projection").click();
    named("Edit population Input").click();
    named("Edit projection Input to Output (w=0.5 p=0.1)").click();
    named("Delete projection Input to Output (w=0.5 p=0.1)").click();

    expect(handlers.onRemovePopulation).toHaveBeenCalledExactlyOnceWith("output");
    expect(handlers.onEditPopulation).toHaveBeenCalledExactlyOnceWith("input");
    expect(handlers.onEditProjection).toHaveBeenCalledExactlyOnceWith("p1");
    expect(handlers.onRemoveProjection).toHaveBeenCalledExactlyOnceWith("p1");
    await act(async () => { root.unmount(); });
  });

  it("gives a projection its controls once, at its source, not again at its target", async () => {
    const container = document.createElement("div");
    const root = createRoot(container);

    await act(async () => {
      root.render(
        <NetworkGraphTable
          populations={populations}
          projections={projections}
          onRemovePopulation={() => undefined}
          {...unused}
        />,
      );
    });
    const labels = [...container.querySelectorAll<HTMLButtonElement>("tbody button")].map(
      (button) => button.getAttribute("aria-label"),
    );

    expect(labels.filter((label) => label?.includes("projection Input to Output"))).toHaveLength(2);
    await act(async () => { root.unmount(); });
  });

  it("reaches every control by keyboard, because each one is a real button", async () => {
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);

    await act(async () => {
      root.render(
        <NetworkGraphTable
          populations={populations}
          projections={projections}
          onRemovePopulation={() => undefined}
          {...unused}
        />,
      );
    });
    const controls = [...container.querySelectorAll<HTMLButtonElement>("tbody button")];

    // Edit and delete per population, edit and delete for the one projection.
    expect(controls).toHaveLength(6);
    for (const control of controls) {
      expect(control.tabIndex).toBe(0);
      control.focus();
      expect(document.activeElement).toBe(control);
    }
    await act(async () => { root.unmount(); });
    container.remove();
  });
});
