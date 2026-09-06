// @vitest-environment happy-dom
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { act } from "react";
import { createRoot } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { PopulationNode, ProjectionEdge } from "../api/client";
import type { PipelineEvidenceModel } from "../pipelineEvidence";
import { useStudioStore } from "../stores/studio";
import NetworkCanvas, { PipelineEvidenceStrip } from "./NetworkCanvas";

const evidence: PipelineEvidenceModel = {
  actionKind: "studio.pipeline.run",
  classification: "compile",
  evidenceArtifact: "pipeline/evidence.json",
  pipeline: "graph -> simulate -> compile -> synthesise",
  replayRoute: "POST /api/pipeline/run",
  resultArtifact: "pipeline/result.json",
  status: "completed",
  step: "complete",
  target: "ICE40",
};

describe("NetworkCanvas", () => {
  it("renders path-free pipeline action evidence metadata", () => {
    const html = renderToStaticMarkup(<PipelineEvidenceStrip evidence={evidence} />);

    expect(html).toContain("class");
    expect(html).toContain("compile");
    expect(html).toContain("studio.pipeline.run");
    expect(html).toContain("completed");
    expect(html).toContain("ICE40");
    expect(html).toContain("POST /api/pipeline/run");
    expect(html).toContain("pipeline/result.json / pipeline/evidence.json");
  });
});

const POPULATIONS: PopulationNode[] = [
  {
    count: 100,
    drive: { current: 1.2, kind: "constant" },
    id: "p1",
    label: "Input",
    model: "LIF",
    neuron_type: "excitatory",
    params: {},
    position: { x: 0, y: 0 },
    type: "population",
  },
  {
    count: 20,
    id: "p2",
    label: "Output",
    model: "LIF",
    neuron_type: "inhibitory",
    params: {},
    position: { x: 200, y: 0 },
    type: "population",
  },
];

const PROJECTIONS: ProjectionEdge[] = [
  {
    delay: 0,
    id: "e1",
    probability: 0.1,
    rule: "random",
    source: "p1",
    target: "p2",
    weight: 0.5,
  },
];

/**
 * The table view is the canvas's keyboard and screen-reader equivalent, so the
 * wiring — not only the table component — has to be exercised: the toggle has
 * to reach the same graph the canvas draws, and the canvas has to leave the tab
 * order while the table stands in for it.
 */
describe("NetworkCanvas table view", () => {
  const pristine = useStudioStore.getState();

  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("[]", { headers: { "content-type": "application/json" } })),
    );
    useStudioStore.setState({ graphPopulations: POPULATIONS, graphProjections: PROJECTIONS });
  });

  afterEach(() => {
    useStudioStore.setState(pristine, true);
    vi.unstubAllGlobals();
  });

  it("shows the graph as a table when the toggle is pressed, and hides it again", async () => {
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);

    await act(async () => {
      root.render(<NetworkCanvas />);
    });
    const toggle = container.querySelector<HTMLButtonElement>("button[aria-pressed]");

    expect(toggle?.textContent).toBe("Table view");
    expect(toggle?.getAttribute("aria-pressed")).toBe("false");
    expect(container.querySelector("table")).toBeNull();

    await act(async () => toggle?.click());

    expect(toggle?.getAttribute("aria-pressed")).toBe("true");
    expect(container.querySelector("caption")?.textContent).toBe(
      "Network topology: 2 populations holding 120 neurons, connected by 1 projection.",
    );
    expect([...container.querySelectorAll('th[scope="row"] span:first-child')].map(
      (cell) => cell.textContent,
    )).toEqual(["Input", "Output"]);
    // The canvas sits inside a hidden container, so it leaves the tab order
    // with it rather than staying reachable behind the table.
    const hidden = [...container.querySelectorAll("div")].filter(
      (element) => element.style.display === "none",
    );
    expect(
      hidden.some((element) => element.querySelector('div[style*="position: relative"]') !== null),
    ).toBe(true);

    await act(async () => toggle?.click());

    expect(toggle?.getAttribute("aria-pressed")).toBe("false");
    expect(container.querySelector("table")).toBeNull();
    await act(async () => root.unmount());
    container.remove();
  });

  it("deletes through the table the same way the canvas does", async () => {
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);

    await act(async () => {
      root.render(<NetworkCanvas />);
    });
    await act(async () => container.querySelector<HTMLButtonElement>("button[aria-pressed]")?.click());
    const remove = container.querySelector<HTMLButtonElement>(
      'button[aria-label="Delete population Input and its 1 projection"]',
    );

    await act(async () => remove?.click());

    expect(useStudioStore.getState().graphPopulations.map((population) => population.id)).toEqual(["p2"]);
    expect(useStudioStore.getState().graphProjections).toEqual([]);
    expect(container.querySelector("caption")?.textContent).toBe(
      "Network topology: 1 population holding 20 neurons, connected by 0 projections.",
    );
    await act(async () => root.unmount());
    container.remove();
  });
});

/**
 * The property editor is only useful if the canvas opens it on the projection
 * the user picked and closes it when that projection is gone.
 */
describe("NetworkCanvas projection editor", () => {
  const pristine = useStudioStore.getState();

  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("[]", { headers: { "content-type": "application/json" } })),
    );
    useStudioStore.setState({ graphPopulations: POPULATIONS, graphProjections: PROJECTIONS });
  });

  afterEach(() => {
    useStudioStore.setState(pristine, true);
    vi.unstubAllGlobals();
  });

  it("opens the editor on the selected projection and closes it on none", async () => {
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);

    await act(async () => {
      root.render(<NetworkCanvas />);
    });
    expect(container.querySelector("#projection-weight")).toBeNull();

    await act(async () => useStudioStore.getState().selectProjection("e1"));

    expect(container.querySelector("section")?.getAttribute("aria-label")).toBe(
      "Projection Input → Output",
    );
    expect(container.querySelector<HTMLInputElement>("#projection-weight")?.value).toBe("0.5");

    await act(async () => useStudioStore.getState().selectProjection(null));

    expect(container.querySelector("#projection-weight")).toBeNull();
    await act(async () => root.unmount());
    container.remove();
  });

  it("edits the selected projection through the store", async () => {
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);

    await act(async () => {
      root.render(<NetworkCanvas />);
    });
    await act(async () => useStudioStore.getState().selectProjection("e1"));
    const delay = container.querySelector<HTMLInputElement>("#projection-delay");
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype,
      "value",
    )?.set;

    await act(async () => {
      setter?.call(delay!, "2");
      delay!.dispatchEvent(new Event("input", { bubbles: true }));
    });

    expect(useStudioStore.getState().graphProjections[0].delay).toBe(2);
    await act(async () => root.unmount());
    container.remove();
  });

  it("stops editing a projection that left with its population", () => {
    useStudioStore.getState().selectProjection("e1");

    useStudioStore.getState().removePopulation("p1");

    expect(useStudioStore.getState().graphProjections).toEqual([]);
    expect(useStudioStore.getState().selectedProjectionId).toBeNull();
  });

  it("stops editing a projection that was removed on its own", () => {
    useStudioStore.getState().selectProjection("e1");

    useStudioStore.getState().removeProjection("e1");

    expect(useStudioStore.getState().selectedProjectionId).toBeNull();
  });

  it("keeps editing when a different projection is removed", () => {
    useStudioStore.setState({
      graphProjections: [
        ...PROJECTIONS,
        {
          delay: 0,
          id: "e2",
          probability: 0.1,
          rule: "random",
          source: "p2",
          target: "p1",
          weight: -0.5,
        },
      ],
    });
    useStudioStore.getState().selectProjection("e1");

    useStudioStore.getState().removeProjection("e2");

    expect(useStudioStore.getState().selectedProjectionId).toBe("e1");
  });
});
