// @vitest-environment happy-dom
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { at } from "../arrayAt";
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
    await act(async () => { root.unmount(); });
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
    await act(async () => { root.unmount(); });
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

    await act(async () => { useStudioStore.getState().selectProjection("e1"); });

    expect(container.querySelector("section")?.getAttribute("aria-label")).toBe(
      "Projection Input → Output",
    );
    expect(container.querySelector<HTMLInputElement>("#projection-weight")?.value).toBe("0.5");

    await act(async () => { useStudioStore.getState().selectProjection(null); });

    expect(container.querySelector("#projection-weight")).toBeNull();
    await act(async () => { root.unmount(); });
    container.remove();
  });

  it("edits the selected projection through the store", async () => {
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);

    await act(async () => {
      root.render(<NetworkCanvas />);
    });
    await act(async () => { useStudioStore.getState().selectProjection("e1"); });
    const delay = container.querySelector<HTMLInputElement>("#projection-delay");
    if (delay === null) throw new Error("the projection editor drew no delay input");
    // A React-controlled input ignores `element.value = x`; the change has to
    // go through the prototype's own setter before the event is dispatched.
    const descriptor = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype,
      "value",
    );
    // eslint-disable-next-line @typescript-eslint/unbound-method -- `.call` supplies the receiver
    const setter = descriptor?.set;
    if (setter === undefined) throw new Error("HTMLInputElement has no value setter");

    await act(async () => {
      setter.call(delay, "2");
      delay.dispatchEvent(new Event("input", { bubbles: true }));
    });

    expect(at(useStudioStore.getState().graphProjections, 0).delay).toBe(2);
    await act(async () => { root.unmount(); });
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

/**
 * The two editors share one panel position, so selecting either must close the
 * other, and a population's model contract has to be asked for by the canvas.
 */
describe("NetworkCanvas population editor", () => {
  const pristine = useStudioStore.getState();

  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: unknown) => {
        const url = String(input);
        if (url.includes("/graph/models/")) {
          return new Response(
            JSON.stringify({
              drive: { kind: "float", parameter: "current", positional_only: false },
              model: "LIF",
              parameters: [{ default: 1.1, kind: "float", name: "capacitance" }],
              schema_version: "studio.population-model-contract.v1",
              unsupported: [],
            }),
            { headers: { "content-type": "application/json" } },
          );
        }
        return new Response("[]", { headers: { "content-type": "application/json" } });
      }),
    );
    useStudioStore.setState({ graphPopulations: POPULATIONS, graphProjections: PROJECTIONS });
  });

  afterEach(() => {
    useStudioStore.setState(pristine, true);
    vi.unstubAllGlobals();
  });

  it("opens the editor on the selected population and fetches its contract", async () => {
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);

    await act(async () => {
      root.render(<NetworkCanvas />);
    });
    await act(async () => { useStudioStore.getState().selectPopulation("p1"); });

    expect(container.querySelector("section")?.getAttribute("aria-label")).toBe(
      "Population Input",
    );
    expect(container.querySelector<HTMLInputElement>("#population-count")?.value).toBe("100");
    expect(useStudioStore.getState().populationModelContract?.model).toBe("LIF");
    await act(async () => { root.unmount(); });
    container.remove();
  });

  it("closes the projection editor when a population is selected, and the reverse", () => {
    useStudioStore.getState().selectProjection("e1");
    expect(useStudioStore.getState().selectedPopulationId).toBeNull();

    useStudioStore.getState().selectPopulation("p1");

    expect(useStudioStore.getState().selectedProjectionId).toBeNull();
    expect(useStudioStore.getState().selectedPopulationId).toBe("p1");

    useStudioStore.getState().selectProjection("e1");

    expect(useStudioStore.getState().selectedPopulationId).toBeNull();
  });

  it("stops editing a population that was removed", () => {
    useStudioStore.getState().selectPopulation("p1");

    useStudioStore.getState().removePopulation("p1");

    expect(useStudioStore.getState().selectedPopulationId).toBeNull();
  });

  it("keeps editing when a different population is removed", () => {
    useStudioStore.getState().selectPopulation("p1");

    useStudioStore.getState().removePopulation("p2");

    expect(useStudioStore.getState().selectedPopulationId).toBe("p1");
  });
});

/**
 * The duplicate is the first canvas operation that reaches the server several
 * times for one user action, and the first that has to tell the user about
 * something the diagram does not show.
 */
describe("NetworkCanvas duplicate", () => {
  const pristine = useStudioStore.getState();

  beforeEach(() => {
    let minted = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: unknown, init?: { body?: string }) => {
        const url = String(input);
        const body = JSON.parse(init?.body ?? "{}") as Record<string, unknown>;
        minted += 1;
        if (url.includes("/graph/population")) {
          return new Response(
            JSON.stringify({
              count: body.count,
              drive: body.drive,
              id: `pop_new${minted}`,
              label: body.label,
              model: body.model,
              neuron_type: body.neuron_type,
              params: body.params ?? {},
              position: { x: body.x, y: body.y },
              type: "population",
            }),
            { headers: { "content-type": "application/json" } },
          );
        }
        if (url.includes("/graph/projection")) {
          return new Response(
            JSON.stringify({
              delay: body.delay,
              id: `proj_new${minted}`,
              probability: body.probability,
              rule: body.rule,
              source: body.source_id,
              target: body.target_id,
              weight: body.weight,
            }),
            { headers: { "content-type": "application/json" } },
          );
        }
        return new Response("[]", { headers: { "content-type": "application/json" } });
      }),
    );
    useStudioStore.setState({ graphPopulations: POPULATIONS, graphProjections: PROJECTIONS });
  });

  afterEach(() => {
    useStudioStore.setState(pristine, true);
    vi.unstubAllGlobals();
  });

  it("copies the selected populations and the projection between them", async () => {
    useStudioStore.getState().selectPopulations(["p1", "p2"]);

    await useStudioStore.getState().duplicateSelection();

    const state = useStudioStore.getState();
    expect(state.graphPopulations).toHaveLength(4);
    expect(state.graphProjections).toHaveLength(2);
    // The copy joins the originals; it does not replace them.
    expect(state.graphPopulations.map((one) => one.label)).toEqual([
      "Input",
      "Output",
      "Input copy",
      "Output copy",
    ]);
  });

  it("wires the copied projection between the copies, not the originals", async () => {
    useStudioStore.getState().selectPopulations(["p1", "p2"]);

    await useStudioStore.getState().duplicateSelection();

    const state = useStudioStore.getState();
    const copies = new Set(state.graphPopulations.slice(2).map((one) => one.id));
    const copied = at(state.graphProjections, 1);
    expect(copies.has(copied.source)).toBe(true);
    expect(copies.has(copied.target)).toBe(true);
  });

  it("leaves a projection that crosses the selection, and says so", async () => {
    useStudioStore.getState().selectPopulations(["p1"]);

    await useStudioStore.getState().duplicateSelection();

    const state = useStudioStore.getState();
    expect(state.graphProjections).toHaveLength(1);
    expect(state.graphNotice).toContain("1 projection left the selection");
  });

  it("selects the copies, so a second duplicate copies the copy", async () => {
    useStudioStore.getState().selectPopulations(["p1"]);

    await useStudioStore.getState().duplicateSelection();

    expect(useStudioStore.getState().selectedPopulationIds).toEqual(["pop_new1"]);
  });

  it("is one undo, not one per created object", async () => {
    useStudioStore.getState().selectPopulations(["p1", "p2"]);

    await useStudioStore.getState().duplicateSelection();
    useStudioStore.getState().undoGraphEdit();

    const state = useStudioStore.getState();
    expect(state.graphPopulations).toHaveLength(2);
    expect(state.graphProjections).toHaveLength(1);
  });

  it("does nothing at all when the selection is empty", async () => {
    useStudioStore.getState().selectPopulations([]);

    await useStudioStore.getState().duplicateSelection();

    expect(useStudioStore.getState().graphPopulations).toHaveLength(2);
    expect(fetch).not.toHaveBeenCalled();
  });

  it("refuses loudly if a copied projection loses an endpoint", async () => {
    // The plan cannot produce this; if it ever did, dropping the projection
    // would return a graph quietly smaller than the one asked for.
    const { studioDuplicatePlan } = await import("../studioGraphDuplicate");
    const spy = vi.spyOn(await import("../studioGraphDuplicate"), "studioDuplicatePlan");
    spy.mockImplementation((populations, projections, ids) => ({
      ...studioDuplicatePlan(populations, projections, ids),
      projections: [
        {
          autapses: undefined,
          delay: 0,
          probability: 0.1,
          rule: "random" as const,
          seed: undefined,
          sourceId: "p1",
          targetId: "not-copied",
          weight: 0.5,
        },
      ],
    }));
    useStudioStore.getState().selectPopulations(["p1"]);

    await useStudioStore.getState().duplicateSelection();

    expect(useStudioStore.getState().error).toContain("endpoint was not copied");
    spy.mockRestore();
  });

  it("offers the duplicate control only when something is selected", async () => {
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => {
      root.render(<NetworkCanvas />);
    });
    const duplicate = container.querySelector<HTMLButtonElement>(
      'button[aria-label^="Duplicate the"]',
    );

    expect(duplicate?.disabled).toBe(true);

    await act(async () => { useStudioStore.getState().selectPopulations(["p1", "p2"]); });

    expect(
      container.querySelector<HTMLButtonElement>('button[aria-label^="Duplicate the"]')?.disabled,
    ).toBe(false);
    expect(
      container.querySelector('button[aria-label^="Duplicate the"]')?.getAttribute("aria-label"),
    ).toBe("Duplicate the 2 selected populations and the projections between them");
    await act(async () => { root.unmount(); });
    container.remove();
  });
});
