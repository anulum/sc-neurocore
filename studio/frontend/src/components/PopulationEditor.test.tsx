// @vitest-environment happy-dom
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — The population property editor as a reader meets it

/**
 * The parameters section is the one that can lie.
 *
 * Identity and drive fields are fixed by the graph specification, but which
 * constructor fields a population may override belongs to the model contract,
 * which arrives from the server. Offering a guess while it is in flight, or
 * keeping the previous model's parameters after the model changes, would both
 * present inputs the graph refuses. These cases drive the real DOM and hold
 * the editor to waiting, to saying why a field is not editable, and to the
 * same aria wiring the projection editor has.
 */

import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";

import type { PopulationModelContract, PopulationNode } from "../api/client";
import type { StudioGraphIssueLocation } from "../studioGraphValidation";
import PopulationEditor from "./PopulationEditor";

const MODELS = ["SCLapicqueLIFNeuron", "AdExNeuron"];

const CONTRACT: PopulationModelContract = {
  drive: { kind: "float", parameter: "current", positional_only: false },
  model: "SCLapicqueLIFNeuron",
  parameters: [
    { default: 1.1, kind: "float", name: "capacitance" },
    { default: 10, kind: "float", name: "tau" },
  ],
  schema_version: "studio.population-model-contract.v1",
  unsupported: [{ name: "profile", reason: "non-numeric field" }],
};

/** One valid population, overridden where a case needs a particular value. */
function population(overrides: Partial<PopulationNode> = {}): PopulationNode {
  return {
    count: 80,
    drive: { kind: "none" },
    id: "p1",
    label: "Exc 0",
    model: "SCLapicqueLIFNeuron",
    neuron_type: "excitatory",
    params: {},
    position: { x: 0, y: 0 },
    type: "population",
    ...overrides,
  };
}

interface Mounted {
  container: HTMLDivElement;
  onChange: ReturnType<typeof vi.fn>;
  onValidate: ReturnType<typeof vi.fn>;
  unmount: () => Promise<void>;
}

/** Mount the editor on a real DOM and return the handles a case needs. */
async function mount(
  node: PopulationNode = population(),
  contract: PopulationModelContract | null = CONTRACT,
  issues: StudioGraphIssueLocation[] = [],
): Promise<Mounted> {
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  const onChange = vi.fn();
  const onValidate = vi.fn();
  await act(async () => {
    root.render(
      <PopulationEditor
        population={node}
        models={MODELS}
        contract={contract}
        issues={issues}
        onChange={onChange}
        onValidate={onValidate}
      />,
    );
  });
  return {
    container,
    onChange,
    onValidate,
    unmount: async () => {
      await act(async () => { root.unmount(); });
      container.remove();
    },
  };
}

/** Return one input by the id its label points at, or fail saying which. */
function field(container: HTMLElement, id: string): HTMLInputElement {
  const found = container.querySelector<HTMLInputElement>(`#population-${id}`);
  if (found === null) throw new Error(`no input for ${id}`);
  return found;
}

/**
 * Type into a controlled input the way a browser does.
 *
 * React reads the value through the prototype setter it patched, so assigning
 * `element.value` directly is invisible to it; going through the original
 * descriptor is what a real keystroke does.
 */
async function type(element: HTMLInputElement, text: string): Promise<void> {
  // The receiver is supplied by `.call` below, which is the whole point of
  // going through the prototype descriptor.

  // eslint-disable-next-line @typescript-eslint/unbound-method -- `.call` supplies the receiver
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype,
    "value",
  )?.set;
  await act(async () => {
    setter?.call(element, text);
    element.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

describe("what the editor offers", () => {
  it("names the population it is editing", async () => {
    const mounted = await mount();

    expect(mounted.container.querySelector("section")?.getAttribute("aria-label")).toBe(
      "Population Exc 0",
    );
    await mounted.unmount();
  });

  it("labels the identity, drive and parameter inputs", async () => {
    const mounted = await mount();
    const labels = [...mounted.container.querySelectorAll("label")].map(
      (label) => label.getAttribute("for"),
    );

    expect(labels).toEqual([
      "population-label",
      "population-model",
      "population-count",
      "population-neuron_type",
      "population-drive-kind",
      "population-params-capacitance",
      "population-params-tau",
    ]);
    await mounted.unmount();
  });

  it("ties each contract statement to its own input", async () => {
    const mounted = await mount();
    const count = field(mounted.container, "count");

    expect(count.getAttribute("aria-describedby")).toBe("population-count-help");
    expect(
      mounted.container.querySelector("#population-count-help")?.textContent,
    ).toContain("positive whole number");
    await mounted.unmount();
  });

  it("shows a parameter's declared default when nothing overrides it", async () => {
    const mounted = await mount();

    expect(field(mounted.container, "params-capacitance").value).toBe("1.1");
    await mounted.unmount();
  });

  it("lists the fields that are not inputs, with the reason each is not", async () => {
    const mounted = await mount();

    expect(mounted.container.textContent).toContain("profile: non-numeric field");
    await mounted.unmount();
  });
});

describe("while the model contract has not arrived", () => {
  it("offers no parameters and says it is waiting", async () => {
    const mounted = await mount(population(), null);

    expect(mounted.container.querySelector("#population-params-capacitance")).toBeNull();
    expect(mounted.container.textContent).toContain("Waiting for the model contract");
    await mounted.unmount();
  });

  it("still offers the identity and drive fields", async () => {
    const mounted = await mount(population(), null);

    expect(field(mounted.container, "count").value).toBe("80");
    expect(mounted.container.querySelector("#population-drive-kind")).not.toBeNull();
    await mounted.unmount();
  });

  it("offers no parameters when the contract is for another model", async () => {
    // Keeping them would present inputs the graph refuses for this model.
    const mounted = await mount(population({ model: "AdExNeuron" }), CONTRACT);

    expect(mounted.container.querySelector("#population-params-capacitance")).toBeNull();
    expect(mounted.container.textContent).not.toContain("profile: non-numeric field");
    await mounted.unmount();
  });
});

describe("the external input", () => {
  it("offers only the kind when nothing drives the population", async () => {
    const mounted = await mount();

    expect(mounted.container.querySelector("#population-drive-current")).toBeNull();
    await mounted.unmount();
  });

  it("offers the current once the drive is constant", async () => {
    const mounted = await mount(population({ drive: { current: 1.2, kind: "constant" } }));

    expect(field(mounted.container, "drive-current").value).toBe("1.2");
    await mounted.unmount();
  });

  it("changes one drive field without losing the rest of the drive", async () => {
    const mounted = await mount(
      population({ drive: { kind: "poisson", rate_hz: 20, weight: 0.5 } }),
    );

    await type(field(mounted.container, "drive-rate_hz"), "30");

    expect(mounted.onChange).toHaveBeenCalledWith("p1", {
      drive: { kind: "poisson", rate_hz: 30, weight: 0.5 },
    });
    await mounted.unmount();
  });
});

describe("editing a field", () => {
  it("sends the parsed value and asks the server", async () => {
    const mounted = await mount();

    await type(field(mounted.container, "count"), "40");

    expect(mounted.onChange).toHaveBeenCalledWith("p1", { count: 40 });
    expect(mounted.onValidate).toHaveBeenCalledOnce();
    await mounted.unmount();
  });

  it("writes a parameter override rather than the whole params object", async () => {
    const mounted = await mount(population({ params: { tau: 10 } }));

    await type(field(mounted.container, "params-capacitance"), "2.5");

    expect(mounted.onChange).toHaveBeenCalledWith("p1", {
      params: { capacitance: 2.5, tau: 10 },
    });
    await mounted.unmount();
  });

  it("says which field is not a number, beside that field", async () => {
    const mounted = await mount();
    const count = field(mounted.container, "count");

    await type(count, "many");

    expect(count.getAttribute("aria-invalid")).toBe("true");
    expect(
      mounted.container.querySelector("#population-count-error")?.textContent,
    ).toContain("count must be a number");
    await mounted.unmount();
  });

  it("refuses an empty label rather than an unnameable population", async () => {
    const mounted = await mount();

    await type(field(mounted.container, "label"), "  ");

    expect(
      mounted.container.querySelector("#population-label-error")?.textContent,
    ).toContain("must not be empty");
    expect(mounted.onChange).not.toHaveBeenCalled();
    await mounted.unmount();
  });

  it("clears the parameters when the model changes", async () => {
    const mounted = await mount(population({ params: { tau: 12 } }));
    const model = mounted.container.querySelector<HTMLSelectElement>("#population-model");
    if (model === null) throw new Error("no model select");

    await act(async () => {
      model.value = "AdExNeuron";
      model.dispatchEvent(new Event("change", { bubbles: true }));
    });

    expect(mounted.onChange).toHaveBeenCalledWith("p1", {
      model: "AdExNeuron",
      params: {},
    });
    await mounted.unmount();
  });
});

describe("what the server said about a field", () => {
  const issues: StudioGraphIssueLocation[] = [
    {
      attribute: "count",
      field: "populations[0].count",
      id: "p1",
      kind: "population",
      message: "Population Exc 0 count must be a positive integer",
      subject: "Exc 0",
    },
    {
      attribute: "params.tau",
      field: "populations[0].params.tau",
      id: "p1",
      kind: "population",
      message: "Population Exc 0 params.tau: unknown parameter",
      subject: "Exc 0",
    },
    {
      attribute: "count",
      field: "populations[1].count",
      id: "p2",
      kind: "population",
      message: "Population Inh 0 count must be a positive integer",
      subject: "Inh 0",
    },
  ];

  it("puts an identity failure on its own input", async () => {
    const mounted = await mount(population({ count: 0 }), CONTRACT, issues);

    expect(field(mounted.container, "count").getAttribute("aria-invalid")).toBe("true");
    expect(
      mounted.container.querySelector("#population-count-error")?.textContent,
    ).toContain("must be a positive integer");
    await mounted.unmount();
  });

  it("puts a parameter failure on that parameter's input", async () => {
    const mounted = await mount(population(), CONTRACT, issues);

    expect(
      mounted.container.querySelector("#population-params-tau-error")?.textContent,
    ).toContain("unknown parameter");
    await mounted.unmount();
  });

  it("leaves another population's failure out of this editor", async () => {
    const mounted = await mount(population(), CONTRACT, issues);

    expect(mounted.container.textContent).not.toContain("Inh 0");
    await mounted.unmount();
  });
});
