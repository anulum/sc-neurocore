// @vitest-environment happy-dom
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — The projection property editor as a reader meets it

/**
 * An input without its contract beside it is a guess.
 *
 * These cases drive the real DOM, because what is being asserted is what a
 * user and a screen reader receive: that each input is labelled, that the
 * contract and any failure are tied to it with `aria-describedby` rather than
 * left elsewhere on the page, that half-typed text is not rewritten under the
 * hands typing it, and that an edit reaches the graph and asks the server.
 */

import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";

import type { PopulationNode, ProjectionEdge } from "../api/client";
import type { StudioGraphIssueLocation } from "../studioGraphValidation";
import ProjectionEditor from "./ProjectionEditor";

const POPULATIONS: PopulationNode[] = [
  {
    count: 80,
    id: "p1",
    label: "Exc 0",
    model: "SCLapicqueLIFNeuron",
    neuron_type: "excitatory",
    params: {},
    position: { x: 0, y: 0 },
    type: "population",
  },
  {
    count: 20,
    id: "p2",
    label: "Inh 0",
    model: "SCLapicqueLIFNeuron",
    neuron_type: "inhibitory",
    params: {},
    position: { x: 200, y: 0 },
    type: "population",
  },
];

function projection(overrides: Partial<ProjectionEdge> = {}): ProjectionEdge {
  return {
    delay: 0,
    id: "e1",
    probability: 0.2,
    rule: "random",
    source: "p1",
    target: "p2",
    weight: 40,
    ...overrides,
  };
}

interface Mounted {
  container: HTMLDivElement;
  onChange: ReturnType<typeof vi.fn>;
  onValidate: ReturnType<typeof vi.fn>;
  unmount: () => Promise<void>;
}

async function mount(
  edge: ProjectionEdge = projection(),
  issues: StudioGraphIssueLocation[] = [],
): Promise<Mounted> {
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  const onChange = vi.fn();
  const onValidate = vi.fn();
  await act(async () => {
    root.render(
      <ProjectionEditor
        projection={edge}
        populations={POPULATIONS}
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
      await act(async () => root.unmount());
      container.remove();
    },
  };
}

function input(container: HTMLElement, field: string): HTMLInputElement {
  const found = container.querySelector<HTMLInputElement>(`#projection-${field}`);
  if (found === null) throw new Error(`no input for ${field}`);
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
  it("names the projection by its endpoints", async () => {
    const mounted = await mount();

    expect(mounted.container.querySelector("h3")?.textContent).toBe("Exc 0 → Inh 0");
    expect(mounted.container.querySelector("section")?.getAttribute("aria-label")).toBe(
      "Projection Exc 0 → Inh 0",
    );
    await mounted.unmount();
  });

  it("labels every executed field", async () => {
    const mounted = await mount();
    const labels = [...mounted.container.querySelectorAll("label")].map(
      (label) => label.getAttribute("for"),
    );

    expect(labels).toEqual([
      "projection-weight",
      "projection-rule",
      "projection-probability",
      "projection-delay",
      "projection-seed",
      "projection-autapses",
    ]);
    await mounted.unmount();
  });

  it("ties the contract to its own input for a screen reader", async () => {
    const mounted = await mount();
    const weight = input(mounted.container, "weight");
    const described = weight.getAttribute("aria-describedby");

    expect(described).toBe("projection-weight-help");
    expect(mounted.container.querySelector(`#${described}`)?.textContent).toContain(
      "excitatory source needs a positive weight",
    );
    await mounted.unmount();
  });

  it("shows the current values rather than the defaults", async () => {
    const mounted = await mount(projection({ autapses: true, delay: 2, seed: 7, weight: -3.5 }));

    expect(input(mounted.container, "weight").value).toBe("-3.5");
    expect(input(mounted.container, "delay").value).toBe("2");
    expect(input(mounted.container, "seed").value).toBe("7");
    expect(input(mounted.container, "autapses").checked).toBe(true);
    await mounted.unmount();
  });

  it("disables the probability the all_to_all rule does not use", async () => {
    const mounted = await mount(projection({ probability: undefined, rule: "all_to_all" }));

    expect(input(mounted.container, "probability").disabled).toBe(true);
    await mounted.unmount();
  });
});

describe("editing a field", () => {
  it("sends the parsed value to the graph and asks the server", async () => {
    const mounted = await mount();

    await type(input(mounted.container, "weight"), "-40");

    expect(mounted.onChange).toHaveBeenCalledWith("e1", { weight: -40 });
    expect(mounted.onValidate).toHaveBeenCalledOnce();
    await mounted.unmount();
  });

  it("does not rewrite half-typed text under the hands typing it", async () => {
    const mounted = await mount();
    const weight = input(mounted.container, "weight");

    await type(weight, "-");

    // "-" is not a number, so nothing reaches the graph; the box keeps it so
    // the next keystroke can finish it.
    expect(weight.value).toBe("-");
    expect(mounted.onChange).not.toHaveBeenCalled();
    await mounted.unmount();
  });

  it("says which field is not a number, beside that field", async () => {
    const mounted = await mount();
    const delay = input(mounted.container, "delay");

    await type(delay, "soon");

    expect(delay.getAttribute("aria-invalid")).toBe("true");
    expect(delay.getAttribute("aria-describedby")).toContain("projection-delay-error");
    expect(
      mounted.container.querySelector("#projection-delay-error")?.textContent,
    ).toContain("delay must be a number");
    await mounted.unmount();
  });

  it("clears its own refusal once the text becomes a value", async () => {
    const mounted = await mount();
    const delay = input(mounted.container, "delay");

    await type(delay, "soon");
    await type(delay, "2");

    expect(mounted.container.querySelector("#projection-delay-error")).toBeNull();
    expect(mounted.onChange).toHaveBeenCalledWith("e1", { delay: 2 });
    await mounted.unmount();
  });

  it("drops the probability when the rule stops using one", async () => {
    const mounted = await mount();
    const rule = mounted.container.querySelector<HTMLSelectElement>("#projection-rule");

    await act(async () => {
      rule!.value = "all_to_all";
      rule!.dispatchEvent(new Event("change", { bubbles: true }));
    });

    expect(mounted.onChange).toHaveBeenCalledWith("e1", {
      probability: undefined,
      rule: "all_to_all",
    });
    await mounted.unmount();
  });

  it("reads the autapse box", async () => {
    const mounted = await mount();
    const autapses = input(mounted.container, "autapses");

    await act(async () => autapses.click());

    expect(mounted.onChange).toHaveBeenCalledWith("e1", { autapses: true });
    await mounted.unmount();
  });

  it("sends a value the server will refuse, because the server owns that answer", async () => {
    // An excitatory source with a negative weight is inadmissible. The editor
    // must not decide that: a second copy of the contract in the browser is a
    // copy free to drift from the one that runs.
    const mounted = await mount();

    await type(input(mounted.container, "weight"), "-40");

    expect(mounted.onChange).toHaveBeenCalledWith("e1", { weight: -40 });
    await mounted.unmount();
  });
});

describe("what the server said about a field", () => {
  const issues: StudioGraphIssueLocation[] = [
    {
      attribute: "weight",
      field: "projections[0].weight",
      id: "e1",
      kind: "projection",
      message: "Projection e1 weight -40 conflicts with the excitatory source population p1",
      subject: "Exc 0 → Inh 0",
    },
    {
      attribute: "",
      field: "projections[0]",
      id: "e1",
      kind: "projection",
      message: "Projection e1 has unknown fields: colour",
      subject: "Exc 0 → Inh 0",
    },
    {
      attribute: "delay",
      field: "projections[1].delay",
      id: "e2",
      kind: "projection",
      message: "Projection e2 delay is not a whole step",
      subject: "Inh 0 → Exc 0",
    },
  ];

  it("puts a field's failure on that field and marks it invalid", async () => {
    const mounted = await mount(projection({ weight: -40 }), issues);
    const weight = input(mounted.container, "weight");

    expect(weight.getAttribute("aria-invalid")).toBe("true");
    expect(
      mounted.container.querySelector("#projection-weight-error")?.textContent,
    ).toContain("conflicts with the excitatory source population");
    await mounted.unmount();
  });

  it("shows a failure about the projection as a whole rather than losing it", async () => {
    const mounted = await mount(projection(), issues);

    expect(mounted.container.textContent).toContain("has unknown fields: colour");
    await mounted.unmount();
  });

  it("leaves another projection's failure out of this editor", async () => {
    const mounted = await mount(projection(), issues);

    expect(mounted.container.textContent).not.toContain("Projection e2");
    expect(input(mounted.container, "delay").getAttribute("aria-invalid")).toBe("false");
    await mounted.unmount();
  });
});
