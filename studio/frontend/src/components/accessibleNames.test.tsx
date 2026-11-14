// @vitest-environment happy-dom
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Accessible names of the model list and the RTL panel

import { renderToStaticMarkup } from "react-dom/server";
import type { ReactElement } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

/*
 * Measured before this was written, in a real browser's accessibility tree: a model row's
 * select button was named "AdExvalidation parity · integrator euler · …", its short label
 * run into the contract summary; the co-simulation controls' aria-label sat on a div with
 * no role, so no name was exposed; and the RTL viewer was announced as Monaco's default
 * "Editor content". happy-dom computes no accessible names, so these tests resolve the
 * references the way the accessible-name computation's step 2B does: the text of each
 * referenced element, joined by spaces.
 */

const { mockStore } = vi.hoisted(() => {
  const store: { current: Record<string, unknown> } = { current: {} };
  return { mockStore: store };
});

vi.mock("../stores/studio", () => ({ useStudioStore: () => mockStore.current }));
vi.mock("../useModelScanJob", async () => {
  const { initialModelScanJobState } = await vi.importActual<typeof import("../modelScanJob")>("../modelScanJob");
  return {
    useModelScanJob: () => ({
      busy: false, canSubmit: true, phaseLabel: "Scan", startScan: vi.fn(), state: initialModelScanJobState(),
    }),
  };
});
vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return { ...actual, fetchModelFacets: async () => null };
});
vi.mock("@monaco-editor/react", () => ({
  default: (props: { options?: { ariaLabel?: string } }) => (
    <div data-testid="monaco" data-aria-label={props.options?.ariaLabel ?? "Editor content"} />
  ),
}));

/**
 * One catalogue row, with the fields the model list reads.
 *
 * @param name - The model's class name.
 * @param metadataState - Whether a committed descriptor describes it.
 * @returns The row.
 */
function catalogueRow(name: string, metadataState: string): Record<string, unknown> {
  return {
    name, category: "IF", family: "IF", tier: 2, science_tier: 3, silicon_tier: null,
    science_label: "S3", silicon_label: "none", validation_metric: "parity",
    integration_method: "euler", terminal_silicon_tier: "H1", terminal_reason: "RTL parity target met.",
    behavior_tags: [], description: name, maturity: "validated", evidence_kind: "publication",
    provenance: null, state_var_names: ["v"], n_params: 1, metadata_state: metadataState,
  };
}

/**
 * Render a component's markup into the test document.
 *
 * @param element - The component to render.
 * @returns The document body holding it.
 */
function mount(element: ReactElement): HTMLElement {
  document.body.innerHTML = renderToStaticMarkup(element);
  return document.body;
}

/**
 * The text of the elements an ID reference list names, joined by spaces (accname step 2B).
 *
 * @param refs - A space-separated ID reference list.
 * @returns The joined text; every reference must resolve to exactly one element.
 */
function referencedText(refs: string | null): string {
  expect(refs, "the element names no references").toBeTruthy();
  return (refs ?? "").split(/\s+/).map((id) => {
    expect(document.querySelectorAll(`[id="${id}"]`).length, id).toBe(1);
    return (document.getElementById(id)?.textContent ?? "").trim();
  }).join(" ");
}

describe("model list rows", () => {
  beforeEach(() => {
    mockStore.current = {
      models: [catalogueRow("AdExNeuron", "available"), catalogueRow("ThetaNeuron", "unavailable")],
      selectedModelName: "AdExNeuron", modelFilter: "",
      loadModels: vi.fn(), selectModel: vi.fn(), setModelFilter: vi.fn(),
    };
  });

  it("names each row's select button by the label it shows, and describes it by its contract", async () => {
    const { default: ModelBrowser } = await import("./ModelBrowser");
    const body = mount(<ModelBrowser />);
    const [adex, theta, ...others] = [...body.querySelectorAll("button.model-row-select")];
    if (!adex || !theta || others.length > 0) throw new Error("expected exactly two model rows");

    expect(referencedText(adex.getAttribute("aria-labelledby"))).toBe("AdEx");
    expect(referencedText(theta.getAttribute("aria-labelledby"))).toBe("Theta [unavailable]");
    expect(referencedText(adex.getAttribute("aria-describedby"))).toBe(
      "validation parity · integrator euler · terminal H1: RTL parity target met.",
    );
  });
});

describe("RTL panel", () => {
  beforeEach(() => {
    mockStore.current = {
      cosimResult: null, compileTraceability: null, isSimulating: false,
      modelDetail: { compile_configuration: { cosim_integrators: ["map"] } }, modelIntegrator: "map",
      runCosim: vi.fn(), sourceMode: "model", verilogSrc: "module sc_model; endmodule",
    };
  });

  it("exposes the co-simulation controls as a named group", async () => {
    const { default: VerilogPreview } = await import("./VerilogPreview");
    const body = mount(<VerilogPreview />);
    const named = body.querySelector('[aria-label="Selected model RTL co-simulation"]');

    expect(named?.getAttribute("role")).toBe("group");
    expect(named?.querySelector("button")?.textContent).toBe("Run RTL co-sim");
  });

  it("names the RTL viewer for what it shows, not Monaco's default", async () => {
    const { default: VerilogPreview } = await import("./VerilogPreview");
    const body = mount(<VerilogPreview />);

    expect(body.querySelector('[data-testid="monaco"]')?.getAttribute("data-aria-label")).toBe("Generated RTL");
  });
});
