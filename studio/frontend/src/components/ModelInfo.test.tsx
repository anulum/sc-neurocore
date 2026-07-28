// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio model information rendering contracts

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import type { ModelDetail } from "../api/client";

const modelDetail = {
  name: "AdExNeuron",
  module: "adex",
  category: "Integrate-and-Fire",
  tier: 3,
  evidence_kind: "measured",
  science_tier: 5,
  science_label: "S5",
  silicon_tier: 1,
  silicon_label: "H1",
  validation_metric: "parity",
  integration_method: "euler",
  terminal_silicon_tier: "H1",
  terminal_reason: "Point-neuron schema→RTL path; higher rungs need proof.",
  category_slug: "integrate-and-fire",
  category_source: "declared",
  family: "Integrate-and-Fire",
  maturity: "validated",
  biophysical_detail: "point",
  n_state_vars: 2,
  n_params: 2,
  state_var_names: ["v", "w"],
  dt: 0.1,
  description: "Adaptive exponential integrate-and-fire neuron.",
  intended_use: [],
  hardware_fit: [],
  behavior_tags: [],
  provenance: null,
  docstring: "Adaptive exponential integrate-and-fire neuron.",
  display_name: "AdEx",
  state_vars: [
    { name: "v", default: -65, unit: "mV", meaning: "membrane voltage" },
    { name: "w", default: 0, unit: "pA", meaning: "adaptation current" },
  ],
  params: [],
  dynamics: {},
  backends: [],
  reproducibility: {
    reference_config: "adex.toml",
    golden_trace_sha256: "",
    reproducible: false,
  },
  documentation_slug: "models/adex",
} satisfies ModelDetail;

let activeModelDetail: ModelDetail = modelDetail;

vi.mock("../stores/studio", () => ({
  useStudioStore: () => ({
    sourceMode: "model",
    modelDetail: activeModelDetail,
    equations: [],
    odeParams: {},
    odeInit: {},
    dt: 0.1,
    duration: 10,
  }),
}));

describe("ModelInfo", () => {
  it("renders the descriptor-backed Studio consumption contract", async () => {
    const { default: ModelInfo } = await import("./ModelInfo");
    const markup = renderToStaticMarkup(<ModelInfo />);

    expect(markup).toContain("data-testid=\"model-validation-metric\"");
    expect(markup).toContain("validation:");
    expect(markup).toContain("parity");
    expect(markup).toContain("data-testid=\"model-integration-method\"");
    expect(markup).toContain("integrator:");
    expect(markup).toContain("euler");
    expect(markup).toContain("data-testid=\"model-terminal-reason\"");
    expect(markup).toContain("H1 —");
    expect(markup).toContain("Point-neuron schema→RTL path; higher rungs need proof.");
  });

  it("renders an honest fallback when no terminal target is declared", async () => {
    activeModelDetail = {
      ...modelDetail,
      terminal_silicon_tier: "",
      terminal_reason: "",
    };
    const { default: ModelInfo } = await import("./ModelInfo");
    const markup = renderToStaticMarkup(<ModelInfo />);

    expect(markup).toContain("none —");
    expect(markup).toContain("no terminal silicon target declared");
    activeModelDetail = modelDetail;
  });
});
