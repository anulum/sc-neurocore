// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { describe, expect, it } from "vitest";

import type { StudioCapability } from "./api/client";
import {
  capabilityById,
  capabilityFailureState,
  capabilityLoadedState,
  capabilityLoadingState,
  panelCapabilityState,
  summarizeCapabilities,
} from "./capabilityShell";

function capability(
  overrides: Partial<StudioCapability> & Pick<StudioCapability, "capability_id" | "title">,
): StudioCapability {
  return {
    capability_id: overrides.capability_id,
    title: overrides.title,
    summary: overrides.summary ?? `${overrides.title} summary`,
    status: overrides.status ?? "stable",
    healthy: overrides.healthy ?? true,
    message: overrides.message ?? "Capability is available.",
    requirements: overrides.requirements ?? [],
    evidence: overrides.evidence ?? ["contract_test"],
    ui_placement: overrides.ui_placement ?? "Build",
    docs_path: overrides.docs_path ?? "docs/studio/index.md",
  };
}

describe("capability shell contract", () => {
  it("summarizes registry health without duplicating backend state", () => {
    const capabilities = [
      capability({ capability_id: "studio.api", title: "Studio API" }),
      capability({
        capability_id: "studio.synthesis_dashboard",
        title: "Synthesis Dashboard",
        status: "unavailable",
        healthy: false,
      }),
      capability({
        capability_id: "studio.network_canvas",
        title: "Network Canvas",
        status: "experimental",
      }),
    ];

    expect(summarizeCapabilities(capabilities)).toEqual({
      total: 3,
      healthy: 2,
      unavailable: 1,
      degraded: 0,
      experimental: 1,
      stable: 1,
      headline: "2/3 ready",
      worstStatus: "unavailable",
    });
  });

  it("resolves capabilities by stable backend ID", () => {
    const api = capability({ capability_id: "studio.api", title: "Studio API" });

    expect(capabilityById([api], "studio.api")).toBe(api);
    expect(capabilityById([api], "studio.missing")).toBeNull();
  });

  it("builds capability registry loading and loaded state patches", () => {
    const api = capability({ capability_id: "studio.api", title: "Studio API" });

    expect(capabilityLoadingState()).toEqual({
      capabilitiesError: null,
      capabilitiesLoading: true,
    });
    expect(capabilityLoadedState([api])).toEqual({
      capabilities: [api],
      capabilitiesError: null,
      capabilitiesLoading: false,
    });
  });

  it("builds capability registry failure state patches", () => {
    expect(capabilityFailureState(new Error("registry offline"))).toEqual({
      capabilitiesError: "registry offline",
      capabilitiesLoading: false,
    });
    expect(capabilityFailureState("bad")).toEqual({
      capabilitiesError: "Capability check failed",
      capabilitiesLoading: false,
    });
  });

  it("blocks unavailable panels with requirement and evidence details", () => {
    const synthesis = capability({
      capability_id: "studio.synthesis_dashboard",
      title: "Synthesis Dashboard",
      status: "unavailable",
      healthy: false,
      message: "One or more capability requirements are unavailable.",
      requirements: [
        { name: "yosys", available: false, detail: "external tool availability not checked" },
      ],
      evidence: ["static_inventory"],
      docs_path: "docs/studio/synthesis-dashboard.md",
    });

    expect(panelCapabilityState([synthesis], "synth")).toEqual({
      panelKey: "synth",
      capabilityId: "studio.synthesis_dashboard",
      title: "Synthesis Dashboard",
      available: false,
      status: "unavailable",
      message: "One or more capability requirements are unavailable.",
      requirements: ["yosys: external tool availability not checked"],
      evidence: ["static_inventory"],
      docsPath: "docs/studio/synthesis-dashboard.md",
    });
  });

  it("binds stateful workbench panels to their backend capability contracts", () => {
    const analysis = capability({
      capability_id: "studio.analysis_suite",
      title: "Analysis Suite",
      status: "unavailable",
      healthy: false,
      message: "Analysis service offline.",
      requirements: [{ name: "analysis", available: false, detail: "endpoint disabled" }],
    });
    const compiler = capability({
      capability_id: "studio.compiler_inspector",
      title: "Compiler Inspector",
      status: "experimental",
      healthy: true,
    });

    expect(panelCapabilityState([analysis, compiler], "fi-curve")).toMatchObject({
      capabilityId: "studio.analysis_suite",
      available: false,
      title: "Analysis Suite",
    });
    expect(panelCapabilityState([analysis, compiler], "ir")).toMatchObject({
      capabilityId: "studio.compiler_inspector",
      available: true,
      title: "Compiler Inspector",
    });
  });

  it("fails closed when a bound panel capability is missing from the registry", () => {
    expect(panelCapabilityState([], "trace")).toEqual({
      panelKey: "trace",
      capabilityId: "studio.simulation_workbench",
      title: "Trace",
      available: false,
      status: "unavailable",
      message: "Backend capability contract is missing from the registry.",
      requirements: [],
      evidence: [],
      docsPath: null,
    });
  });
});
