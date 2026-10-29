// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Hardware catalogue DOM contract
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import * as api from "../api/hardwareApi";
import type { HardwareCatalogueEntry, HardwareCatalogueResponse } from "../hardwareCatalogueTypes";
import HardwareCatalogue from "./HardwareCatalogue";

/** Explicit registration fixture; actual backend conservation has separate HTTP/browser tests. */
function registration(target_id: string, changes: Partial<HardwareCatalogueEntry> = {}): HardwareCatalogueEntry {
  return {
    target_id, name: target_id, kind: "neuromorphic", registration_sources: ["public.registry"],
    planning_profile: false, nir_profile: null, adapter_package: false,
    synthesis_profile: null, driver: null, requirements: ["declared dependency"],
    limitations: ["Not a physical run"], evidence_level: "registration", physical_evidence: [],
    ...changes,
  };
}

const nir = {
  target_id: "loihi2", display_name: "Loihi 2", backend_status: "capability_manifest",
  supported_nir_nodes: ["LIF", "IF"], unsupported_nir_nodes: ["CustomPythonNode"],
  sc_constraints: {
    bitstream_lengths: [64, 128], stream_transport: "packet_encoded_probability",
    precision_modes: ["unipolar"], stochastic_sources: ["deterministic_lfsr_seed"],
    back_annotation_channels: ["spike_drop_rate"],
  },
  notes: ["No SDK invocation"],
};
const payload: HardwareCatalogueResponse = {
  schema_version: "studio.hardware-catalogue.v1", evidence_scope: "Registration only; no physical measurements.",
  entries: [
    registration("loihi2", { name: "Loihi 2", planning_profile: true, nir_profile: nir, adapter_package: true }),
    registration("dynap_se", { name: "DYNAP-SE", nir_profile: { ...nir, target_id: "dynap_se", display_name: "DYNAP-SE" } }),
    registration("board:pynq_z2", { name: "PYNQ-Z2", kind: "board", driver: "sc_neurocore.drivers.SC_NeuroCore_Driver" }),
    ...["ice40", "ecp5", "gowin", "xilinx"].map((target) => registration("fpga:" + target, {
      name: target.toUpperCase(), kind: "fpga",
      synthesis_profile: {
        target, capacity_device: target.toUpperCase(), capacity: { luts: 5280, dsps: 8 },
        synthesis_command: "synth_" + target,
        routing_tool: target === "ecp5" ? "nextpnr-ecp5" : null,
        device_selector: target === "ecp5" ? "25k" : null,
        package_selector: target === "ecp5" ? "CABGA381" : null,
      },
    })),
  ],
};
let host: HTMLDivElement;
let root: Root;
const onSelectTarget = vi.fn<(target: string) => void>();
const onTargetsLoaded = vi.fn<(targets: string[]) => void>();

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  onSelectTarget.mockReset();
  onTargetsLoaded.mockReset();
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  host.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

/** Mount the real component with explicit contract-bound transport responses. */
async function mount(modelMode = false): Promise<void> {
  await act(async () => {
    root.render(<HardwareCatalogue selectedTarget="ice40" modelMode={modelMode}
      onSelectTarget={onSelectTarget} onTargetsLoaded={onTargetsLoaded} />);
  });
}

/** Resolve an actual DOM control rather than a component's private methods. */
function element<T extends HTMLElement>(selector: string, constructor: new () => T): T {
  const node = host.querySelector(selector);
  if (!(node instanceof constructor)) throw new Error("Missing control " + selector);
  return node;
}

/** Change a React input using its native DOM setter and event. */
async function search(value: string): Promise<void> {
  const input = element("input", HTMLInputElement);
  const descriptor = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
  if (descriptor?.set === undefined) throw new Error("Native input setter unavailable");
  const setter = descriptor.set.bind(input);
  await act(async () => {
    setter(value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

it("shows every registration kind, exact constraints and explicit evidence limits", async () => {
  vi.spyOn(api, "fetchHardwareCatalogue").mockResolvedValue(payload);
  await mount();
  expect(host.textContent).toContain("7 matching registrations");
  expect(host.textContent).toContain("Resource-planning profile");
  expect(host.textContent).toContain("LIF, IF");
  expect(host.textContent).toContain("Unsupported nodes: CustomPythonNode");
  expect(host.textContent).toContain("packet_encoded_probability");
  expect(host.textContent).toContain("64, 128");
  expect(host.textContent).toContain("Adapter handoff package exists");
  expect(host.textContent).toContain("CABGA381");
  expect(host.textContent).toContain("not device-bound");
  expect(host.textContent).toContain("not offered by this profile");
  expect(host.textContent).toContain("Physical evidence: no receipts attached");
  expect(onTargetsLoaded).toHaveBeenCalledWith(["ice40", "ecp5", "gowin", "xilinx"]);
  expect(element("button", HTMLButtonElement).getAttribute("aria-pressed")).toBe("true");
  await act(async () => { element('article[aria-label="GOWIN"] button', HTMLButtonElement).click(); });
  expect(onSelectTarget).toHaveBeenCalledWith("gowin");
});

it("filters by trimmed case-insensitive identity and category without changing the target", async () => {
  vi.spyOn(api, "fetchHardwareCatalogue").mockResolvedValue(payload);
  await mount();
  await search("  DyNaP  ");
  expect(host.textContent).toContain("1 matching registrations");
  expect(host.querySelectorAll("article")).toHaveLength(1);
  const select = element("select", HTMLSelectElement);
  await act(async () => { select.value = "board"; select.dispatchEvent(new Event("change", { bubbles: true })); });
  expect(host.textContent).toContain("0 matching registrations");
  await search("");
  expect(host.textContent).toContain("PYNQ-Z2");
  expect(host.querySelectorAll("article")).toHaveLength(1);
  await act(async () => { select.value = "fpga"; select.dispatchEvent(new Event("change", { bubbles: true })); });
  expect(host.querySelectorAll("article")).toHaveLength(4);
  expect(onSelectTarget).not.toHaveBeenCalled();
});

it("keeps catalogue-model selection within its two admitted routing targets", async () => {
  vi.spyOn(api, "fetchHardwareCatalogue").mockResolvedValue(payload);
  await mount(true);
  expect(element('article[aria-label="ICE40"] button', HTMLButtonElement).disabled).toBe(false);
  expect(element('article[aria-label="ECP5"] button', HTMLButtonElement).disabled).toBe(false);
  expect(element('article[aria-label="GOWIN"] button', HTMLButtonElement).disabled).toBe(true);
  expect(element('article[aria-label="XILINX"] button', HTMLButtonElement).disabled).toBe(true);
  await act(async () => { element('article[aria-label="GOWIN"] button', HTMLButtonElement).click(); });
  expect(onSelectTarget).not.toHaveBeenCalled();
});

it("reports loading and transport failure, then retries the genuine API helper", async () => {
  let resolve: (value: HardwareCatalogueResponse) => void = () => { throw new Error("Not started"); };
  vi.spyOn(api, "fetchHardwareCatalogue")
    .mockReturnValueOnce(new Promise((done) => { resolve = done; }))
    .mockRejectedValueOnce(new Error("Backend unavailable"))
    .mockResolvedValueOnce(payload);
  await mount();
  expect(host.textContent).toContain("Loading hardware catalogue");
  await act(async () => { resolve(payload); });
  expect(host.textContent).toContain("Loihi 2");
  await act(async () => {
    root.render(<HardwareCatalogue selectedTarget="ice40" modelMode={false}
      onSelectTarget={onSelectTarget} onTargetsLoaded={vi.fn()} />);
  });
  expect(host.textContent).toContain("Backend unavailable");
  await act(async () => { element("button", HTMLButtonElement).click(); });
  expect(host.textContent).toContain("7 matching registrations");
});

it("uses a readable failure for a non-Error transport rejection", async () => {
  vi.spyOn(api, "fetchHardwareCatalogue").mockRejectedValue("offline");
  await mount();
  expect(host.textContent).toContain("Hardware catalogue request failed.");
});

it.each(["resolve", "reject"] as const)("ignores a late %s after the workbench unmounts", async (outcome) => {
  let resolve: (value: HardwareCatalogueResponse) => void = () => { throw new Error("Not started"); };
  let reject: (cause: Error) => void = () => { throw new Error("Not started"); };
  vi.spyOn(api, "fetchHardwareCatalogue").mockReturnValue(new Promise((done, fail) => {
    resolve = done; reject = fail;
  }));
  await mount();
  await act(async () => { root.unmount(); });
  root = createRoot(host);
  await act(async () => {
    if (outcome === "resolve") resolve(payload);
    else reject(new Error("late"));
  });
  expect(onTargetsLoaded).not.toHaveBeenCalled();
  expect(host.textContent).toBe("");
});
