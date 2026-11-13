// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — the capability strip along the top of the Studio, mounted with its store
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { StudioCapability } from "../api/client";
import { summarizeCapabilities } from "../capabilityShell";
import { publishedDocsUrl } from "../studioDocs";
import CapabilityStrip from "./CapabilityStrip";

const store = vi.hoisted(() => ({
  capabilities: [] as StudioCapability[],
  capabilitiesError: null as string | null,
  capabilitiesLoading: false,
}));

vi.mock("../stores/studio", () => ({
  useStudioStore: () => store,
}));

/**
 * One capability with the fields the strip reads.
 *
 * @param overrides - Fields that differ from a healthy stable capability.
 * @returns The capability.
 */
function capability(overrides: Partial<StudioCapability>): StudioCapability {
  return {
    capability_id: "simulate",
    title: "Simulation",
    summary: "Run a model over a protocol.",
    status: "stable",
    healthy: true,
    message: "",
    requirements: [],
    evidence: ["receipt"],
    ui_placement: "toolbar",
    docs_path: null,
    ...overrides,
  };
}

let root: Root | undefined;
let container: HTMLDivElement;

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  store.capabilities = [];
  store.capabilitiesError = null;
  store.capabilitiesLoading = false;
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root?.unmount(); await Promise.resolve(); });
  document.body.replaceChildren();
  vi.unstubAllGlobals();
});

/** Mount the strip. */
async function mount(): Promise<void> {
  await act(async () => { root?.render(<CapabilityStrip />); await Promise.resolve(); });
}

describe("CapabilityStrip", () => {
  it("says it is loading while the registry is on its way", async () => {
    store.capabilitiesLoading = true;
    await mount();
    const strip = container.querySelector(".capability-strip");
    expect(strip?.textContent).toBe("capabilities...");
    expect(strip?.getAttribute("title")).toBe("Loading Studio capability registry");
  });

  it("says the check failed, with the reason in its title", async () => {
    store.capabilitiesError = "registry unreachable";
    await mount();
    const strip = container.querySelector(".capability-strip-warning");
    expect(strip?.textContent).toBe("capability check failed");
    expect(strip?.getAttribute("title")).toBe("registry unreachable");
  });

  it("draws nothing when the registry is empty", async () => {
    await mount();
    expect(container.innerHTML).toBe("");
  });

  it("states that every capability is available when all are healthy", async () => {
    store.capabilities = [capability({}), capability({ capability_id: "train", title: "Training", status: "experimental" })];
    await mount();
    const details = container.querySelector("details");
    expect(details?.className).toBe("capability-strip");
    expect(details?.getAttribute("title")).toBe("All registered capabilities are available");
    expect(container.querySelector("summary > span")?.textContent).toBe(summarizeCapabilities(store.capabilities).headline);
    expect(container.querySelector(".capability-strip-detail")).toBeNull();
    expect([...container.querySelectorAll(".capability-badge")].map((badge) => badge.className)).toEqual([
      "capability-badge capability-badge-stable",
      "capability-badge capability-badge-experimental",
    ]);
  });

  it("names the first unhealthy capability and lists what it is missing", async () => {
    store.capabilities = [
      capability({}),
      capability({
        capability_id: "rtl",
        title: "RTL generation",
        status: "degraded",
        healthy: false,
        message: "Yosys is not installed",
        evidence: [],
        requirements: [
          { name: "yosys", available: false, detail: "not on the path" },
          { name: "python", available: true, detail: "3.12" },
        ],
      }),
    ];
    await mount();
    const details = container.querySelector("details");
    expect(details?.className).toBe("capability-strip capability-strip-warning");
    expect(details?.getAttribute("title")).toBe("RTL generation: Yosys is not installed");
    expect(container.querySelector(".capability-strip-detail")?.textContent).toBe("RTL generation");
    const rows = container.querySelectorAll(".capability-menu-row");
    expect(rows[1]?.querySelector(".capability-menu-meta")?.textContent).toContain("no evidence");
    expect([...(rows[1]?.querySelectorAll(".capability-menu-requirements li") ?? [])].map((item) => item.textContent)).toEqual([
      "yosys: not on the path",
    ]);
    expect(rows[0]?.querySelector(".capability-menu-requirements")).toBeNull();
  });

  it("links a capability's documentation, named for a reader and opening in a new tab", async () => {
    store.capabilities = [capability({ docs_path: "studio/simulation.md" }), capability({ capability_id: "plain" })];
    await mount();
    const links = container.querySelectorAll("a");
    expect(links).toHaveLength(1);
    expect(links[0]?.getAttribute("href")).toBe(publishedDocsUrl("studio/simulation.md"));
    expect(links[0]?.getAttribute("aria-label")).toBe("Simulation documentation (opens in a new tab)");
    expect(links[0]?.getAttribute("target")).toBe("_blank");
    expect(links[0]?.getAttribute("rel")).toBe("noreferrer");
  });
});
