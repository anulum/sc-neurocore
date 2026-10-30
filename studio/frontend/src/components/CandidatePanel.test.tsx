// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — The candidate panel keeps the draft and refuses to send what is not JSON
// @vitest-environment happy-dom

/**
 * What the panel does before any request: it keeps the workspace's draft as
 * typed, imports a file into it, and refuses to send a draft that is empty or
 * not JSON, saying why. The server-backed actions are exercised against a real
 * backend in the live browser suite.
 */

import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { Window } from "happy-dom";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import { useStudioStore } from "../stores/studio";
import CandidatePanel from "./CandidatePanel";
import { startCandidateTestServer, type CandidateTestServer } from "../candidateTestServer";
import { listCandidateJobs } from "../api/candidatesApi";

let container: HTMLDivElement;
let root: Root;
let server: CandidateTestServer;
beforeAll(async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  server = await startCandidateTestServer();
  const dom = window as unknown as Window;
  dom.happyDOM.setURL(server.origin);
  dom.happyDOM.settings.fetch.requestHeaders = [{ url: /^http:\/\/127\.0\.0\.1:[0-9]+\/api(?:\/|$)/, headers: { "sec-fetch-site": "same-origin" } }];
}, 40_000);
afterAll(async () => { await server.stop(); });

beforeEach(async () => {
  useStudioStore.setState({ candidates: [] });
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => { root.render(<CandidatePanel />); });
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove();
  await (window as unknown as Window).happyDOM.waitUntilComplete();
});

/**
 * Find a button by its text.
 *
 * @param text - The button's label.
 * @returns The button.
 */
function button(text: string): HTMLButtonElement {
  const found = [...container.querySelectorAll("button")].find((each) => each.textContent === text);
  if (found === undefined) throw new Error(`no button ${text}`);
  return found;
}

/** The status region the panel reports into. */
function status(): string {
  return container.querySelector("#candidate-outcome")?.textContent ?? "";
}

/** The fields varied in the actual committed AdEx candidate fixture. */
interface CandidateFixture {
  name: string;
  parent: string | null;
  authors: string[];
  model: { parameters: Record<string, number>; dynamics: Record<string, string> };
  units: { state: Record<string, string> };
  reference_tests: { steps: number; expect: { spike_count: { min?: number; max?: number } } }[];
}

/** Read a fresh copy of the maintained candidate used by the browser suite. */
function candidateFixture(): CandidateFixture {
  return JSON.parse(readFileSync(resolve(import.meta.dirname, "../../e2e/fixtures/SlowAdaptationAdEx.candidate.json"), "utf8")) as CandidateFixture;
}

/** Enter text through the native DOM setter and React's actual input event. */
async function fill(selector: string, value: string): Promise<void> {
  const field = container.querySelector<HTMLInputElement | HTMLTextAreaElement>(selector);
  if (field === null) throw new Error(`No input ${selector}`);
  const prototype = field instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(prototype, "value")?.set?.bind(field);
  if (setter === undefined) throw new Error("Native input setter unavailable.");
  await act(async () => {
    setter(value);
    field.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

/** Observe real HTTP and worker completion without replacing timers or fetch. */
async function until(condition: () => boolean): Promise<void> {
  const deadline = Date.now() + 45_000;
  while (!condition()) {
    if (Date.now() >= deadline) throw new Error(`Candidate action expired: ${status()}`);
    await act(async () => { await new Promise((accept) => setTimeout(accept, 50)); });
  }
}

/** Click the actual panel action and observe its asynchronous result. */
async function press(label: string): Promise<void> {
  await act(async () => { button(label).click(); });
}

describe("CandidatePanel", () => {
  it("adopts actual canonical text only after the explicit workspace action", async () => {
    await act(async () => { useStudioStore.setState({ selectedModelName: "AdExNeuron" }); });
    await press("Prepare catalogue draft");
    await until(() => container.querySelector('[aria-label="Catalogue candidate preview"]') !== null);
    const text = container.querySelector<HTMLTextAreaElement>('[aria-label="Catalogue candidate preview"]')?.value;
    expect(useStudioStore.getState().candidates).toEqual([]);
    await press("Use catalogue draft");
    expect(useStudioStore.getState().candidates).toEqual([{ text }]);
    expect(container.querySelector<HTMLTextAreaElement>("#candidate-draft")?.value).toBe(text);
    expect(status()).toBe("");
  });

  it("says a candidate is a proposal and labels its draft", () => {
    expect(container.textContent).toContain("It is never listed as a catalogue model");
    const draft = container.querySelector<HTMLTextAreaElement>("#candidate-draft");
    if (draft === null) throw new Error("no draft field");
    expect(draft.labels[0]?.textContent).toBe("Candidate package (JSON)");
    expect(draft.getAttribute("aria-describedby")).toBe("candidate-outcome");
    expect(button("Export candidate").disabled).toBe(true);
    expect(button("Export review packet").disabled).toBe(true);
  });

  it("refuses to send an empty draft and says so", async () => {
    await act(async () => { button("Validate").click(); });
    expect(status()).toBe("The draft is empty: import a candidate package or write one.");
  });

  it("keeps a draft that is not JSON and refuses to send it", async () => {
    await act(async () => { useStudioStore.getState().setCandidateDraft('{"name": '); });
    await act(async () => { button("Simulate").click(); });
    expect(status()).toMatch(/^The draft is not JSON: /);
    expect(useStudioStore.getState().candidates).toEqual([{ text: '{"name": ' }]);
    expect(button("Export candidate").disabled).toBe(false);
  });

  it("imports a file into the workspace's draft exactly as it reads", async () => {
    const text = '{\n  "name": "Imported",\n  "notes": "kept as written"\n}\n';
    const input = container.querySelector<HTMLInputElement>('input[type="file"]');
    if (input === null) throw new Error("no file input");
    Object.defineProperty(input, "files", { value: [new File([text], "x.candidate.json")] });
    await act(async () => {
      input.dispatchEvent(new Event("change", { bubbles: true }));
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(useStudioStore.getState().candidates).toEqual([{ text }]);
    expect(container.querySelector<HTMLTextAreaElement>("#candidate-draft")?.value).toBe(text);
  });

  it("clears the draft from the workspace when it is emptied", async () => {
    await act(async () => { useStudioStore.getState().setCandidateDraft("{}"); });
    await act(async () => { useStudioStore.getState().setCandidateDraft(""); });
    expect(useStudioStore.getState().candidates).toEqual([]);
  });

  it("validates, diffs and simulates the typed scientific draft through real workers", async () => {
    const document = candidateFixture();
    await fill("#candidate-draft", JSON.stringify(document));
    expect(useStudioStore.getState().candidates[0]?.text).toBe(JSON.stringify(document));
    await press("Validate");
    await until(() => status().includes("Valid. Digest"));
    await press("Diff against parent");
    await until(() => status().includes("What the candidate changes against AdExNeuron"));
    expect(status()).toContain("tau_w");
    document.model.parameters.tau_w = 100;
    await fill("#candidate-draft", JSON.stringify(document));
    await press("Diff against parent");
    await until(() => status().includes("The candidate changes nothing against AdExNeuron"));
    document.parent = null;
    await fill("#candidate-draft", JSON.stringify(document));
    await press("Diff against parent");
    await until(() => status().includes("The candidate names no parent"));
    document.parent = "ATypeKNeuron";
    await fill("#candidate-draft", JSON.stringify(document));
    await press("Diff against parent");
    await until(() => status().includes("has no canonical schema"));
    await fill("#candidate-current", "800");
    await fill("#candidate-steps", "20");
    await press("Simulate");
    await until(() => status().includes("spikes in 20 steps"));
    expect(status()).toContain("at 800 pA");
    expect(status()).toContain("final state");
    await until(() => container.querySelector('[aria-label="Stored candidate jobs"]') !== null);
    await press("Refresh candidate jobs");
  }, 90_000);

  it("reports located refusals and preserves passing, failing and diverged reference results", async () => {
    const document = candidateFixture();
    document.reference_tests = document.reference_tests.slice(0, 1);
    const reference = document.reference_tests[0];
    if (reference === undefined) throw new Error("Missing actual candidate reference.");
    reference.steps = 3;
    document.units.state.v = "ms";
    await fill("#candidate-draft", JSON.stringify(document));
    await press("Validate");
    await until(() => container.querySelector('[aria-label="Candidate problems"]') !== null);
    expect(status()).toContain("/units/state/v");
    await press("Simulate");
    await until(() => status().includes("The candidate is not valid; fix the fields listed."));
    document.units.state.v = "mV";
    const authors = document.authors;
    document.authors = ["Invalid Unicode \ud800"];
    await fill("#candidate-draft", JSON.stringify(document));
    await press("Validate");
    await until(() => status().includes("(document)"));
    expect(status()).toContain("candidate text must contain valid Unicode");
    document.authors = authors;
    await fill("#candidate-draft", JSON.stringify(document));
    await press("Run reference tests");
    await until(() => status().includes("Reference tests passed."));
    expect(status()).toContain("Not established:");
    expect(button("Export review packet").disabled).toBe(false);
    reference.expect.spike_count = { min: 1 };
    await fill("#candidate-draft", JSON.stringify(document));
    await press("Run reference tests");
    await until(() => status().includes("Reference tests failed."));
    expect(status()).toContain("outside bounds");
    document.model.dynamics.v = "exp(1000) * v / tau";
    await fill("#candidate-draft", JSON.stringify(document));
    await press("Simulate");
    await until(() => status().includes("diverged at step"));
    expect(status()).not.toContain("final state");
  }, 90_000);

  it("exports original job bytes after draft changes and reports actual download and transport failures", async () => {
    const document = candidateFixture();
    document.reference_tests = document.reference_tests.slice(0, 1);
    const reference = document.reference_tests[0];
    if (reference === undefined) throw new Error("Missing actual candidate reference.");
    reference.steps = 3;
    const text = JSON.stringify(document, null, 2) + "\n";
    const urls = vi.spyOn(URL, "createObjectURL");
    const anchors = vi.spyOn(HTMLAnchorElement.prototype, "click");
    /** Read the filename from the actual anchor that initiated the download. */
    function downloadedName(): string {
      const anchor: unknown = anchors.mock.contexts.at(-1);
      if (!(anchor instanceof HTMLAnchorElement)) throw new Error("Download did not click an anchor.");
      return anchor.download;
    }
    const dom = window as unknown as Window;
    try {
      await fill("#candidate-draft", text);
      const input = container.querySelector<HTMLInputElement>('input[type="file"]');
      if (input === null) throw new Error("Missing import input.");
      await press("Import candidate");
      await act(async () => { input.dispatchEvent(new Event("change", { bubbles: true })); });
      expect(useStudioStore.getState().candidates[0]?.text).toBe(text);
      await press("Export candidate");
      const exported = urls.mock.calls.at(-1)?.[0];
      if (!(exported instanceof Blob)) throw new Error("Candidate download did not receive a Blob.");
      expect(await exported.text()).toBe(text);
      expect(downloadedName()).toBe(`${document.name}.candidate.json`);
      // Happy DOM navigates download anchors; Chromium separately checks file delivery.
      dom.happyDOM.setURL(server.origin);
      await fill("#candidate-draft", '{"name": ');
      await press("Export candidate");
      expect(downloadedName()).toBe("candidate.candidate.json");
      dom.happyDOM.setURL(server.origin);
      await fill("#candidate-draft", text);
      await press("Run reference tests");
      await until(() => status().includes("Reference tests passed."));
      const [job] = await listCandidateJobs();
      if (job === undefined) throw new Error("Completed reference job was not retained.");
      expect(job.admission.candidate_operation).toBe("review");
      await fill("#candidate-draft", '{"name":"ChangedDraft"}');
      let count = urls.mock.calls.length;
      await press("Export review packet");
      await until(() => urls.mock.calls.length === count + 1);
      dom.happyDOM.setURL(server.origin);
      expect(downloadedName()).toBe(`${document.name}.review.json`);
      const packet = urls.mock.calls.at(-1)?.[0];
      if (!(packet instanceof Blob)) throw new Error("Review download did not receive a Blob.");
      const stored: unknown = JSON.parse(await packet.text());
      expect(stored).toHaveProperty("candidate.name", document.name);
      expect(stored).toHaveProperty("candidate_sha256", job.admission.candidate_sha256);
      count = urls.mock.calls.length;
      await press("Download request.json");
      await until(() => urls.mock.calls.length === count + 1);
      dom.happyDOM.setURL(server.origin);
      const request = urls.mock.calls.at(-1)?.[0];
      if (!(request instanceof Blob)) throw new Error("Request download did not receive a Blob.");
      expect(createHash("sha256").update(Buffer.from(await request.arrayBuffer())).digest("hex"))
        .toBe(job.admission.request_sha256);
      expect(downloadedName()).toBe(`${job.job_id}.request.json`);
      const path = join(server.root, "jobs", job.job_id, "result.json");
      const original = readFileSync(path);
      try {
        writeFileSync(path, "{}");
        await press("Export review packet");
        await until(() => status().includes("candidate_job_artifact_unavailable"));
      } finally { writeFileSync(path, original); }
      const before = status();
      dom.happyDOM.setURL("http://127.0.0.1:1");
      await press("Validate");
      await until(() => status() !== before);
      expect(status()).not.toBe("");
      expect(button("Validate").disabled).toBe(false);
    } finally {
      dom.happyDOM.setURL(server.origin);
      urls.mockRestore(); anchors.mockRestore();
    }
  }, 90_000);
});
