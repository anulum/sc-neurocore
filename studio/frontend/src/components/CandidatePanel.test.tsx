// @vitest-environment happy-dom
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — The candidate panel keeps the draft and refuses to send what is not JSON

/**
 * What the panel does before any request: it keeps the workspace's draft as
 * typed, imports a file into it, and refuses to send a draft that is empty or
 * not JSON, saying why. The server-backed actions are exercised against a real
 * backend in the live browser suite.
 */

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { useStudioStore } from "../stores/studio";
import CandidatePanel from "./CandidatePanel";

let container: HTMLDivElement;
let root: Root;

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

describe("CandidatePanel", () => {
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
});
