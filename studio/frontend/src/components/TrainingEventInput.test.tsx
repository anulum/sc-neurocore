// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — importing and applying an event training declaration
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { readEventTrainingData } from "../studioEventTrainingData";
import eventData from "../studioEventTrainingData.fixture.json";
import type { StudioProjectTrainingConfig } from "../studioProjectState";
import TrainingEventInput from "./TrainingEventInput";

const reader = vi.hoisted(() => ({ failure: null as Error | null }));

vi.mock("../studioEventTrainingData", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../studioEventTrainingData")>();
  return {
    ...actual,
    readEventTrainingData: (...args: Parameters<typeof actual.readEventTrainingData>) => {
      // Lets one case reach the fallback message for a failure that is not an instance of Error.
      if (reader.failure !== null) throw reader.failure;
      return actual.readEventTrainingData(...args);
    },
  };
});

/** A failure shaped like an Error that is not an instance of this realm's Error, as one from a worker is not. */
const FOREIGN_FAILURE: Error = { name: "NotReadableError", message: "the worker stopped" };

const APPLIED = readEventTrainingData(eventData, "nmnist", 4);
if (APPLIED === undefined) throw new Error("the fixture no longer reads as an event declaration");

const BASE: StudioProjectTrainingConfig = {
  dataset: "nmnist", epochs: 1, batch_size: 3, lr: 0.001, hidden: [4], timesteps: 4,
  surrogate: "atan_surrogate", learn_beta: false, learn_threshold: false,
};

let root: Root | undefined;
let container: HTMLDivElement;
const setConfig = vi.fn();
const onDirtyChange = vi.fn();

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  reader.failure = null;
  setConfig.mockClear();
  onDirtyChange.mockClear();
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root?.unmount(); await Promise.resolve(); });
  document.body.replaceChildren();
  vi.unstubAllGlobals();
});

/**
 * Render the editor.
 *
 * @param config - The run's configuration.
 */
async function render(config: StudioProjectTrainingConfig = BASE): Promise<void> {
  await act(async () => {
    root?.render(<TrainingEventInput config={config} setConfig={setConfig} onDirtyChange={onDirtyChange} />);
    await Promise.resolve();
  });
}

/** The declaration's text area. */
function textarea(): HTMLTextAreaElement {
  const element = container.querySelector<HTMLTextAreaElement>('textarea[aria-label="Event input JSON"]');
  if (element === null) throw new Error("no declaration field");
  return element;
}

/**
 * Type a declaration through the native setter React listens for.
 *
 * @param text - The declaration.
 */
async function type(text: string): Promise<void> {
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set?.call(textarea(), text);
    textarea().dispatchEvent(new Event("input", { bubbles: true }));
    await Promise.resolve();
  });
}

/**
 * Press a button by its text.
 *
 * @param text - The label.
 */
async function press(text: string): Promise<void> {
  const button = [...container.querySelectorAll("button")].find((item) => item.textContent === text);
  if (button === undefined) throw new Error(`no button reads ${text}`);
  await act(async () => { button.click(); await Promise.resolve(); });
}

/** The alert's text, or null without one. */
function alertText(): string | null {
  return container.querySelector('[role="alert"]')?.textContent ?? null;
}

/** A file whose read the test settles by hand. */
interface PendingFile {
  file: File;
  resolve: (text: string) => void;
  reject: (reason: unknown) => void;
}

/**
 * A file whose contents arrive only when the test says so.
 *
 * @returns The file and the hands that settle its read.
 */
function pendingFile(): PendingFile {
  let resolve: (text: string) => void = () => undefined;
  let reject: (reason: unknown) => void = () => undefined;
  const read = new Promise<string>((onResolve, onReject) => { resolve = onResolve; reject = onReject; });
  return { file: { text: () => read } as unknown as File, resolve, reject };
}

/**
 * Select files in the file input.
 *
 * @param files - The selection.
 * @returns The input.
 */
async function select(files: File[]): Promise<HTMLInputElement> {
  const input = container.querySelector<HTMLInputElement>('input[aria-label="Event input file"]');
  if (input === null) throw new Error("no file input");
  Object.defineProperty(input, "files", { configurable: true, value: files });
  await act(async () => { input.dispatchEvent(new Event("change", { bubbles: true })); await Promise.resolve(); });
  return input;
}

/**
 * Let a settled read reach the component.
 *
 * @param settle - What settles the read.
 */
async function settle(settle: () => void): Promise<void> {
  await act(async () => { settle(); await new Promise((done) => { setTimeout(done, 0); }); });
}

describe("TrainingEventInput", () => {
  it("starts empty and clean without an applied declaration", async () => {
    await render();
    expect(textarea().value).toBe("");
    expect(onDirtyChange).toHaveBeenLastCalledWith(false);
    expect(container.querySelector("dl")).toBeNull();
    expect(alertText()).toBeNull();
  });

  it("shows an applied declaration's text and identity", async () => {
    await render({ ...BASE, event_data: APPLIED });
    expect(JSON.parse(textarea().value)).toEqual(APPLIED);
    const read = [...container.querySelectorAll("dd")].map((item) => item.textContent);
    expect(read).toEqual([
      "nmnist",
      "4 steps × 1.002 ms",
      "train / evaluation",
      APPLIED.digests.manifest,
      APPLIED.digests.encoder,
    ]);
  });

  it("marks an edit as an unapplied draft", async () => {
    await render();
    onDirtyChange.mockClear();
    await type("{");
    expect(textarea().value).toBe("{");
    expect(onDirtyChange).toHaveBeenLastCalledWith(true);
  });

  it("applies a matching declaration and its window", async () => {
    await render();
    await type(JSON.stringify(eventData));
    await press("Apply event input");
    expect(setConfig).toHaveBeenCalledWith("timesteps", 4);
    expect(setConfig).toHaveBeenCalledWith("event_data", APPLIED);
    expect(onDirtyChange).toHaveBeenLastCalledWith(false);
    expect(alertText()).toBeNull();
  });

  it("says why text that is not JSON cannot be applied, and keeps it a draft", async () => {
    await render();
    await type("not json");
    await press("Apply event input");
    expect(alertText()).not.toBeNull();
    expect(setConfig).not.toHaveBeenCalled();
    expect(onDirtyChange).toHaveBeenLastCalledWith(true);
  });

  it.each([
    ["a number", 3],
    ["null", null],
    ["no encoder", { manifest: {} }],
    ["a null encoder", { encoder: null }],
    ["no window", { encoder: {} }],
    ["a window in text", { encoder: { n_steps: "4" } }],
    ["a fractional window", { encoder: { n_steps: 2.5 } }],
    ["an empty window", { encoder: { n_steps: 0 } }],
  ])("refuses a declaration with %s in place of a positive integer window", async (_case, value) => {
    await render();
    await type(JSON.stringify(value));
    await press("Apply event input");
    expect(alertText()).toBe("Event input needs a positive integer encoder window");
    expect(setConfig).not.toHaveBeenCalled();
  });

  it("refuses a declaration made for another dataset with the reader's reason", async () => {
    await render({ ...BASE, dataset: "shd" });
    await type(JSON.stringify(eventData));
    await press("Apply event input");
    expect(alertText()).toBe("Event training input is missing or incompatible with this configuration");
    expect(setConfig).not.toHaveBeenCalled();
  });

  it("names a failure that is not an instance of Error with a fixed message", async () => {
    await render();
    await type(JSON.stringify(eventData));
    reader.failure = FOREIGN_FAILURE;
    await press("Apply event input");
    expect(alertText()).toBe("Event input could not be read");
  });

  it("loads a selected file as a draft and empties the picker", async () => {
    await render();
    const pending = pendingFile();
    onDirtyChange.mockClear();
    const input = await select([pending.file]);
    expect(onDirtyChange).toHaveBeenLastCalledWith(true);
    expect(input.value).toBe("");
    await settle(() => { pending.resolve('{"draft": true}'); });
    expect(textarea().value).toBe('{"draft": true}');
    expect(setConfig).not.toHaveBeenCalled();
  });

  it("ignores an empty selection", async () => {
    await render();
    onDirtyChange.mockClear();
    await select([]);
    expect(onDirtyChange).not.toHaveBeenCalled();
  });

  it("drops a file read that a later edit superseded", async () => {
    await render();
    const pending = pendingFile();
    await select([pending.file]);
    await type("typed later");
    await settle(() => { pending.resolve("read earlier"); });
    expect(textarea().value).toBe("typed later");
  });

  it("says why a file could not be read", async () => {
    await render();
    const failed = pendingFile();
    await select([failed.file]);
    await settle(() => { failed.reject(new Error("permission denied")); });
    expect(alertText()).toBe("permission denied");
    const silent = pendingFile();
    await select([silent.file]);
    await settle(() => { silent.reject(FOREIGN_FAILURE); });
    expect(alertText()).toBe("Event input file could not be read");
  });

  it("keeps quiet about a failed read that a later edit superseded", async () => {
    await render();
    const pending = pendingFile();
    await select([pending.file]);
    await type("typed later");
    await settle(() => { pending.reject(new Error("permission denied")); });
    expect(alertText()).toBeNull();
  });

  it("clears the draft and the applied declaration", async () => {
    await render({ ...BASE, event_data: APPLIED });
    await type("draft");
    await press("Clear event input");
    expect(textarea().value).toBe("");
    expect(setConfig).toHaveBeenCalledWith("event_data", undefined);
    expect(onDirtyChange).toHaveBeenLastCalledWith(false);
  });

  it("follows a declaration applied elsewhere and drops a stale alert", async () => {
    await render();
    await type("not json");
    await press("Apply event input");
    expect(alertText()).not.toBeNull();
    await render({ ...BASE, event_data: APPLIED });
    expect(alertText()).toBeNull();
    expect(JSON.parse(textarea().value)).toEqual(APPLIED);
  });
});
