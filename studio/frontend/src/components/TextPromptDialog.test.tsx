// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — the modal text prompt that replaced window.prompt
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import TextPromptDialog, { type TextPromptRequest } from "./TextPromptDialog";

let root: Root | undefined;
let container: HTMLDivElement;
const onClose = vi.fn();
const onSubmit = vi.fn();

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  onClose.mockClear();
  onSubmit.mockClear();
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
 * A request to save something under a name.
 *
 * @param overrides - Fields that differ.
 * @returns The request.
 */
function request(overrides: Partial<TextPromptRequest> = {}): TextPromptRequest {
  return { title: "Save project", label: "Project name", confirmLabel: "Save", onSubmit, ...overrides };
}

/**
 * Render the dialog.
 *
 * @param pending - The request, or none.
 * @param key - The request key.
 */
async function render(pending: TextPromptRequest | null, key = 1): Promise<void> {
  await act(async () => {
    root?.render(<TextPromptDialog request={pending} requestKey={key} onClose={onClose} />);
    await Promise.resolve();
  });
}

/** The dialog element. */
function dialog(): HTMLDialogElement {
  const element = container.querySelector("dialog");
  if (element === null) throw new Error("no dialog");
  return element;
}

/**
 * Type into the open field through the native setter React listens for.
 *
 * @param text - What to type.
 */
async function type(text: string): Promise<void> {
  const field = dialog().querySelector<HTMLInputElement | HTMLTextAreaElement>("input, textarea");
  if (field === null) throw new Error("no field");
  const prototype = field instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  await act(async () => {
    Object.getOwnPropertyDescriptor(prototype, "value")?.set?.call(field, text);
    field.dispatchEvent(new Event("input", { bubbles: true }));
    await Promise.resolve();
  });
}

/**
 * A button of the prompt by its text.
 *
 * @param text - The label.
 * @returns The button.
 */
function button(text: string): HTMLButtonElement {
  const found = [...dialog().querySelectorAll("button")].find((item) => item.textContent === text);
  if (found === undefined) throw new Error(`no button reads ${text}`);
  return found;
}

describe("TextPromptDialog", () => {
  it("stays closed and empty without a request", async () => {
    await render(null);
    expect(dialog().open).toBe(false);
    expect(dialog().querySelector("form")).toBeNull();
    expect(dialog().getAttribute("aria-labelledby")).toBe("text-prompt-title");
  });

  it("opens as a modal named by its heading, with a labelled single-line field", async () => {
    await render(request());
    expect(dialog().open).toBe(true);
    expect(dialog().querySelector("#text-prompt-title")?.textContent).toBe("Save project");
    expect(dialog().querySelector(".text-prompt-field span")?.textContent).toBe("Project name");
    expect(dialog().querySelector("label input[type=text]")).not.toBeNull();
    expect(dialog().querySelector("textarea")).toBeNull();
  });

  it("offers a multi-line field for a pasted trace", async () => {
    await render(request({ title: "Import trace", multiline: true }));
    expect(dialog().querySelector("label textarea")?.getAttribute("rows")).toBe("8");
    await type("  1.0\n2.0  ");
    await act(async () => { button("Save").click(); await Promise.resolve(); });
    expect(onSubmit).toHaveBeenCalledWith("1.0\n2.0");
  });

  it("refuses an empty or blank answer, then submits the trimmed one and closes", async () => {
    await render(request());
    expect(button("Save").disabled).toBe(true);
    await type("   ");
    expect(button("Save").disabled).toBe(true);
    await act(async () => {
      dialog().querySelector("form")?.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
      await Promise.resolve();
    });
    expect(onSubmit).not.toHaveBeenCalled();
    await type("  my project ");
    expect(button("Save").disabled).toBe(false);
    await act(async () => { button("Save").click(); await Promise.resolve(); });
    expect(onSubmit).toHaveBeenCalledWith("my project");
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("closes without an answer on Cancel and on the dialog's own close (Escape)", async () => {
    await render(request());
    await act(async () => { button("Cancel").click(); await Promise.resolve(); });
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(onSubmit).not.toHaveBeenCalled();
    await act(async () => { dialog().dispatchEvent(new Event("close")); await Promise.resolve(); });
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it("closes when the request goes away, and stays open across a re-render of the same request", async () => {
    const pending = request();
    await render(pending);
    await render(pending);
    expect(dialog().open).toBe(true);
    await render(null);
    expect(dialog().open).toBe(false);
  });

  it("starts every new request with an empty field", async () => {
    await render(request(), 1);
    await type("first");
    await render(request(), 2);
    expect(dialog().querySelector<HTMLInputElement>("input")?.value).toBe("");
  });
});
