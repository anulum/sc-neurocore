// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — the keyboard help overlay and the keys it must leave to text controls
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import KeyboardHelp from "./KeyboardHelp";

let root: Root | undefined;
let container: HTMLDivElement;

beforeEach(async () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => { root?.render(<KeyboardHelp />); await Promise.resolve(); });
});

afterEach(async () => {
  await act(async () => { root?.unmount(); await Promise.resolve(); });
  document.body.replaceChildren();
  vi.unstubAllGlobals();
});

/**
 * Press a key on a target, as the browser delivers it, and report whether the help took it.
 *
 * @param key - The key's value.
 * @param target - Where focus is.
 * @returns Whether the key's default action was prevented.
 */
async function press(key: string, target: Element = document.body): Promise<boolean> {
  const event = new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true });
  await act(async () => { target.dispatchEvent(event); await Promise.resolve(); });
  return event.defaultPrevented;
}

/**
 * Add a control to the page, outside the help.
 *
 * @param markup - The control's HTML, with the focused element marked ``data-focus``.
 * @returns The marked element.
 */
function control(markup: string): Element {
  const host = document.createElement("div");
  host.innerHTML = markup;
  document.body.append(host);
  const target = host.querySelector("[data-focus]");
  if (target === null) throw new Error("no element is marked data-focus");
  return target;
}

/** The help dialog, or null while it is closed. */
function help(): HTMLElement | null {
  return document.querySelector<HTMLElement>('[role="dialog"][aria-labelledby="keyboard-help-title"]');
}

describe("KeyboardHelp", () => {
  it("opens and closes on ? pressed outside a text control, and takes the key", async () => {
    expect(help()).toBeNull();
    expect(await press("?")).toBe(true);
    expect(help()).not.toBeNull();
    expect(await press("?")).toBe(true);
    expect(help()).toBeNull();
  });

  it("opens on / too, which is ? without Shift on many layouts", async () => {
    expect(await press("/")).toBe(true);
    expect(help()).not.toBeNull();
  });

  it("still opens with focus on a button or a link, which type no text", async () => {
    expect(await press("?", control("<button data-focus>Run</button>"))).toBe(true);
    expect(help()).not.toBeNull();
    await press("Escape");
    expect(await press("?", control('<a data-focus href="/docs">Manual</a>'))).toBe(true);
    expect(help()).not.toBeNull();
  });

  it.each([
    ['<input data-focus aria-label="Search models">', "a search field"],
    ["<textarea data-focus></textarea>", "a text area"],
    // Monaco in Chromium focuses a div with role textbox that receives text through EditContext.
    ['<div class="monaco-editor"><div class="native-edit-context" role="textbox" tabindex="0" data-focus></div></div>', "a code editor's EditContext element"],
    ['<div role="textbox" tabindex="0"><span data-focus>dv</span></div>', "content inside a textbox"],
    ['<div role="searchbox" tabindex="0" data-focus></div>', "a searchbox"],
    ['<div role="combobox" tabindex="0" data-focus></div>', "a combobox"],
    ['<div role="spinbutton" tabindex="0" data-focus></div>', "a spinbutton"],
    ["<select data-focus><option>synthetic</option></select>", "a select"],
    ['<div contenteditable="true" data-focus></div>', "an editable region"],
  ])("leaves / and ? to %s, so they are typed there", async (markup) => {
    const target = control(markup);
    expect(await press("/", target)).toBe(false);
    expect(await press("?", target)).toBe(false);
    expect(help()).toBeNull();
  });

  it("treats content marked not editable as outside a text control", async () => {
    expect(await press("?", control('<div contenteditable="false" data-focus>Read-only</div>'))).toBe(true);
    expect(help()).not.toBeNull();
  });

  it("closes on Escape wherever focus is, inside the help or in a text control", async () => {
    await press("?");
    const panel = help();
    if (panel === null) throw new Error("the help did not open");
    await press("Escape", panel);
    expect(help()).toBeNull();
    await press("?");
    expect(await press("Escape", control('<div role="textbox" tabindex="0" data-focus></div>'))).toBe(false);
    expect(help()).toBeNull();
  });

  it("is a labelled modal dialog that lists every shortcut", async () => {
    await press("?");
    const panel = help();
    expect(panel?.getAttribute("aria-modal")).toBe("true");
    expect(document.getElementById("keyboard-help-title")?.textContent.trim()).toBe("Keyboard shortcuts");
    expect(panel?.querySelectorAll("kbd").length ?? 0).toBeGreaterThanOrEqual(12);
    expect(panel?.textContent).toContain("Toggle this help");
  });

  it("closes when the backdrop is clicked, and not when the panel is", async () => {
    await press("?");
    await act(async () => { help()?.click(); await Promise.resolve(); });
    expect(help()).not.toBeNull();
    const backdrop = help()?.parentElement;
    if (!(backdrop instanceof HTMLElement)) throw new Error("the help has no backdrop");
    await act(async () => { backdrop.click(); await Promise.resolve(); });
    expect(help()).toBeNull();
  });

  it("ignores other keys", async () => {
    expect(await press("a")).toBe(false);
    expect(await press("Enter")).toBe(false);
    expect(help()).toBeNull();
  });

  it("stops listening when it is removed", async () => {
    await act(async () => { root?.unmount(); await Promise.resolve(); });
    root = undefined;
    expect(await press("?")).toBe(false);
  });
});
