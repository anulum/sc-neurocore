// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Native keyboard control ownership tests
// @vitest-environment happy-dom

import { expect, it } from "vitest";
import { isTextEntryTarget, isWorkbenchShortcutTarget } from "./workbenchKeyboard";

it.each([
  "<button data-focus>Open catalogue</button>",
  "<button><span data-focus>Nested label</span></button>",
  "<input data-focus>",
  "<textarea data-focus></textarea>",
  "<select data-focus><option>FPGA</option></select>",
  '<a data-focus href="/docs">Manual</a>',
  '<div contenteditable><span data-focus>Editable</span></div>',
  '<div contenteditable="plaintext-only" data-focus>Editable</div>',
  '<dialog open><div data-focus>Modal content</div></dialog>',
  '<div role="dialog"><div data-focus>Modal content</div></div>',
  ...["button", "textbox", "combobox", "listbox", "slider", "spinbutton", "menu", "checkbox", "radio", "switch", "link"]
    .map((role) => `<div role="${role}"><span data-focus>Control</span></div>`),
])("keeps global actions out of the real control %s", (markup) => {
  const host = document.createElement("div");
  host.innerHTML = markup;
  const target = host.querySelector("[data-focus]");
  expect(target).not.toBeNull();
  expect(isWorkbenchShortcutTarget(target)).toBe(false);
});

it("retains body shortcuts and distinguishes explicitly non-editable content", () => {
  const host = document.createElement("div");
  host.innerHTML = '<div contenteditable="FALSE">Read-only</div><a>No link destination</a>';
  expect(isWorkbenchShortcutTarget(host.firstElementChild)).toBe(true);
  expect(isWorkbenchShortcutTarget(host.lastElementChild)).toBe(true);
  expect(isWorkbenchShortcutTarget(document.body)).toBe(true);
  expect(isWorkbenchShortcutTarget(window)).toBe(true);
  expect(isWorkbenchShortcutTarget(null)).toBe(true);
});

it.each([
  ["<input data-focus>", true],
  ["<textarea data-focus></textarea>", true],
  ["<select data-focus><option>synthetic</option></select>", true],
  ['<div class="native-edit-context" role="textbox" tabindex="0" data-focus></div>', true],
  ['<div role="textbox"><span data-focus>dv</span></div>', true],
  ['<div role="searchbox" data-focus></div>', true],
  ['<div role="combobox" data-focus></div>', true],
  ['<div role="spinbutton" data-focus></div>', true],
  ['<div contenteditable="plaintext-only" data-focus></div>', true],
  ["<button data-focus>Run</button>", false],
  ['<a data-focus href="/docs">Manual</a>', false],
  ['<div role="slider" data-focus></div>', false],
  ['<div contenteditable="false" data-focus>Read-only</div>', false],
  ["<p data-focus>Text</p>", false],
])("says whether %s takes typed characters (%s)", (markup, expected) => {
  const host = document.createElement("div");
  host.innerHTML = markup;
  expect(isTextEntryTarget(host.querySelector("[data-focus]"))).toBe(expected);
});

it("treats a target that is not an element as taking no characters", () => {
  expect(isTextEntryTarget(null)).toBe(false);
  expect(isTextEntryTarget(window)).toBe(false);
  expect(isTextEntryTarget(document)).toBe(false);
});
