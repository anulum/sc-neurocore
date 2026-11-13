// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Workbench keyboard ownership

const INTERACTIVE = [
  "input", "textarea", "button", "select", "a[href]",
  "dialog[open]",
  '[role="dialog"]', '[role="button"]', '[role="textbox"]', '[role="combobox"]',
  '[role="listbox"]', '[role="slider"]', '[role="spinbutton"]', '[role="menu"]',
  '[role="checkbox"]', '[role="radio"]', '[role="switch"]', '[role="link"]',
].join(",");

/**
 * Leave keys with their focused control or modal before applying workbench shortcuts.
 *
 * @param target - The real keyboard event target, including nested control content.
 * @returns Whether global simulation/view shortcuts may handle this target.
 */
export function isWorkbenchShortcutTarget(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return true;
  if (target.closest(INTERACTIVE) !== null) return false;
  return !isEditableRegion(target);
}

/** Controls that take typed characters, by element or by role. */
const TEXT_ENTRY = [
  "input", "textarea", "select",
  '[role="textbox"]', '[role="searchbox"]', '[role="combobox"]', '[role="spinbutton"]',
].join(",");

/**
 * Whether the target is inside content marked editable, and not marked otherwise.
 *
 * @param target - The keyboard event target.
 * @returns Whether typed characters belong to an editable region around the target.
 */
function isEditableRegion(target: Element): boolean {
  const editable = target.closest("[contenteditable]");
  return editable !== null && String(editable.getAttribute("contenteditable")).toLowerCase() !== "false";
}

/**
 * Whether typed characters belong to the focused control rather than to a character shortcut.
 *
 * A single-character shortcut such as `?` must not fire while a person is typing. Buttons and links
 * type nothing, so a character shortcut stays available on them; text fields, selects, editable
 * regions and anything with a text-entry role keep their characters. That includes a code editor
 * that receives text through EditContext: Monaco in Chromium focuses a `div` with role `textbox`,
 * which is neither an input nor a text area.
 *
 * @param target - The real keyboard event target, including nested control content.
 * @returns Whether the target takes typed characters.
 */
export function isTextEntryTarget(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false;
  return target.closest(TEXT_ENTRY) !== null || isEditableRegion(target);
}
