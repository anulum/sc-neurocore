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
  const editable = target.closest("[contenteditable]");
  return editable === null || String(editable.getAttribute("contenteditable")).toLowerCase() === "false";
}
