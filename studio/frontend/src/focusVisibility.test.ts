// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

const SRC = new URL(".", import.meta.url).pathname;
const CSS = readFileSync(join(SRC, "index.css"), "utf8");

/** Every `.ts`/`.tsx` source under the frontend tree, excluding tests. */
function sourceFiles(dir: string): string[] {
  const found: string[] = [];
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) {
      if (entry !== "node_modules") found.push(...sourceFiles(path));
    } else if (/\.tsx?$/.test(entry) && !/\.test\.tsx?$/.test(entry)) {
      found.push(path);
    }
  }
  return found;
}

/** The kinds of control whose own outline the stylesheet removes and replaces with a ring. */
const RINGED = [
  "input:focus-visible",
  "textarea:focus-visible",
  "select:focus-visible",
  "summary:focus-visible",
  "a:focus-visible",
  "button:focus-visible",
];

/**
 * Return the stylesheet's forced-colours block and where it starts.
 *
 * @returns The block's body and its offset in the stylesheet, or null when there is none.
 */
function forcedColoursBlock(): { body: string; start: number } | null {
  const start = CSS.indexOf("@media (forced-colors: active)");
  if (start < 0) return null;
  const open = CSS.indexOf("{", start);
  let depth = 0;
  for (let index = open; index < CSS.length; index += 1) {
    if (CSS[index] === "{") depth += 1;
    if (CSS[index] === "}") {
      depth -= 1;
      if (depth === 0) return { body: CSS.slice(open + 1, index), start };
    }
  }
  return null;
}

describe("focus visibility under forced colours", () => {
  it("draws an outline in the system highlight colour, since forced colours drop every box-shadow", () => {
    // WCAG 2.4.7, measured with Chromium's forced-colors emulation: a focused
    // control kept `outline-style: none` and lost its box-shadow ring, so the
    // keyboard's place had no visible indicator at all.
    const block = forcedColoursBlock();
    if (block === null) throw new Error("The stylesheet has no forced-colours block");
    for (const selector of RINGED) {
      expect(block.body).toContain(selector);
    }
    expect(block.body).toMatch(/outline:\s*2px solid Highlight;/);
  });

  it("comes after every rule that removes an outline, so the cascade cannot take it back", () => {
    // `select` and the range slider remove their outline outright, with the same
    // specificity as the focus rules; only a later rule wins over them.
    const block = forcedColoursBlock();
    if (block === null) throw new Error("The stylesheet has no forced-colours block");
    expect(block.start).toBeGreaterThan(CSS.lastIndexOf("outline: none"));
  });
});

describe("focus visibility", () => {
  it("puts a ring on the kinds whose outline was removed", () => {
    // WCAG 2.4.7, measured in a browser before this was written: with no rule,
    // a text input and a range slider computed `outlineStyle: none` and
    // `boxShadow: none` on a real Tab. `select` already replaced its own; these
    // two removed theirs and put nothing back. The rule now names every input
    // type at once (text, number, password, range, file, checkbox, radio), the
    // other form controls, and the two keyboard targets that are not controls.
    for (const selector of [
      "input:focus-visible",
      "textarea:focus-visible",
      "select:focus-visible",
      "summary:focus-visible",
      "a:focus-visible",
      "button:focus-visible",
    ]) {
      expect(CSS).toContain(selector);
    }
    expect(CSS).toMatch(/:focus-visible[\s\S]{0,400}box-shadow:\s*0 0 0 2px/);
  });

  it("draws the ring in the focus-ring colour, framed by a gap, so it shows on every surface", () => {
    // The gap of --bg-primary keeps the ring apart from a control filled with
    // the accent; src/paletteContrast.test.ts holds both colours to 3:1.
    expect(CSS).toMatch(/--focus-ring:\s*var\(--accent\);/);
    expect(CSS).toMatch(
      /button:focus-visible \{[^}]*box-shadow:\s*0 0 0 2px var\(--bg-primary\), 0 0 0 4px var\(--focus-ring\);/,
    );
  });

  it("uses :focus-visible, so a pointer click does not ring what it clicked", () => {
    // `:focus` would ring a control the reader is already looking at; the
    // keyboard arrival is the one that needs saying.
    expect(CSS).toContain(":focus-visible");
  });

  it("no component removes an outline inline, where no stylesheet can reach it", () => {
    // An inline style beats the stylesheet. The model browser's search field
    // carried `outline: "none"` inline and so kept no ring at all, whatever
    // `index.css` said.
    const offenders = sourceFiles(SRC)
      .filter((path) => /outline:\s*["']none["']/.test(readFileSync(path, "utf8")))
      .map((path) => path.slice(SRC.length));

    expect(offenders).toEqual([]);
  });
});
