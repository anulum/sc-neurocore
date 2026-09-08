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

describe("focus visibility", () => {
  it("puts a ring on the kinds whose outline was removed", () => {
    // WCAG 2.4.7, measured in a browser before this was written: with no rule,
    // a text input and a range slider computed `outlineStyle: none` and
    // `boxShadow: none` on a real Tab. `select` already replaced its own; these
    // two removed theirs and put nothing back.
    expect(CSS).toContain('input[type="range"]:focus-visible');
    expect(CSS).toContain('input[type="text"]:focus-visible');
    expect(CSS).toContain("button:focus-visible");
    expect(CSS).toMatch(/:focus-visible[\s\S]{0,400}box-shadow:\s*0 0 0 2px/);
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
