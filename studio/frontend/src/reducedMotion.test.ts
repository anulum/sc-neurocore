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

/**
 * How many inline `transition:` / `animation:` declarations each file holds.
 *
 * Counted per declaration, not per line, so a second one on an existing line
 * cannot hide behind the first — and keyed by file rather than by line, so an
 * unrelated edit above a motion site is not a failure. The first version of
 * this pin carried line numbers and broke when a hook was added elsewhere in
 * `App.tsx`, which is noise, not a finding.
 */
function inlineMotionSites(): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const path of sourceFiles(SRC)) {
    const found = [...readFileSync(path, "utf8").matchAll(/\b(transition|animation):\s*"/g)];
    if (found.length > 0) counts[path.slice(SRC.length)] = found.length;
  }
  return counts;
}

describe("reduced motion", () => {
  const css = readFileSync(join(SRC, "index.css"), "utf8");

  it("neutralises motion when the reader has asked for less of it", () => {
    const block = /@media \(prefers-reduced-motion: reduce\) \{([\s\S]*?)\n\}/.exec(css);
    expect(block, "index.css declares no prefers-reduced-motion block").not.toBeNull();

    const body = block?.[1] ?? "";
    expect(body).toMatch(/animation-duration:\s*0\.01ms\s*!important/);
    expect(body).toMatch(/transition-duration:\s*0\.01ms\s*!important/);
    expect(body).toMatch(/scroll-behavior:\s*auto\s*!important/);
  });

  it("applies to every element, because three motion sites are inline styles", () => {
    // A stylesheet reaches an inline `style` attribute only through
    // `!important`, and only a universal selector reaches all three at once.
    const block = /@media \(prefers-reduced-motion: reduce\) \{\s*([^{]*)\{/.exec(css);
    expect(block?.[1]).toMatch(/\*/);
  });

  it("pins the inline motion sites so a new one is a deliberate choice", () => {
    // Not a style rule: these are the declarations a stylesheet can only reach
    // with `!important`. A new one should be seen, not absorbed silently.
    expect(inlineMotionSites()).toEqual({
      "App.tsx": 1,
      "components/SynthesisDashboard.tsx": 1,
      "components/TrainingMonitor.tsx": 1,
    });
  });
});
