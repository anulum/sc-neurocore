// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — every browser spec runs under exactly the config that can serve it

import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

const ROOT = new URL("..", import.meta.url).pathname;

/** The quoted spec file names a config's source lists in one array property. */
function listed(config: string, property: "testIgnore" | "testMatch"): string[] {
  const source = readFileSync(join(ROOT, config), "utf8");
  const start = source.indexOf(`${property}:`);
  if (start < 0) return [];
  const rest = source.slice(start);
  const end = rest.startsWith(`${property}: [`) ? rest.indexOf("]") : rest.indexOf("\n");
  return [...rest.slice(0, end).matchAll(/"([\w.-]+\.spec\.ts)"/g)].map((match) => match[1] ?? "");
}

describe("browser spec configuration", () => {
  const specs = readdirSync(join(ROOT, "e2e")).filter((name) => name.endsWith(".spec.ts"));
  const liveSpecs = specs.filter((name) => name.endsWith("-live.spec.ts"));
  const liveConfigs = readdirSync(ROOT).filter(
    (name) => /^playwright\..+\.config\.ts$/.test(name),
  );

  it("keeps every live-backend spec out of the mocked default run", () => {
    // Three live training specs once ran in the mocked run, where no backend
    // can serve them, and failed there on every run.
    const ignored = new Set(listed("playwright.config.ts", "testIgnore"));
    expect(liveSpecs.length).toBeGreaterThan(10);
    expect(liveSpecs.filter((name) => !ignored.has(name))).toEqual([]);
  });

  it("runs every live-backend spec under a config that starts its backend", () => {
    const matched = new Set(liveConfigs.flatMap((config) => listed(config, "testMatch")));
    expect(liveSpecs.filter((name) => !matched.has(name))).toEqual([]);
  });

  it("ignores only specs that exist", () => {
    const present = new Set(specs);
    expect(listed("playwright.config.ts", "testIgnore").filter((name) => !present.has(name)))
      .toEqual([]);
  });
});
