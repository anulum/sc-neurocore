// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — tests for the Studio's documentation links

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { DOCS_SITE, publishedDocsUrl } from "./studioDocs";

const REPO = new URL("../../../", import.meta.url).pathname;

/** Every `docs_path` the Studio's Python sources declare for a capability. */
function declaredDocsPaths(): string[] {
  const studio = join(REPO, "src/sc_neurocore/studio");
  const paths = new Set<string>();
  const walk = (dir: string) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) walk(path);
      else if (entry.name.endsWith(".py")) {
        for (const match of readFileSync(path, "utf8").matchAll(/docs_path="([^"]+)"/g)) {
          if (match[1] !== undefined) paths.add(match[1]);
        }
      }
    }
  };
  walk(studio);
  return [...paths].sort();
}

describe("documentation links", () => {
  it("points at the site mkdocs.yml publishes", () => {
    const mkdocs = readFileSync(join(REPO, "mkdocs.yml"), "utf8");
    expect(mkdocs).toMatch(new RegExp(`^site_url: ${DOCS_SITE.replace(/[.]/g, "\\.")}$`, "m"));
  });

  it("maps a page to its MkDocs directory URL", () => {
    expect(publishedDocsUrl("docs/studio/index.md")).toBe(`${DOCS_SITE}studio/`);
    expect(publishedDocsUrl("docs/studio/network-canvas.md")).toBe(`${DOCS_SITE}studio/network-canvas/`);
    expect(publishedDocsUrl("docs/index.md")).toBe(DOCS_SITE);
    expect(publishedDocsUrl("README.md")).toBe(DOCS_SITE);
  });

  it("links every capability to a page that exists in the documentation tree", () => {
    const paths = declaredDocsPaths();
    expect(paths.length).toBeGreaterThan(3);
    expect(paths.filter((path) => !existsSync(join(REPO, path)))).toEqual([]);
    for (const path of paths) expect(publishedDocsUrl(path)).not.toBe(DOCS_SITE);
  });
});
