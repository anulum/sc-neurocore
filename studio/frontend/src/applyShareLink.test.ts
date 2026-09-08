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

/** Files that reference a name, excluding the one that declares it. */
function callers(name: string, declaredIn: string): string[] {
  return sourceFiles(SRC)
    .filter((path) => !path.endsWith(declaredIn))
    .filter((path) => readFileSync(path, "utf8").includes(name))
    .map((path) => path.slice(SRC.length))
    .sort();
}

describe("a share link is wired", () => {
  // Wiring only. What a link *asks for* is judged by
  // `studioShareLinkDecision` and tested behaviourally in
  // `shareLinkApplication.test.ts`; asserting its message text here would
  // pin where the implementation lives rather than what it does — which is
  // exactly what broke when that logic moved into its own module.
  it("has a production caller, which is what it lacked", () => {
    // The Share button produced links from the beginning and
    // `readStudioStartupHashState` had no caller outside its own test, so
    // opening one gave the default Studio with no sign the link carried
    // anything. A link handed out and never read is worse than no link.
    expect(callers("readStudioStartupHashState", "studioStartupRuntime.ts")).toContain(
      "stores/studioStoreActions.ts",
    );
  });

  it("is applied from the application, not only defined in the store", () => {
    expect(callers("applyShareLink", "stores/studioStoreActions.ts")).toContain("App.tsx");
  });

  it("waits for the catalogue before deciding a model is unknown", () => {
    // Applying before the models arrive would report every link as naming a
    // model this catalogue does not hold.
    const app = readFileSync(join(SRC, "App.tsx"), "utf8");
    const effect = /if \(modelCount === 0\) return;[\s\S]{0,80}applyShareLink\(\);/.exec(app);

    expect(effect, "the share-link effect does not guard on the catalogue").not.toBeNull();
  });

});
