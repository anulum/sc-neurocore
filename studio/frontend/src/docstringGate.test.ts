// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Proof that the documentation gate fails closed

/**
 * A gate is only a gate if something has been shown to fail it.
 *
 * The owner directive of 2026-09-06 is explicit that an installed tool and a
 * green run are not evidence of enforcement: the gate must be verified with a
 * **known-invalid candidate** as well as a valid one. The failure it exists to
 * prevent is a real one — a documentation workflow elsewhere in the ecosystem
 * was green for a long time while measuring nothing, because building a
 * reference was mistaken for checking that the documentation exists.
 *
 * These cases run the project's own ESLint configuration over candidate source
 * held in this file, so what is proved is the configuration that actually runs
 * in `npm run lint`, not a copy of it that could drift.
 */

import { ESLint } from "eslint";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

// Building a type-aware ESLint program is slow, and one instance is reused
// across the cases so the cost is paid once rather than four times.
const TIMEOUT_MS = 120_000;

/** An exported function with no docblock: the gate must reject this. */
const UNDOCUMENTED = `export function unlabelled(value: number): number {
  return value + 1;
}
`;

/** The same function, documented: the gate must accept this. */
const DOCUMENTED = `/**
 * Return the value one greater than the one given.
 *
 * @param value - The number to advance.
 * @returns That number plus one.
 */
export function unlabelled(value: number): number {
  return value + 1;
}
`;

/** An exported interface with no docblock: the gate must reject this too. */
const UNDOCUMENTED_INTERFACE = `export interface Carrier {
  value: number;
}
`;

/** A docblock whose tags have no description preceding them. */
const DESCRIPTION_MISSING = `/**
 * @param value - The number to advance.
 * @returns That number plus one.
 */
export function unlabelled(value: number): number {
  return value + 1;
}
`;

/** An interface with an empty docblock. */
const EMPTY_INTERFACE = `/** */
export interface Carrier {
  value: number;
}
`;

/** A type alias with an empty docblock. */
const EMPTY_ALIAS = `/** */
export type Carried = number;
`;

/** The parser sees every candidate before its TypeScript project is built. */
const candidates = [
  UNDOCUMENTED,
  DOCUMENTED,
  UNDOCUMENTED_INTERFACE,
  DESCRIPTION_MISSING,
  EMPTY_INTERFACE,
  EMPTY_ALIAS,
];
const candidatePaths = new Map<string, string>();
let candidateDirectory: string | undefined;

beforeAll(async () => {
  const directory = await mkdtemp(fileURLToPath(new URL("./docstring-gate-", import.meta.url)));
  candidateDirectory = directory;
  await Promise.all(
    candidates.map(async (source, index) => {
      const candidate = join(directory, `candidate-${index}.ts`);
      await writeFile(candidate, source);
      candidatePaths.set(source, candidate);
    }),
  );
});

afterAll(async () => {
  if (candidateDirectory !== undefined) {
    await rm(candidateDirectory, { recursive: true, force: true });
  }
});

/** The instance every case shares; building a type-aware program is slow. */
let shared: ESLint | undefined;

/**
 * Lint one candidate under the project's own configuration.
 *
 * @param source - The candidate source.
 * @returns The rule ids that fired, in the order they were reported.
 */
async function lint(source: string): Promise<string[]> {
  const candidate = candidatePaths.get(source);
  if (candidate === undefined) throw new Error("Unregistered documentation gate candidate");
  shared ??= new ESLint({ cwd: fileURLToPath(new URL("..", import.meta.url)) });
  const results = await shared.lintFiles([candidate]);
  return results.flatMap((result) => result.messages.map((message) => message.ruleId ?? "unknown"));
}

/**
 * The severity a calculated configuration gives one rule.
 *
 * The result of `calculateConfigForFile` is read structurally rather than
 * through a cast: it is a plain object from a tool, and the gate that proves
 * other files are honest should be honest itself.
 *
 * @param config - What the configuration resolved to for some path.
 * @param ruleId - The rule to look up.
 * @returns `error`, `warn`, `off`, or `absent` when the rule is not configured
 *   at all -- which is what an unenforced file looks like.
 */
function ruleSeverity(config: unknown, ruleId: string): string {
  if (typeof config !== "object" || config === null || !("rules" in config)) {
    return "absent";
  }
  const { rules } = config;
  if (typeof rules !== "object" || rules === null || !(ruleId in rules)) {
    return "absent";
  }
  const entry: unknown = (rules as Record<string, unknown>)[ruleId];
  const severity: unknown = Array.isArray(entry) ? entry[0] : entry;
  if (severity === 2 || severity === "error") return "error";
  if (severity === 1 || severity === "warn") return "warn";
  return "off";
}

describe("the documentation gate", () => {
  it(
    "covers a file that has never existed, which a list-based scope did not",
    { timeout: TIMEOUT_MS },
    async () => {
      // The scope was a list of audited paths until 2026-09-07, and a new file
      // nobody added to it was silently unenforced: it produced no finding and
      // no count anywhere. The scope is a glob now.
      //
      // This asks the configuration what it would apply to a path that has
      // never existed, rather than linting text under it -- the type-aware
      // parser refuses a path no tsconfig contains, which would prove nothing
      // about the scope either way.
      shared ??= new ESLint({ cwd: new URL("..", import.meta.url).pathname });
      const config: unknown = await shared.calculateConfigForFile(
        new URL("./aModuleNobodyHasWrittenYet.ts", import.meta.url).pathname,
      );

      expect(ruleSeverity(config, "jsdoc/require-jsdoc")).toBe("error");
      expect(ruleSeverity(config, "@typescript-eslint/no-explicit-any")).toBe("error");
    },
  );

  it("rejects an exported function that carries no docblock", { timeout: TIMEOUT_MS }, async () => {
    const rules = await lint(UNDOCUMENTED);

    expect(rules).toContain("jsdoc/require-jsdoc");
  });

  it("accepts the same function once it is documented", { timeout: TIMEOUT_MS }, async () => {
    const rules = await lint(DOCUMENTED);

    expect(rules).toEqual([]);
  });

  it("rejects an exported interface that carries no docblock", { timeout: TIMEOUT_MS }, async () => {
    // A type is part of the surface a reader meets; the gate covers it too.
    const rules = await lint(UNDOCUMENTED_INTERFACE);

    expect(rules).toContain("jsdoc/require-jsdoc");
  });

  it(
    "rejects a docblock with no description, which is decoration not documentation",
    { timeout: TIMEOUT_MS },
    async () => {
      const rules = await lint(DESCRIPTION_MISSING);

      expect(rules).toContain("jsdoc/require-description");
    },
  );

  it(
    "rejects an empty docblock on an interface, which passed until 2026-09-07",
    { timeout: TIMEOUT_MS },
    async () => {
      // `require-description` covers functions and classes by default, so an
      // empty block on an interface satisfied `require-jsdoc` and nothing else
      // looked at it. Four had already reached the audited scope. The contexts
      // are now stated in the configuration, and this is the candidate that
      // proves it.
      const rules = await lint(EMPTY_INTERFACE);

      expect(rules).toContain("jsdoc/require-description");
    },
  );

  it(
    "rejects an empty docblock on a type alias for the same reason",
    { timeout: TIMEOUT_MS },
    async () => {
      const rules = await lint(EMPTY_ALIAS);

      expect(rules).toContain("jsdoc/require-description");
    },
  );
});
