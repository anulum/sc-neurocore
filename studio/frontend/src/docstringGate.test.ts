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
import { describe, expect, it } from "vitest";

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

/**
 * Lint one candidate as if it were an audited file.
 *
 * The candidate is linted under the path of a file the policy already covers,
 * so it is judged by the same configuration as the rest of the audited scope.
 */
let shared: ESLint | undefined;

/**
 * Lint one candidate under the audited configuration and return its rule ids.
 */
async function lint(source: string): Promise<string[]> {
  shared ??= new ESLint({ cwd: new URL("..", import.meta.url).pathname });
  const results = await shared.lintText(source, {
    filePath: new URL("./studioGraphTable.ts", import.meta.url).pathname,
  });
  return results.flatMap((result) => result.messages.map((message) => message.ruleId ?? "unknown"));
}

describe("the documentation gate", () => {
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
      const rules = await lint(`/**
 * @param value - The number to advance.
 * @returns That number plus one.
 */
export function unlabelled(value: number): number {
  return value + 1;
}
`);

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
      const rules = await lint(`/** */
export interface Carrier {
  value: number;
}
`);

      expect(rules).toContain("jsdoc/require-description");
    },
  );

  it(
    "rejects an empty docblock on a type alias for the same reason",
    { timeout: TIMEOUT_MS },
    async () => {
      const rules = await lint(`/** */
export type Carried = number;
`);

      expect(rules).toContain("jsdoc/require-description");
    },
  );
});
