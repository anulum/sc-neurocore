// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Scoped strict TypeScript lint and docstring policy

/**
 * Every function this repository maintains carries a docblock, in every
 * language it is written in — and the check is done by the strictest standard
 * tool for that language, not by something hand-rolled here.
 *
 * Python already had this: ruff's `D` rules (pydocstyle, numpy convention) run
 * against the audited file list in `docs/docstring_policy.toml`, which grows
 * package by package until `D` can be promoted to the global select. That
 * design is deliberate and it is copied here rather than reinvented.
 *
 * TypeScript had no linter at all. It now has `typescript-eslint` at
 * **strictTypeChecked + stylisticTypeChecked** and `eslint-plugin-jsdoc` at its
 * TypeScript error preset, with `require-jsdoc` extended to every function-like
 * declaration. Measured when this was introduced, the frontend carried **719
 * undocumented declarations across 151 files**, so the same incremental scope
 * applies: `AUDITED` lists the files whose whole surface has been documented
 * and which are enforced completely; everything else is not yet linted, and
 * the number left is written down rather than hidden.
 *
 * A **new** file belongs in `AUDITED`. That is the point: new code is
 * documented and strictly typed by default instead of joining the backlog.
 */

import js from "@eslint/js";
import jsdoc from "eslint-plugin-jsdoc";
import tseslint from "typescript-eslint";

/**
 * Files whose entire surface has been audited, documented and made safe to
 * enforce. Sorted, so an addition is one reviewable line of diff.
 */
const AUDITED = [
  "e2e/network-canvas-live.spec.ts",
  "playwright.export.config.ts",
  "playwright.graph.config.ts",
  "src/components/NetworkGraphTable.test.tsx",
  "src/components/NetworkGraphTable.tsx",
  "src/components/PopulationEditor.test.tsx",
  "src/components/PopulationEditor.tsx",
  "src/components/ProjectionEditor.test.tsx",
  "src/components/ProjectionEditor.tsx",
  "src/evidenceSeal.test.ts",
  "src/evidenceSeal.ts",
  "src/simulationRaw.test.ts",
  "src/simulationRaw.ts",
  "src/stores/studioStoreActions.test.ts",
  "src/studioGraphHistory.test.ts",
  "src/studioGraphHistory.ts",
  "src/studioGraphTable.test.ts",
  "src/studioGraphTable.ts",
  "src/studioGraphValidation.test.ts",
  "src/studioGraphValidation.ts",
  "src/studioNodeDeletion.test.ts",
  "src/studioPopulationEditor.test.ts",
  "src/studioPopulationEditor.ts",
  "src/studioProjectionEditor.test.ts",
  "src/studioProjectionEditor.ts",
];

/**
 * How much of the frontend is still outside the audited scope.
 *
 * Recorded so the remaining work is readable rather than implied, and so a
 * later sweep can state what it actually reduced.
 */
export const LEGACY_OUTSIDE_SCOPE = {
  declarations: 719,
  files: 151,
  measuredOn: "2026-09-06",
  /**
   * The next strictness step, measured rather than guessed at.
   *
   * `noUncheckedIndexedAccess` is the compiler option that stops an index
   * lookup pretending it always finds something. Turning it on today reports
   * **198** errors across the frontend, so it is recorded here as the next
   * increment instead of being switched on and suppressed.
   */
  noUncheckedIndexedAccessErrors: 198,
  /**
   * The file to audit next, and what it will cost.
   *
   * `e2e/experiment-export-live.spec.ts` parses untyped JSON from the server
   * and carries **25** `no-unsafe-*` findings as a result. Typing those
   * payloads is worth doing and is its own piece of work; listing the file
   * before it is done would mean either a red gate or a suppression, and both
   * are worse than saying what is left.
   */
  nextToAudit: { file: "e2e/experiment-export-live.spec.ts", findings: 25 },
};

export default tseslint.config(
  {
    ignores: ["dist/**", "node_modules/**", "test-results/**", "playwright-report/**", "@mf-types/**"],
  },
  {
    files: AUDITED,
    extends: [
      js.configs.recommended,
      ...tseslint.configs.strictTypeChecked,
      ...tseslint.configs.stylisticTypeChecked,
      jsdoc.configs["flat/recommended-typescript-error"],
    ],
    languageOptions: {
      parserOptions: {
        // Type-aware rules need a project that contains the file. `src` is
        // covered by tsconfig.json; the audited e2e specs and Playwright
        // configs live outside it and are covered by tsconfig.audited.json.
        project: ["./tsconfig.json", "./tsconfig.audited.json"],
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      // Every function a reader of the module meets carries a docblock, and a
      // docblock that says nothing is not one.
      "jsdoc/require-description": "error",
      "jsdoc/require-jsdoc": [
        "error",
        {
          contexts: [
            // An exported arrow is part of the module's surface and is
            // documented like any other function. A closure inside a function
            // body is a detail of that function, whose own docblock covers it.
            "ExportNamedDeclaration > VariableDeclaration > VariableDeclarator > ArrowFunctionExpression",
            "TSInterfaceDeclaration",
            "TSTypeAliasDeclaration",
            "TSEnumDeclaration",
          ],
          publicOnly: false,
          require: {
            ArrowFunctionExpression: false,
            ClassDeclaration: true,
            ClassExpression: true,
            FunctionDeclaration: true,
            FunctionExpression: false,
            MethodDefinition: true,
          },
        },
      ],
      // The types are in the signature; repeating them in the docblock is a
      // second copy free to drift from the first.
      "jsdoc/require-param-type": "off",
      "jsdoc/require-returns-type": "off",
      // One blank line between the prose and the tag block, so a long
      // description does not run into its own parameter table.
      "jsdoc/tag-lines": ["error", "any", { startLines: 1 }],
      // A React component takes one destructured props object whose fields are
      // documented on the props interface, which `require-jsdoc` already
      // demands. Repeating each field as `@param root0.thing` would be a
      // second copy of that documentation, free to drift from the first.
      "jsdoc/check-param-names": ["error", { checkDestructured: false }],
      "jsdoc/require-param": [
        "error",
        { checkDestructured: false, checkDestructuredRoots: false },
      ],
      // The hazard this rule exists for is `${someObject}` rendering as
      // "[object Object]", or a nullish value rendering as "null". A number
      // has exactly one text form and no surprise, so it is admitted; every
      // other non-string still has to be converted deliberately.
      "@typescript-eslint/restrict-template-expressions": [
        "error",
        { allowNumber: true },
      ],
    },
  },
  {
    // A test's fixtures and cases are documented at the file and case level;
    // requiring a parameter table on every arrow callback inside a `describe`
    // would produce ceremony, not description.
    files: AUDITED.filter(
      (file) => file.includes(".test.") || file.startsWith("e2e/"),
    ),
    plugins: { jsdoc },
    rules: {
      "jsdoc/require-jsdoc": [
        "error",
        {
          publicOnly: false,
          require: {
            ArrowFunctionExpression: false,
            ClassDeclaration: true,
            FunctionDeclaration: true,
            FunctionExpression: false,
            MethodDefinition: false,
          },
        },
      ],
      "jsdoc/require-param": "off",
      "jsdoc/require-returns": "off",
      // React's `act` takes an async callback whether or not the body awaits
      // anything; the async form is its contract, not a style slip.
      "@typescript-eslint/require-await": "off",
    },
  },
);
