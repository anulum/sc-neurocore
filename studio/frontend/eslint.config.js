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
 * Every TypeScript source in this package.
 *
 * This used to be a list. It was the incremental scope the documentation lane
 * grew file by file, from 719 declarations of debt to zero, and while it was a
 * list a **new** file that nobody added to it was silently unenforced -- the
 * measurement counted only what the list already covered, so the gap did not
 * appear as a number anywhere. Verified with exactly that candidate: a new
 * undocumented module produced no finding at all.
 *
 * It is a glob now, so the default for a new file is enforcement rather than
 * exemption. Anything genuinely outside is named in `LEGACY_OUTSIDE_SCOPE`
 * with its reason.
 */
const AUDITED = ["**/*.ts", "**/*.tsx"];

/**
 * How much of the frontend is still outside the audited scope.
 *
 * Recorded so the remaining work is readable rather than implied, and so a
 * later sweep can state what it actually reduced.
 */
export const LEGACY_OUTSIDE_SCOPE = {
  /**
   * Undocumented declarations outside the audited scope, as **ESLint** counts
   * them — the tool that enforces the rule, run over the whole frontend with
   * `eslint.measure.js`.
   *
   * The first figure recorded here was 719, taken with a regex over source
   * lines. The owner directive of 2026-09-06 is explicit that a heuristic may
   * size a problem but may not become a ceiling, and this is why: re-taken
   * with the real tool the same surface reported **1297**. The heuristic
   * understated the debt by nearly half, and had it been used as a ratchet it
   * would have enforced the estimate rather than the surface.
   *
   * Every figure recorded here was a re-measurement, never a subtraction: each
   * sweep grew `AUDITED` and the remainder was re-taken with the tool
   * afterwards rather than reduced on paper. The sequence, each figure taken
   * over the tree that carried it: 1297, 848, 547, 413, 312, 252, 161, 81, 69,
   * **0**.
   *
   * **Zero, as of 2026-09-07**, and the scope is now a glob rather than a
   * list: every `.ts` and `.tsx` file in this package is enforced at
   * strictTypeChecked + stylisticTypeChecked with `jsdoc/require-jsdoc` over
   * every function-like declaration. The three files that are not TypeScript
   * are named in `unlintableFiles` below; nothing else is excluded.
   *
   * That change is the more important half. While the scope was a list, a new
   * file nobody added to it was silently unenforced and produced no count
   * anywhere -- verified with exactly that candidate, a new undocumented
   * module that yielded no finding at all. With a glob the default for a new
   * file is enforcement, and the same candidate is now reported.
   */
  declarations: 0,
  /**
   * Files carrying at least one of those declarations.
   *
   * Recorded as 190 when the figure was first taken, which counted files with
   * *any* finding from the measurement config rather than files with the rule
   * this figure is about. Re-counted by the stated method it was **176**. The
   * declaration count did not move; only the file count was measuring
   * something other than what it said.
   */
  files: 0,
  /**
   * The files the measurement config reads that `AUDITED` does not list, and
   * why each is out.
   *
   * Recorded because a scope of zero is only meaningful alongside what it
   * excludes. None of these is TypeScript: two are the lint configuration
   * itself and one is a plain build script. ESLint reads all three under the
   * base configuration; the strict profile applies to `.ts` and `.tsx` only,
   * which includes `src/vite-env.d.ts`.
   */
  unlintableFiles: {
    "eslint.config.js": "JavaScript: the configuration that defines the scope",
    "eslint.measure.js": "JavaScript: the configuration that measures the scope",
    "scripts/verify-federation-build.mjs": "JavaScript, not in any tsconfig",
  },
  measuredBy:
    "eslint 10.10.0 + eslint-plugin-jsdoc 64.3.6, " +
    "`npx eslint . --config eslint.measure.js -f json`, jsdoc/require-jsdoc",
  measuredOn: "2026-09-07",
  /**
   * The commit the measurement was taken over, plus this file's own change.
   *
   * A scope figure can only be taken from the tree that carries the scope, so
   * it is measured before the commit that records it exists. What is pinned
   * is the parent; the commit this figure lands in adds files to `AUDITED`
   * and nothing else that ESLint reads.
   */
  measuredOnSourceSha: "79339c1fe759a0abad2de466220c77c25f455e25",
  nodeVersion: "v22.23.1",
  typescriptVersion: "5.8.3",
  /**
   * The strictness step that used to be recorded here, now taken.
   *
   * `noUncheckedIndexedAccess` stops an index lookup pretending it always
   * finds something. It reported 198 errors when first measured and 218 by the
   * time the audited scope had grown; it is now **on** in `tsconfig.json` and
   * reports zero. Every site was narrowed, given a stated fallback, or read
   * through `src/arrayAt.ts`, which fails with the index and the length rather
   * than asserting the element exists. No `!` was used and no rule was
   * relaxed, so this number is not coming back.
   */
  noUncheckedIndexedAccessErrors: 0,
  /**
   * The `e2e` suite is fully audited as of 2026-09-07; nothing remains here.
   *
   * Closing it removed 2 454 lines of duplication: the six admin-operator
   * specs each carried an identical 500-line copy of the same mocked Studio,
   * and the unused fixtures the strict profile reported were a symptom of six
   * copies each using a different subset. They share
   * `e2e/adminOperatorHarness.ts` now.
   */
  e2eOutsideScope: {},
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
      //
      // The contexts are stated because the rule's own default covers
      // functions and classes only. Without them `/** */` on an interface
      // satisfied `require-jsdoc` and nothing else looked at it — the gate
      // accepted an empty docblock, which is the padding this scope exists to
      // refuse. Verified against that exact candidate.
      "jsdoc/require-description": [
        "error",
        {
          contexts: [
            "ArrowFunctionExpression",
            "ClassDeclaration",
            "ClassExpression",
            "FunctionDeclaration",
            "FunctionExpression",
            "MethodDefinition",
            "TSEnumDeclaration",
            "TSInterfaceDeclaration",
            "TSTypeAliasDeclaration",
          ],
        },
      ],
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
      // A parameter that exists only to satisfy a signature is not dead code,
      // and the leading underscore is how TypeScript codebases say so. This
      // recognises that convention and nothing else: a parameter without the
      // underscore is still reported, and so is every unused local and import.
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          args: "after-used",
          argsIgnorePattern: "^_",
          caughtErrors: "all",
          ignoreRestSiblings: false,
          varsIgnorePattern: "^$",
        },
      ],
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
    files: ["**/*.test.ts", "**/*.test.tsx", "e2e/**/*.ts"],
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
