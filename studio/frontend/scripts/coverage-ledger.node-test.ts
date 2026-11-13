// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Frontend coverage ledger: no source file's coverage may fall

/** Exercise the coverage ledger on fixture summaries and through its command, with native Node coverage. */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, test } from "node:test";
import { fileURLToPath } from "node:url";

import {
  LEDGER_SCHEMA,
  LedgerInputError,
  compare,
  main,
  percent,
  readLedger,
  readSummary,
  render,
} from "./coverage-ledger.ts";

const command = fileURLToPath(new URL("./coverage-ledger.ts", import.meta.url));
const temporary: string[] = [];

/** Remove only temporary directories allocated by these cases. */
function cleanup(): void {
  for (const folder of temporary.splice(0)) rmSync(folder, { recursive: true, force: true });
}
afterEach(cleanup);

/**
 * Allocate one owned folder that stands for the frontend folder of a case.
 *
 * @returns The new absolute path.
 */
function folder(): string {
  const path = mkdtempSync(join(tmpdir(), "studio-coverage-ledger-"));
  temporary.push(path);
  return path;
}

/**
 * Build one json-summary entry from counts.
 *
 * @param statements - Covered and total statements.
 * @param branches - Covered and total branches.
 * @returns The entry as Vitest's v8 provider writes it.
 */
function entry(statements: [number, number], branches: [number, number]): unknown {
  const counts = ([covered, total]: [number, number]) => ({
    total,
    covered,
    skipped: 0,
    pct: total === 0 ? 100 : (covered * 100) / total,
  });
  return { lines: counts(statements), statements: counts(statements), functions: counts([1, 1]), branches: counts(branches) };
}

/**
 * Build a summary for files under a root, with the total entry Vitest adds.
 *
 * @param root - The frontend folder.
 * @param files - Relative path to counts.
 * @returns The summary object.
 */
function summary(root: string, files: Record<string, [[number, number], [number, number]]>): unknown {
  const result: Record<string, unknown> = { total: entry([0, 0], [0, 0]) };
  for (const [path, [statements, branches]] of Object.entries(files)) {
    result[join(root, path)] = entry(statements, branches);
  }
  return result;
}

/**
 * Run the command in process and collect what it prints.
 *
 * @param argv - The command's arguments.
 * @param root - The frontend folder.
 * @returns The exit status and the printed lines.
 */
function run(argv: string[], root: string): { status: number; lines: string[] } {
  const lines: string[] = [];
  const status = main(argv, root, (line) => lines.push(line));
  return { status, lines };
}

/**
 * Write a summary and, when given, a ledger into a case folder.
 *
 * @param root - The case folder.
 * @param content - The summary.
 * @param ledger - The ledger's text, or nothing for no ledger.
 * @returns The two paths.
 */
function inputs(root: string, content: unknown, ledger?: string): { summaryPath: string; ledgerPath: string } {
  const summaryPath = join(root, "coverage-summary.json");
  const ledgerPath = join(root, "coverage-ledger.json");
  writeFileSync(summaryPath, JSON.stringify(content));
  if (ledger !== undefined) writeFileSync(ledgerPath, ledger);
  return { summaryPath, ledgerPath };
}

/**
 * Return the first line a run printed.
 *
 * @param lines - The printed lines.
 * @returns The first one; the case fails when nothing was printed.
 */
function first(lines: readonly string[]): string {
  const [line] = lines;
  assert.ok(line !== undefined, "the run printed nothing");
  return line;
}

const BASE = { "src/a.ts": [[9, 10], [3, 4]], "src/b.tsx": [[5, 5], [0, 0]] } as const;

/**
 * Return the base summary's files as a mutable copy.
 *
 * @returns Relative path to counts.
 */
function base(): Record<string, [[number, number], [number, number]]> {
  return structuredClone(BASE) as unknown as Record<string, [[number, number], [number, number]]>;
}

await test("a percentage has two decimals, rounded down, and an empty measure counts as full", () => {
  assert.equal(percent(2, 3), 66.66);
  assert.equal(percent(1, 3), 33.33);
  assert.equal(percent(3, 3), 100);
  assert.equal(percent(0, 0), 100);
  assert.equal(percent(0, 7), 0);
});

await test("a run equal to the ledger passes", () => {
  const root = folder();
  const ledger = render(readSummary(summary(root, base()), root));
  const { summaryPath, ledgerPath } = inputs(root, summary(root, base()), ledger);
  const result = run(["check", summaryPath, ledgerPath], root);
  assert.equal(result.status, 0);
  assert.deepEqual(result.lines, ["coverage ledger holds: 2 files"]);
});

await test("one file below its ledger value fails and is named with the measure", () => {
  const root = folder();
  const ledger = render(readSummary(summary(root, base()), root));
  const lower = base();
  lower["src/a.ts"] = [[9, 10], [2, 4]];
  const { summaryPath, ledgerPath } = inputs(root, summary(root, lower), ledger);
  const result = run(["check", summaryPath, ledgerPath], root);
  assert.equal(result.status, 1);
  assert.deepEqual(result.lines, [
    "src/a.ts: branches 50 % is below its ledger value 75 %",
    "coverage ledger: 1 failures",
  ]);
});

await test("a statement fall is caught as well as a branch fall", () => {
  const root = folder();
  const ledger = readSummary(summary(root, base()), root);
  const lower = base();
  lower["src/a.ts"] = [[8, 10], [3, 4]];
  assert.deepEqual(compare(readSummary(summary(root, lower), root), ledger), [
    "src/a.ts: statements 80 % is below its ledger value 90 %",
  ]);
});

await test("a new file below 100 % fails and is named", () => {
  const root = folder();
  const ledger = readSummary(summary(root, base()), root);
  const grown = { ...base(), "src/c.ts": [[4, 5], [2, 2]] as [[number, number], [number, number]] };
  assert.deepEqual(compare(readSummary(summary(root, grown), root), ledger), [
    "src/c.ts: a file new to the ledger must be at 100 % (statements 80 %, branches 100 %)",
  ]);
  const branchesOnly = { ...base(), "src/c.ts": [[5, 5], [1, 2]] as [[number, number], [number, number]] };
  assert.equal(compare(readSummary(summary(root, branchesOnly), root), ledger).length, 1);
});

await test("a new file at 100 % passes", () => {
  const root = folder();
  const ledger = readSummary(summary(root, base()), root);
  const grown = { ...base(), "src/c.ts": [[5, 5], [2, 2]] as [[number, number], [number, number]] };
  assert.deepEqual(compare(readSummary(summary(root, grown), root), ledger), []);
});

await test("a file the run no longer measures fails the check and is dropped by accept", () => {
  const root = folder();
  const ledger = render(readSummary(summary(root, base()), root));
  const shrunk = base();
  delete shrunk["src/b.tsx"];
  const { summaryPath, ledgerPath } = inputs(root, summary(root, shrunk), ledger);
  const checked = run(["check", summaryPath, ledgerPath], root);
  assert.equal(checked.status, 1);
  assert.equal(first(checked.lines), "src/b.tsx: in the ledger but not measured by this run; accept the run to drop it");
  const accepted = run(["accept", summaryPath, ledgerPath], root);
  assert.equal(accepted.status, 0);
  assert.deepEqual([...readLedger(JSON.parse(readFileSync(ledgerPath, "utf-8"))).keys()], ["src/a.ts"]);
});

await test("accept refuses a fall and writes nothing", () => {
  const root = folder();
  const ledger = render(readSummary(summary(root, base()), root));
  const lower = base();
  lower["src/a.ts"] = [[1, 10], [3, 4]];
  const { summaryPath, ledgerPath } = inputs(root, summary(root, lower), ledger);
  const result = run(["accept", summaryPath, ledgerPath], root);
  assert.equal(result.status, 1);
  assert.deepEqual(result.lines, [
    "src/a.ts: statements 10 % is below its ledger value 90 %",
    "accept refused: the ledger only rises; nothing was written",
  ]);
  assert.equal(readFileSync(ledgerPath, "utf-8"), ledger);
});

await test("accept refuses a new file below 100 %", () => {
  const root = folder();
  const ledger = render(readSummary(summary(root, base()), root));
  const grown = { ...base(), "src/c.ts": [[1, 2], [0, 0]] as [[number, number], [number, number]] };
  const { summaryPath, ledgerPath } = inputs(root, summary(root, grown), ledger);
  assert.equal(run(["accept", summaryPath, ledgerPath], root).status, 1);
  assert.equal(readFileSync(ledgerPath, "utf-8"), ledger);
});

await test("accept raises an improved file and writes sorted, stable JSON", () => {
  const root = folder();
  const ledger = render(readSummary(summary(root, base()), root));
  const better = base();
  better["src/a.ts"] = [[10, 10], [4, 4]];
  const grown = { "src/z.ts": [[1, 1], [0, 0]] as [[number, number], [number, number]], ...better };
  const { summaryPath, ledgerPath } = inputs(root, summary(root, grown), ledger);
  assert.equal(run(["accept", summaryPath, ledgerPath], root).status, 0);
  const written = readFileSync(ledgerPath, "utf-8");
  assert.equal(
    written,
    `${JSON.stringify(
      {
        schema: LEDGER_SCHEMA,
        files: {
          "src/a.ts": { statements: 100, branches: 100 },
          "src/b.tsx": { statements: 100, branches: 100 },
          "src/z.ts": { statements: 100, branches: 100 },
        },
      },
      null,
      2,
    )}\n`,
  );
  assert.equal(run(["accept", summaryPath, ledgerPath], root).status, 0);
  assert.equal(readFileSync(ledgerPath, "utf-8"), written);
});

await test("without a ledger, check refuses and accept writes the first one as measured", () => {
  const root = folder();
  const { summaryPath, ledgerPath } = inputs(root, summary(root, base()));
  const checked = run(["check", summaryPath, ledgerPath], root);
  assert.equal(checked.status, 2);
  assert.match(first(checked.lines), /^no ledger at .*; write the first one with accept$/);
  const accepted = run(["accept", summaryPath, ledgerPath], root);
  assert.deepEqual(accepted, { status: 0, lines: ["first ledger written: 2 files, as measured"] });
  assert.deepEqual(readLedger(JSON.parse(readFileSync(ledgerPath, "utf-8"))).get("src/a.ts"), {
    statements: 90,
    branches: 75,
  });
});

await test("a malformed summary is refused with its reason", () => {
  const root = folder();
  const cases: [unknown, RegExp][] = [
    [[], /the summary is not an object of files/],
    [null, /the summary is not an object of files/],
    [{ total: entry([0, 0], [0, 0]) }, /the summary measures no file/],
    [{ [join(root, "src/a.ts")]: { statements: { covered: 1, total: 2 } } }, /src\/a\.ts has no valid branches counts/],
    [{ [join(root, "src/a.ts")]: entry([3, 2], [0, 0]) }, /src\/a\.ts has no valid statements counts/],
    [{ [join(root, "src/a.ts")]: entry([1.5, 2], [0, 0]) }, /has no valid statements counts/],
    [{ [join(root, "src/a.ts")]: null }, /has no valid statements counts/],
    [{ "/elsewhere/x.ts": entry([1, 1], [0, 0]) }, /\/elsewhere\/x\.ts lies outside the frontend folder/],
    [{ [root]: entry([1, 1], [0, 0]) }, /lies outside the frontend folder/],
  ];
  for (const [content, reason] of cases) {
    const { summaryPath, ledgerPath } = inputs(root, content, render(new Map()));
    const result = run(["check", summaryPath, ledgerPath], root);
    assert.equal(result.status, 2, String(reason));
    assert.match(first(result.lines), reason);
  }
  const { ledgerPath } = inputs(root, {});
  writeFileSync(join(root, "broken.json"), "{");
  const broken = run(["check", join(root, "broken.json"), ledgerPath], root);
  assert.equal(broken.status, 2);
  assert.match(first(broken.lines), /^the summary .*broken\.json cannot be read as JSON: /);
});

await test("a relative summary path is read from the frontend folder", () => {
  const root = folder();
  assert.deepEqual([...readSummary({ "src/a.ts": entry([1, 1], [1, 1]) }, root).keys()], ["src/a.ts"]);
});

await test("a malformed ledger is refused with its reason", () => {
  const root = folder();
  const cases: [string, RegExp][] = [
    ["{", /^the ledger .* cannot be read as JSON: /],
    [JSON.stringify({ schema: "other", files: {} }), /the ledger's schema is not/],
    [JSON.stringify(null), /the ledger's schema is not/],
    [JSON.stringify({ schema: LEDGER_SCHEMA, files: [] }), /the ledger's files are not an object/],
    [JSON.stringify({ schema: LEDGER_SCHEMA, files: null }), /the ledger's files are not an object/],
    [JSON.stringify({ schema: LEDGER_SCHEMA, files: { "src/a.ts": { statements: 101, branches: 1 } } }), /src\/a\.ts is not two percentages/],
    [JSON.stringify({ schema: LEDGER_SCHEMA, files: { "src/a.ts": { statements: 1 } } }), /is not two percentages/],
    [JSON.stringify({ schema: LEDGER_SCHEMA, files: { "src/a.ts": null } }), /is not two percentages/],
  ];
  for (const [ledger, reason] of cases) {
    const { summaryPath, ledgerPath } = inputs(root, summary(root, base()), ledger);
    const result = run(["check", summaryPath, ledgerPath], root);
    assert.equal(result.status, 2, String(reason));
    assert.match(first(result.lines), reason);
  }
});

await test("a wrong command line prints the usage", () => {
  const root = folder();
  for (const argv of [[], ["check"], ["check", "a"], ["verify", "a", "b"], ["check", "a", "b", "c"], ["check", "", "b"]]) {
    assert.deepEqual(run(argv, root), {
      status: 2,
      lines: ["usage: coverage-ledger.ts <check|accept> <summary.json> <ledger.json>"],
    });
  }
});

await test("an error that is not about the inputs is not swallowed", () => {
  const root = folder();
  const { summaryPath } = inputs(root, summary(root, base()));
  assert.throws(() => run(["accept", summaryPath, join(root, "no-such-folder", "ledger.json")], root), {
    code: "ENOENT",
  });
  assert.ok(new LedgerInputError("x") instanceof Error);
});

await test("the command, run as CI runs it, prints its verdict and exits with its status", () => {
  const root = folder();
  const ledger = render(readSummary(summary(root, base()), root));
  const lower = base();
  lower["src/b.tsx"] = [[4, 5], [0, 0]];
  inputs(root, summary(root, base()), ledger);
  const passed = execFileSync(process.execPath, [command, "check", "coverage-summary.json", "coverage-ledger.json"], {
    cwd: root,
    encoding: "utf-8",
  });
  assert.equal(passed, "coverage ledger holds: 2 files\n");
  inputs(root, summary(root, lower), ledger);
  assert.throws(
    () =>
      execFileSync(process.execPath, [command, "check", "coverage-summary.json", "coverage-ledger.json"], {
        cwd: root,
        encoding: "utf-8",
        stdio: "pipe",
      }),
    (error: { status?: number; stdout?: string }) =>
      error.status === 1 && (error.stdout ?? "").startsWith("src/b.tsx: statements 80 % is below its ledger value 100 %"),
  );
});
