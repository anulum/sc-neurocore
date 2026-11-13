// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Frontend coverage ledger: no source file's coverage may fall

/**
 * Compare a Vitest v8 `json-summary` with the frontend coverage ledger.
 *
 * The ledger records, per source file, the statement and branch percentage it
 * reached. `check` fails when a file falls below its ledger value, when a file
 * the ledger does not know is below 100 %, or when the ledger names a file the
 * run no longer measured. `accept` writes the ledger again from a run and
 * refuses, writing nothing, whenever `check` would report a fall or a new file
 * below 100 %; it raises improved values and drops files no longer measured.
 * Without a ledger, `accept` writes the first one from the run as it is.
 *
 * Usage: `node scripts/coverage-ledger.ts <check|accept> <summary.json> <ledger.json>`,
 * run in the frontend folder; summary paths are made relative to it.
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { isAbsolute, relative, resolve, sep } from "node:path";

export const LEDGER_SCHEMA = "sc-neurocore.studio-frontend-coverage-ledger.v1";

/** The two measures the ledger keeps for one source file, in percent. */
export interface FileCoverage {
  readonly statements: number;
  readonly branches: number;
}

/** One source file's measures, keyed by its path relative to the frontend folder. */
export type Coverage = ReadonlyMap<string, FileCoverage>;

/** A summary or ledger that cannot be read as one. */
export class LedgerInputError extends Error {}

/**
 * Return a percentage with two decimals, rounded down, so a recorded value never exceeds the measured one.
 *
 * @param covered - Covered items.
 * @param total - All items; a file without any counts as fully covered.
 * @returns The percentage in [0, 100].
 */
export function percent(covered: number, total: number): number {
  if (total === 0) return 100;
  return Math.floor((covered * 10000) / total) / 100;
}

/**
 * Read one measure of a json-summary entry.
 *
 * @param entry - The file's entry.
 * @param name - `statements` or `branches`.
 * @param file - The file, for the message.
 * @returns The measure's percentage.
 */
function measure(entry: unknown, name: "statements" | "branches", file: string): number {
  const counts = (entry as Record<string, unknown> | null)?.[name] as
    | { covered?: unknown; total?: unknown }
    | undefined;
  const covered = counts?.covered;
  const total = counts?.total;
  if (
    typeof covered !== "number" ||
    typeof total !== "number" ||
    !Number.isInteger(covered) ||
    !Number.isInteger(total) ||
    covered < 0 ||
    covered > total
  ) {
    throw new LedgerInputError(`${file} has no valid ${name} counts`);
  }
  return percent(covered, total);
}

/**
 * Read a Vitest v8 json-summary into per-file coverage.
 *
 * @param summary - The parsed summary.
 * @param root - The frontend folder; every file must lie inside it.
 * @returns The coverage of every file but the `total` entry.
 */
export function readSummary(summary: unknown, root: string): Coverage {
  if (summary === null || typeof summary !== "object" || Array.isArray(summary)) {
    throw new LedgerInputError("the summary is not an object of files");
  }
  const files = new Map<string, FileCoverage>();
  for (const [key, entry] of Object.entries(summary)) {
    if (key === "total") continue;
    const path = relative(root, isAbsolute(key) ? key : resolve(root, key)).split(sep).join("/");
    if (path === "" || path.startsWith("../") || path === ".." || isAbsolute(path)) {
      throw new LedgerInputError(`${key} lies outside the frontend folder`);
    }
    files.set(path, {
      statements: measure(entry, "statements", path),
      branches: measure(entry, "branches", path),
    });
  }
  if (files.size === 0) throw new LedgerInputError("the summary measures no file");
  return files;
}

/**
 * Read a ledger file's parsed content.
 *
 * @param ledger - The parsed ledger.
 * @returns Its coverage per file.
 */
export function readLedger(ledger: unknown): Coverage {
  const record = ledger as { schema?: unknown; files?: unknown } | null;
  if (record?.schema !== LEDGER_SCHEMA) {
    throw new LedgerInputError(`the ledger's schema is not ${LEDGER_SCHEMA}`);
  }
  if (record.files === null || typeof record.files !== "object" || Array.isArray(record.files)) {
    throw new LedgerInputError("the ledger's files are not an object");
  }
  const files = new Map<string, FileCoverage>();
  for (const [path, entry] of Object.entries(record.files)) {
    const value = entry as { statements?: unknown; branches?: unknown } | null;
    const statements = value?.statements;
    const branches = value?.branches;
    if (!isPercent(statements) || !isPercent(branches)) {
      throw new LedgerInputError(`the ledger's entry for ${path} is not two percentages`);
    }
    files.set(path, { statements, branches });
  }
  return files;
}

/**
 * Return whether a value is a finite number between 0 and 100.
 *
 * @param value - Any value.
 * @returns Whether it is a percentage.
 */
function isPercent(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 100;
}

/**
 * Compare a run with the ledger.
 *
 * @param run - The run's coverage.
 * @param ledger - The ledger's coverage.
 * @returns One sentence per failure, sorted by file; empty when the run keeps the ledger.
 */
export function compare(run: Coverage, ledger: Coverage): string[] {
  const failures: string[] = [];
  for (const [file, now] of run) {
    const kept = ledger.get(file);
    if (kept === undefined) {
      if (now.statements < 100 || now.branches < 100) {
        failures.push(
          `${file}: a file new to the ledger must be at 100 % ` +
            `(statements ${now.statements} %, branches ${now.branches} %)`,
        );
      }
      continue;
    }
    for (const name of ["statements", "branches"] as const) {
      if (now[name] < kept[name]) {
        failures.push(`${file}: ${name} ${now[name]} % is below its ledger value ${kept[name]} %`);
      }
    }
  }
  for (const file of ledger.keys()) {
    if (!run.has(file)) {
      failures.push(`${file}: in the ledger but not measured by this run; accept the run to drop it`);
    }
  }
  return failures.sort();
}

/**
 * Render coverage as a ledger: sorted files, each with its two measures in a fixed order.
 *
 * @param coverage - The coverage to record.
 * @returns The ledger's text, with a final newline.
 */
export function render(coverage: Coverage): string {
  const files: Record<string, FileCoverage> = {};
  const sorted = [...coverage.entries()].sort(([left], [right]) => (left < right ? -1 : 1));
  for (const [file, { statements, branches }] of sorted) {
    files[file] = { statements, branches };
  }
  return `${JSON.stringify({ schema: LEDGER_SCHEMA, files }, null, 2)}\n`;
}

/**
 * Parse a JSON file for one of the two inputs.
 *
 * @param path - The file.
 * @param what - `summary` or `ledger`, for the message.
 * @returns The parsed content.
 */
function readJson(path: string, what: string): unknown {
  try {
    return JSON.parse(readFileSync(path, "utf-8")) as unknown;
  } catch (error) {
    throw new LedgerInputError(`the ${what} ${path} cannot be read as JSON: ${(error as Error).message}`);
  }
}

/**
 * Run the command.
 *
 * @param argv - `<check|accept> <summary.json> <ledger.json>`.
 * @param root - The frontend folder.
 * @param log - Where to print the verdict.
 * @returns 0 when the ledger holds (or was written), 1 when the run falls, 2 on unusable input.
 */
export function main(argv: readonly string[], root: string, log: (line: string) => void): number {
  const [mode, summaryPath, ledgerPath] = argv;
  if ((mode !== "check" && mode !== "accept") || !summaryPath || !ledgerPath || argv.length !== 3) {
    log("usage: coverage-ledger.ts <check|accept> <summary.json> <ledger.json>");
    return 2;
  }
  try {
    const run = readSummary(readJson(summaryPath, "summary"), root);
    if (!existsSync(ledgerPath)) {
      if (mode === "check") {
        log(`no ledger at ${ledgerPath}; write the first one with accept`);
        return 2;
      }
      writeFileSync(ledgerPath, render(run));
      log(`first ledger written: ${run.size} files, as measured`);
      return 0;
    }
    const failures = compare(run, readLedger(readJson(ledgerPath, "ledger")));
    const falls = failures.filter((failure) => !failure.endsWith("accept the run to drop it"));
    if (mode === "check") {
      for (const failure of failures) log(failure);
      log(
        failures.length === 0
          ? `coverage ledger holds: ${run.size} files`
          : `coverage ledger: ${failures.length} failures`,
      );
      return failures.length === 0 ? 0 : 1;
    }
    if (falls.length > 0) {
      for (const failure of falls) log(failure);
      log("accept refused: the ledger only rises; nothing was written");
      return 1;
    }
    writeFileSync(ledgerPath, render(run));
    log(`ledger accepted: ${run.size} files`);
    return 0;
  } catch (error) {
    if (!(error instanceof LedgerInputError)) throw error;
    log(error.message);
    return 2;
  }
}

if (import.meta.main) {
  process.exitCode = main(process.argv.slice(2), process.cwd(), (line) => {
    console.log(line);
  });
}
