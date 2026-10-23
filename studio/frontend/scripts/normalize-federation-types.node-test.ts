// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Reproducible federation declaration archive

/** Exercise real producer archives and the public command, with native Node coverage. */
import AdmZip from "adm-zip";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, test } from "node:test";
import { fileURLToPath } from "node:url";

const command = fileURLToPath(new URL("./normalize-federation-types.ts", import.meta.url));
const temporary: string[] = [];

/** Remove only temporary directories allocated by these acceptance cases. */
function cleanup(): void {
  for (const directory of temporary.splice(0)) rmSync(directory, { recursive: true, force: true });
}
afterEach(cleanup);

/**
 * Allocate one owned directory for a complete archive scenario.
 *
 * @returns The newly allocated absolute path.
 */
function directory(): string {
  const path = mkdtempSync(join(tmpdir(), "studio-type-archive-"));
  temporary.push(path);
  return path;
}

/**
 * Produce real declaration entries with distinct source dates and metadata.
 *
 * @param path - Destination ZIP filename.
 * @param year - Source declaration date used by the real producer library.
 */
function producer(path: string, year: number): void {
  const zip = new AdmZip();
  zip.addFile("compiled-types/", Buffer.alloc(0));
  zip.addFile("compiled-types/panel.d.ts", Buffer.from("export interface Panel { ready: boolean; }\n"), "panel API", 0o100644);
  zip.addFile("compiled-types/neurón.d.ts", Buffer.from("export type Spike = 0 | 1;\n"), "Unicode name", 0o100444);
  zip.addZipComment("sc_neurocore complete declaration contract");
  for (const entry of zip.getEntries()) entry.header.time = new Date(year, 5, 7, 8, 9, 10);
  zip.writeZip(path);
}

/** Check date/timezone independence, payload metadata and idempotence through the CLI. */
function reproducibility(): void {
  const root = directory();
  const first = join(root, "first.zip");
  const second = join(root, "second.zip");
  producer(first, 2024);
  producer(second, 2026);
  assert.notDeepEqual(readFileSync(first), readFileSync(second));
  const original = new AdmZip(first);
  execFileSync(process.execPath, [command, first], { env: { ...process.env, TZ: "Pacific/Honolulu" } });
  execFileSync(process.execPath, [command, second], { env: { ...process.env, TZ: "Europe/Zurich" } });
  const normalized = readFileSync(first);
  assert.deepEqual(normalized, readFileSync(second));
  const actual = new AdmZip(first);
  assert.equal(actual.getZipComment(), original.getZipComment());
  assert.equal(actual.getEntries().length, original.getEntries().length);
  for (const entry of original.getEntries()) {
    const retained = actual.getEntry(entry.entryName);
    assert.ok(retained);
    assert.deepEqual(retained.getData(), entry.getData());
    assert.equal(retained.comment, entry.comment);
    assert.equal(retained.attr, entry.attr);
  }
  execFileSync(process.execPath, [command, first], { env: { ...process.env, TZ: "UTC" } });
  assert.deepEqual(readFileSync(first), normalized);
}
await test("normalizes real archives across source dates and timezones while preserving declarations", reproducibility);

/**
 * Refuse invalid inputs without replacing archives or unrelated output.
 *
 * @param kind - Missing argument, absent archive or corrupt archive scenario.
 */
function refusal(kind: string): void {
  const root = directory();
  const archive = join(root, "types.zip");
  const sibling = join(root, "retained.txt");
  writeFileSync(sibling, "preserve unrelated output");
  if (kind === "corrupt") writeFileSync(archive, "invalid ZIP input");
  const args = kind === "argument" ? [command] : [command, archive];
  assert.throws(execFileSync.bind(null, process.execPath, args, { stdio: "pipe" }));
  assert.equal(readFileSync(sibling, "utf8"), "preserve unrelated output");
  if (kind === "corrupt") assert.equal(readFileSync(archive, "utf8"), "invalid ZIP input");
}
for (const kind of ["argument", "missing", "corrupt"]) {
  await test(`refuses ${kind} without replacing an archive or sibling`, refusal.bind(null, kind));
}
