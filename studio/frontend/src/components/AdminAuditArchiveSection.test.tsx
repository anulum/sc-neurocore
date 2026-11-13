// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — the admin controls for quarantined audit records
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { AdminAuditArchiveModel } from "../adminShell";
import AdminAuditArchiveSection from "./AdminAuditArchiveSection";

let root: Root | undefined;
let container: HTMLDivElement;
const actions = {
  onCreateAuditArchive: vi.fn(async (_limit: number) => undefined),
  onLoadAuditArchiveRetention: vi.fn(async (_retainLatest: number) => undefined),
  onPurgeAuditArchiveRetention: vi.fn(async (_retainLatest: number) => undefined),
  onRestoreAuditArchive: vi.fn(async (_archive: Record<string, unknown>, _manifest: Record<string, unknown> | null) => undefined),
  onValidateAuditArchive: vi.fn(async (_archive: Record<string, unknown>, _manifest: Record<string, unknown> | null) => undefined),
};

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  for (const action of Object.values(actions)) action.mockClear();
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root?.unmount(); await Promise.resolve(); });
  document.body.replaceChildren();
  vi.unstubAllGlobals();
});

/**
 * The archive picture of a server with two archives, one a prune candidate.
 *
 * @param overrides - Fields that differ.
 * @returns The model.
 */
function model(overrides: Partial<AdminAuditArchiveModel> = {}): AdminAuditArchiveModel {
  return {
    archiveCount: 2,
    archivedEventCount: 40,
    archiveId: "aa_2",
    artifactCount: 3,
    error: null,
    latestEntries: [],
    lastPurge: "never",
    pruneCandidateCount: 1,
    purgedArchiveCount: 0,
    reasonCounts: "schema: 4",
    retainedArchiveCount: 1,
    retainCount: 1,
    retainLatest: 5,
    restoreArchiveId: "none",
    restoreArtifactCount: 0,
    restoreJobId: "none",
    restoreRows: 0,
    skippedRecordCount: 2,
    validationArchiveId: "none",
    validationErrors: "none",
    validationStatus: "not checked",
    validationWarnings: "none",
    ...overrides,
  };
}

/**
 * Render the section.
 *
 * @param archive - The archive picture.
 * @param auditLoading - Whether an audit request is on its way.
 */
async function render(archive: AdminAuditArchiveModel = model(), auditLoading = false): Promise<void> {
  await act(async () => {
    root?.render(<AdminAuditArchiveSection archive={archive} auditLoading={auditLoading} {...actions} />);
    await Promise.resolve();
  });
}

/**
 * An element by its accessible name.
 *
 * @param name - The aria-label.
 * @returns The element.
 */
function named(name: string): Element {
  const element = container.querySelector(`[aria-label="${name}"]`);
  if (element === null) throw new Error(`no control is named ${name}`);
  return element;
}

/**
 * An input by its accessible name.
 *
 * @param name - The aria-label.
 * @returns The input.
 */
function input(name: string): HTMLInputElement {
  const element = named(name);
  if (!(element instanceof HTMLInputElement)) throw new Error(`${name} is not an input`);
  return element;
}

/**
 * A button by its accessible name.
 *
 * @param name - The aria-label.
 * @returns The button.
 */
function button(name: string): HTMLButtonElement {
  const element = named(name);
  if (!(element instanceof HTMLButtonElement)) throw new Error(`${name} is not a button`);
  return element;
}

/**
 * Set a field through the native setter React listens for.
 *
 * @param name - The field's aria-label.
 * @param text - Its new value.
 */
async function fill(name: string, text: string): Promise<void> {
  const element = named(name);
  const field = element instanceof HTMLTextAreaElement ? element : input(name);
  const prototype = field instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  await act(async () => {
    Object.getOwnPropertyDescriptor(prototype, "value")?.set?.call(field, text);
    field.dispatchEvent(new Event("input", { bubbles: true }));
    await Promise.resolve();
  });
}

/**
 * Press a button by its accessible name.
 *
 * @param name - The aria-label.
 */
async function press(name: string): Promise<void> {
  await act(async () => { button(name).click(); await Promise.resolve(); });
}

/** The warnings the section shows. */
function warnings(): (string | null)[] {
  return [...container.querySelectorAll(".admin-warning")].map((item) => item.textContent);
}

describe("AdminAuditArchiveSection", () => {
  it("shows the archive picture, metric by metric", async () => {
    await render();
    expect(container.querySelector(".admin-section-header span")?.textContent).toBe("2 archives");
    const metrics = Object.fromEntries([...container.querySelectorAll(".admin-metrics > div")].map((row): [string, string] => [
      row.querySelector("span")?.textContent ?? "", row.querySelector("strong")?.textContent ?? "",
    ]));
    expect(metrics).toEqual({
      "Latest archive": "aa_2",
      "Archived rows": "40",
      "Archive artifacts": "3",
      Reasons: "schema: 4",
      Retain: "1",
      "Prune candidates": "1",
      Skipped: "2",
      "Last purge": "never",
      Validation: "not checked",
      "Validation archive": "none",
      "Restore archive": "none",
      "Restore artifacts": "0",
      "Restore rows": "0",
      "Restore job": "none",
    });
    expect(warnings()).toEqual([]);
  });

  it("asks for a retention review before any purge when no inventory is loaded", async () => {
    await render();
    const row = container.querySelector(".admin-audit-row");
    expect(row?.textContent).toBe("unavailableNo archive retention inventory loadedreview retention before executing a purge");
  });

  it("lists each archive with its disposition and counts", async () => {
    await render(model({
      latestEntries: [
        { archiveId: "aa_2", disposition: "retain", eventCount: 30, finishedAt: "2026-11-13T10:00:00Z", jobId: "sj_2", retainedEventCount: 30 },
        { archiveId: "aa_1", disposition: "prune_candidate", eventCount: 10, finishedAt: "2026-11-12T10:00:00Z", jobId: "sj_1", retainedEventCount: 0 },
      ],
    }));
    const rows = [...container.querySelectorAll(".admin-audit-row")];
    expect(rows.map((row) => row.querySelector("span")?.textContent)).toEqual(["retain", "prune_candidate"]);
    expect(rows[1]?.querySelector("strong")?.textContent).toBe("aa_1");
    expect(rows[1]?.querySelector("small")?.textContent).toBe("sj_1 - 10 rows - 0 retained - 2026-11-12T10:00:00Z");
  });

  it("shows the server's error and the validation's errors and warnings", async () => {
    await render(model({ error: "archive store offline", validationErrors: "digest mismatch", validationWarnings: "one row skipped" }));
    expect(warnings()).toEqual(["archive store offline", "digest mismatch", "one row skipped"]);
  });

  it("archives up to the requested limit, and the browser holds back a limit out of range", async () => {
    await render();
    await press("Create audit quarantine archive");
    expect(actions.onCreateAuditArchive).toHaveBeenLastCalledWith(100);
    await fill("Audit archive limit", "250");
    await press("Create audit quarantine archive");
    expect(actions.onCreateAuditArchive).toHaveBeenLastCalledWith(250);
    await fill("Audit archive limit", "5000");
    await press("Create audit quarantine archive");
    expect(actions.onCreateAuditArchive).toHaveBeenCalledTimes(2);
  });

  it("holds a limit that skips the browser's checks between one and a thousand", async () => {
    await render();
    const form = input("Audit archive limit").form;
    for (const [typed, sent] of [["5000", 1000], ["0", 1], ["2.7", 2], ["1e400", 100]] as const) {
      await fill("Audit archive limit", typed);
      await act(async () => {
        form?.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
        await Promise.resolve();
      });
      expect(actions.onCreateAuditArchive).toHaveBeenLastCalledWith(sent);
    }
  });

  it("reviews retention for the requested count and shows the count it sent", async () => {
    await render();
    expect(input("Audit archive retain latest").value).toBe("5");
    await fill("Audit archive retain latest", "7");
    await press("Review audit archive retention");
    expect(actions.onLoadAuditArchiveRetention).toHaveBeenLastCalledWith(7);
    await fill("Audit archive retain latest", "5000");
    await act(async () => {
      input("Audit archive retain latest").form?.dispatchEvent(
        new Event("submit", { bubbles: true, cancelable: true }),
      );
      await Promise.resolve();
    });
    expect(actions.onLoadAuditArchiveRetention).toHaveBeenLastCalledWith(1000);
    expect(input("Audit archive retain latest").value).toBe("1000");
  });

  it("purges with the reviewed count, and only when there is something to purge", async () => {
    await render();
    await fill("Audit archive retain latest", "3");
    await press("Purge audit archive prune candidates");
    expect(actions.onPurgeAuditArchiveRetention).toHaveBeenLastCalledWith(3);
    await render(model({ pruneCandidateCount: 0 }));
    expect(button("Purge audit archive prune candidates").disabled).toBe(true);
  });

  it("falls back to the server's retention count when the field is empty", async () => {
    await render();
    await fill("Audit archive retain latest", "");
    await press("Purge audit archive prune candidates");
    expect(actions.onPurgeAuditArchiveRetention).toHaveBeenLastCalledWith(1);
  });

  it.each([
    ["", "Audit archive JSON is required"],
    ["   ", "Audit archive JSON is required"],
    ["{not json", "Audit archive JSON must be valid JSON"],
    ["[1, 2]", "Audit archive JSON must be a JSON object"],
    ["null", "Audit archive JSON must be a JSON object"],
    ["3", "Audit archive JSON must be a JSON object"],
  ])("refuses the archive %j with a message naming the field", async (typed, message) => {
    await render();
    await fill("Audit archive JSON", typed);
    await press("Validate audit archive restore payload");
    await press("Materialize audit archive restore");
    expect(warnings()).toEqual([message]);
    expect(actions.onValidateAuditArchive).not.toHaveBeenCalled();
    expect(actions.onRestoreAuditArchive).not.toHaveBeenCalled();
  });

  it("refuses a malformed manifest with its own message", async () => {
    await render();
    await fill("Audit archive JSON", '{"rows": []}');
    await fill("Audit archive manifest JSON", "[]");
    await press("Validate audit archive restore payload");
    expect(warnings()).toEqual(["Audit archive manifest JSON must be a JSON object"]);
    expect(actions.onValidateAuditArchive).not.toHaveBeenCalled();
  });

  it("validates and restores an archive, with or without its manifest, and drops a stale message", async () => {
    await render();
    await press("Validate audit archive restore payload");
    expect(warnings()).toEqual(["Audit archive JSON is required"]);
    await fill("Audit archive JSON", ' {"rows": [1]} ');
    await press("Validate audit archive restore payload");
    expect(warnings()).toEqual([]);
    expect(actions.onValidateAuditArchive).toHaveBeenLastCalledWith({ rows: [1] }, null);
    await fill("Audit archive manifest JSON", '{"sha256": "abc"}');
    await press("Materialize audit archive restore");
    expect(actions.onRestoreAuditArchive).toHaveBeenLastCalledWith({ rows: [1] }, { sha256: "abc" });
  });

  it("locks every control while an audit request is on its way", async () => {
    await render(model(), true);
    const controls = [...container.querySelectorAll<HTMLInputElement | HTMLTextAreaElement | HTMLButtonElement>("input:not([type=hidden]), textarea, button")];
    expect(controls).toHaveLength(9);
    expect(controls.every((element) => element.disabled)).toBe(true);
  });

  it("keeps a submitted count in range when its field sends nothing", async () => {
    await render(model(), true);
    await act(async () => {
      for (const form of container.querySelectorAll("form")) {
        form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
      }
      await Promise.resolve();
    });
    expect(actions.onCreateAuditArchive).toHaveBeenLastCalledWith(100);
    expect(actions.onLoadAuditArchiveRetention).toHaveBeenLastCalledWith(5);
  });
});
