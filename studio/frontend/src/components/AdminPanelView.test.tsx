// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — what the admin panel sends when an operator acts on it
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { buildAdminShellModel, type AdminShellModel } from "../adminShell";
import type { StudioEvidenceBundleRequest } from "../api/client";
import AdminPanelView, { type AdminPanelViewProps } from "./AdminPanelView";

/** The panel before anything has arrived from the server. */
const EMPTY = buildAdminShellModel({
  auditArchive: null,
  auditArchivePurge: null,
  auditArchiveRestore: null,
  auditArchiveRetention: null,
  auditArchiveValidation: null,
  auditError: null,
  auditExport: null,
  auditStatus: null,
  capabilities: [],
  evidenceBundle: null,
  evidenceBundleError: null,
  evidenceBundleLoading: false,
  identityBrowserUsers: [],
  identityServiceAccounts: [],
  jobRecords: [],
  jobStatus: null,
  operatorStatus: null,
});

type Actions = Omit<AdminPanelViewProps, "auditLoading" | "model">;

const actions = {
  onCreateAuditArchive: vi.fn(async (_limit: number) => undefined),
  onCreateEvidenceBundle: vi.fn(async (_request: StudioEvidenceBundleRequest) => undefined),
  onCreateIdentityBrowserUser: vi.fn(async (_create: Parameters<Actions["onCreateIdentityBrowserUser"]>[0]) => undefined),
  onDownloadEvidenceArtifact: vi.fn(async (_relativePath: string) => undefined),
  onLoadAuditExport: vi.fn(async () => undefined),
  onLoadAuditArchiveRetention: vi.fn(async (_retainLatest: number) => undefined),
  onLoadAuditStatus: vi.fn(async () => undefined),
  onLoadIdentityServiceAccounts: vi.fn(async () => undefined),
  onLoadJobStatus: vi.fn(async () => undefined),
  onLoadOperatorStatus: vi.fn(async () => undefined),
  onPurgeAuditArchiveRetention: vi.fn(async (_retainLatest: number) => undefined),
  onRestoreAuditArchive: vi.fn(async (_archive: Record<string, unknown>, _manifest: Record<string, unknown> | null) => undefined),
  onRotateIdentityBrowserUserPassword: vi.fn(async (_username: string, _password: string) => undefined),
  onUpdateIdentityBrowserUser: vi.fn(async (_username: string, _update: Parameters<Actions["onUpdateIdentityBrowserUser"]>[1]) => undefined),
  onUpdateIdentityServiceAccount: vi.fn(async (_principalId: string, _update: Parameters<Actions["onUpdateIdentityServiceAccount"]>[1]) => undefined),
  onValidateAuditArchive: vi.fn(async (_archive: Record<string, unknown>, _manifest: Record<string, unknown> | null) => undefined),
} satisfies Actions;

let root: Root | undefined;
let container: HTMLDivElement;

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  for (const action of Object.values(actions)) action.mockReset();
  for (const action of Object.values(actions)) action.mockResolvedValue(undefined);
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
 * Render the panel.
 *
 * @param overrides - Sections that differ from an empty panel.
 * @param auditLoading - Whether an audit or identity request is on its way.
 */
async function render(overrides: Partial<AdminShellModel> = {}, auditLoading = false): Promise<void> {
  await act(async () => {
    root?.render(<AdminPanelView auditLoading={auditLoading} model={{ ...EMPTY, ...overrides }} {...actions} />);
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
 * A text area by its accessible name.
 *
 * @param name - The aria-label.
 * @returns The text area.
 */
function textarea(name: string): HTMLTextAreaElement {
  const element = named(name);
  if (!(element instanceof HTMLTextAreaElement)) throw new Error(`${name} is not a text area`);
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
 * Press a button and let the work it starts settle.
 *
 * @param button - The button.
 */
async function press(button: HTMLButtonElement): Promise<void> {
  await act(async () => { button.click(); await new Promise((done) => { setTimeout(done, 0); }); });
}

/**
 * Type into a field through the native setter React listens for.
 *
 * @param name - The field's aria-label.
 * @param text - What to type.
 */
async function type(name: string, text: string): Promise<void> {
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
 * A button by its text.
 *
 * @param text - What it reads.
 * @returns The button.
 */
function buttonReading(text: string): HTMLButtonElement {
  const found = [...container.querySelectorAll("button")].find((item) => item.textContent === text);
  if (found === undefined) throw new Error(`no button reads ${text}`);
  return found;
}

/** The text of every warning on the panel. */
function warnings(): (string | null)[] {
  return [...container.querySelectorAll(".admin-warning")].map((item) => item.textContent);
}

/** A promise settled by hand. */
interface Deferred {
  promise: Promise<undefined>;
  resolve: () => void;
}

/**
 * A promise the test settles when it chooses.
 *
 * @returns The promise and its settling hand.
 */
function deferred(): Deferred {
  let resolve: () => void = () => undefined;
  const promise = new Promise<undefined>((done) => { resolve = () => { done(undefined); }; });
  return { promise, resolve };
}

const BROWSER_USER = {
  active: false, activeLabel: "disabled" as const, expiresAt: "2030-01-01T00:00:00Z",
  principalId: "human-operator", rolesText: "studio.viewer", username: "operator",
};

describe("AdminPanelView", () => {
  it("says what has not been loaded yet", async () => {
    await render();
    const text = container.textContent;
    expect(text).toContain("No persistent accounts loaded");
    expect(text).toContain("No browser users loaded");
    expect(text).toContain("All registered capabilities healthy");
    expect(container.querySelector(".admin-evidence-section .admin-section-header span")?.textContent).toBe("ready");
    expect(container.querySelectorAll(".admin-manifest-row, .admin-artifact-row, .admin-job-row")).toHaveLength(0);
    expect(warnings()).toEqual([]);
  });

  it("refreshes each status on request, the readiness panel's two buttons included", async () => {
    await render();
    for (const name of ["Refresh operator status", "Refresh identity accounts", "Refresh audit status", "Export audit events", "Refresh job status"]) {
      await press(button(name));
    }
    await press(buttonReading("Refresh"));
    await press(buttonReading("Refresh status"));
    expect(actions.onLoadOperatorStatus).toHaveBeenCalledTimes(3);
    expect(actions.onLoadIdentityServiceAccounts).toHaveBeenCalledTimes(1);
    expect(actions.onLoadAuditStatus).toHaveBeenCalledTimes(1);
    expect(actions.onLoadAuditExport).toHaveBeenCalledTimes(1);
    expect(actions.onLoadJobStatus).toHaveBeenCalledTimes(1);
  });

  it("saves a service account's roles and activity", async () => {
    await render({
      identityAccounts: [{ active: true, activeLabel: "active", expiresAt: "never", principalId: "svc-admin", rolesText: "studio.admin, studio.viewer" }],
    });
    input("svc-admin roles").value = " studio.viewer ,, ";
    input("svc-admin active").click();
    await press(button("Save svc-admin identity"));
    expect(actions.onUpdateIdentityServiceAccount).toHaveBeenCalledWith("svc-admin", {
      active: false, expires_at_utc: null, roles: ["studio.viewer"],
    });
  });

  it("creates a browser user, then clears the form for the next one with Active checked again", async () => {
    await render();
    input("New browser username").value = " ada ";
    input("New browser principal").value = " human-ada ";
    input("New browser roles").value = "studio.viewer, studio.operator";
    input("New browser secret").value = "s3cret";
    input("New browser active").click();
    await press(button("Create browser user"));
    expect(actions.onCreateIdentityBrowserUser).toHaveBeenCalledWith({
      active: false,
      expires_at_utc: null,
      password: "s3cret",
      principal_id: "human-ada",
      roles: ["studio.viewer", "studio.operator"],
      username: "ada",
    });
    expect(input("New browser username").value).toBe("");
    expect(input("New browser secret").value).toBe("");
    expect(input("New browser roles").value).toBe("studio.viewer");
    expect(input("New browser active").checked).toBe(true);
  });

  it("leaves the create form as typed until the server has answered", async () => {
    const pending = deferred();
    actions.onCreateIdentityBrowserUser.mockReturnValueOnce(pending.promise);
    await render();
    input("New browser username").value = "ada";
    input("New browser principal").value = "human-ada";
    input("New browser secret").value = "s3cret";
    await press(button("Create browser user"));
    expect(input("New browser username").value).toBe("ada");
    await act(async () => { pending.resolve(); await new Promise((done) => { setTimeout(done, 0); }); });
    expect(input("New browser username").value).toBe("");
  });

  it("updates a browser user without touching its credential when no new secret is typed", async () => {
    await render({ identityBrowserUsers: [BROWSER_USER] });
    expect(container.textContent).toContain("human-operator - 2030-01-01T00:00:00Z");
    input("operator browser roles").value = "studio.viewer, studio.admin";
    input("operator browser active").click();
    await press(button("Save operator browser user"));
    expect(actions.onUpdateIdentityBrowserUser).toHaveBeenCalledWith("operator", {
      active: true, expires_at_utc: null, roles: ["studio.viewer", "studio.admin"],
    });
    expect(actions.onRotateIdentityBrowserUserPassword).not.toHaveBeenCalled();
  });

  it("rotates a typed secret only after the update has landed, then empties the field", async () => {
    const update = deferred();
    actions.onUpdateIdentityBrowserUser.mockReturnValueOnce(update.promise);
    await render({ identityBrowserUsers: [BROWSER_USER] });
    input("operator new secret").value = "n3w-secret";
    await press(button("Save operator browser user"));
    expect(actions.onUpdateIdentityBrowserUser).toHaveBeenCalledTimes(1);
    expect(actions.onRotateIdentityBrowserUserPassword).not.toHaveBeenCalled();
    expect(input("operator new secret").value).toBe("n3w-secret");
    await act(async () => { update.resolve(); await new Promise((done) => { setTimeout(done, 0); }); });
    expect(actions.onRotateIdentityBrowserUserPassword).toHaveBeenCalledWith("operator", "n3w-secret");
    expect(input("operator new secret").value).toBe("");
  });

  it("locks the identity and audit controls while such a request is on its way", async () => {
    await render({ identityAccounts: [{ active: true, activeLabel: "active", expiresAt: "never", principalId: "svc-admin", rolesText: "" }], identityBrowserUsers: [BROWSER_USER] }, true);
    for (const name of [
      "Refresh operator status", "Refresh identity accounts", "Refresh audit status", "Export audit events", "Refresh job status",
      "svc-admin roles", "svc-admin active", "Save svc-admin identity",
      "New browser username", "New browser principal", "New browser roles", "New browser secret", "New browser active", "Create browser user",
      "operator browser roles", "operator browser active", "operator new secret", "Save operator browser user",
    ]) {
      const element = named(name);
      expect(element instanceof HTMLButtonElement ? element.disabled : input(name).disabled, name).toBe(true);
    }
  });

  it("shows the audit picture's errors and recent events, naming an anonymous principal", async () => {
    await render({
      audit: { ...EMPTY.audit, error: "export failed", lastError: "AuditPathIsDirectory", latestBrowserAuthAction: "studio.auth.login", latestIdentityLifecycleAction: null },
      recentAuditEvents: [
        {
          action: "studio.audit.export", decision: "deny", event_hash: "hash-1", previous_event_hash: null, principal_id: "operator-1",
          reason: "missing_admin_role", request_id: "req-1", route: "/api/studio/audit/export", schema_version: "studio.audit.v1", timestamp_utc: "2026-06-19T20:01:00Z",
        },
        {
          action: "studio.auth.login", decision: "deny", event_hash: null, previous_event_hash: null, principal_id: null,
          reason: "invalid_browser_login", request_id: null, route: "/api/studio/auth/login", schema_version: "studio.audit.v1", timestamp_utc: null,
        },
      ],
    });
    expect(warnings()).toEqual(["export failed", "AuditPathIsDirectory"]);
    const rows = [...container.querySelectorAll(".admin-audit-row")].filter((row) => row.querySelector("span")?.textContent === "deny");
    expect(rows.map((row) => row.querySelector("small")?.textContent)).toEqual([
      "operator-1 - missing_admin_role",
      "anonymous - invalid_browser_login",
    ]);
    const latest = Object.fromEntries([...container.querySelectorAll(".admin-metrics > div")].map((row): [string, string] => [
      row.firstElementChild?.textContent ?? "", row.lastElementChild?.textContent ?? "",
    ]));
    expect(latest["Latest auth"]).toBe("studio.auth.login");
    expect(latest["Latest identity"]).toBe("none");
  });

  it("lists jobs with their failures and adds each to the bundle once, after what was typed", async () => {
    const job = {
      artifactCount: 2, artifactPaths: "a.txt, b.json", createdAt: "t0", evidenceArtifactCount: 1, error: null,
      executionModel: "process", finishedAt: "t1", jobId: "sj_1", kind: "synthesis", owner: "operator-1", status: "completed" as const,
    };
    await render({
      jobs: { ...EMPTY.jobs, configured: true, resourceProfiles: ["compiler: 3s"] },
      jobRecords: [job, { ...job, jobId: "sj_2", status: "failed", error: "yosys exited 1" }],
    });
    const rows = [...container.querySelectorAll(".admin-job-row small:first-of-type")].map((item) => item.textContent);
    expect(rows).toEqual([
      "operator-1 - process - t1 - 2 artifacts - 1 evidence",
      "operator-1 - process - t1 - 2 artifacts - 1 evidence - yosys exited 1",
    ]);
    expect(container.textContent).toContain("configured");
    expect(container.textContent).toContain("compiler: 3s");
    await type("Evidence job IDs", "sj_0, , ");
    await press(button("Add sj_1 to evidence bundle"));
    expect(input("Evidence job IDs").value).toBe("sj_0, sj_1");
    await press(button("Add sj_1 to evidence bundle"));
    await press(button("Add sj_2 to evidence bundle"));
    expect(input("Evidence job IDs").value).toBe("sj_0, sj_1, sj_2");
  });

  it("exports a bundle exactly as the form reads", async () => {
    await render();
    input("Evidence project name").value = "demo";
    await type("Evidence job IDs", "sj_1, sj_2");
    textarea("Evidence simulation JSON").value = '{"run": 1}';
    input("Include audit export").click();
    input("Evidence audit limit").value = "50";
    input("Evidence replay route").value = "/api/studio/simulate";
    input("Evidence operator note").value = "for the review";
    await press(button("Create evidence bundle"));
    expect(actions.onCreateEvidenceBundle).toHaveBeenCalledWith({
      audit_limit: 50,
      analysis_results: [],
      command_replay: { method: "POST", route: "/api/studio/simulate", note: "for the review" },
      default_flow_attestations: [],
      default_flow_runs: [],
      include_audit: false,
      job_ids: ["sj_1", "sj_2"],
      model_scan_results: [],
      project_name: "demo",
      simulation_results: [{ run: 1 }],
      weight_restore_results: [],
      weight_restore_attach_results: [],
    });
  });

  it("shows a bundle's manifest and downloads its artefacts", async () => {
    await render({
      evidenceBundle: {
        ...EMPTY.evidenceBundle,
        bundleId: "seb_1",
        error: "one source job was missing",
        entries: [{ classification: "audit", detail: "evidence/audit-export.json", index: 0, source: "bundle", type: "audit_export" }],
        artifacts: [{ relativePath: "evidence/manifest.json", sha256: "b".repeat(64), sha256Label: "bbbbbbbbbbbb", sizeBytes: 256, sizeLabel: "256 B" }],
      },
    });
    expect(warnings()).toEqual(["one source job was missing"]);
    expect(container.querySelector(".admin-manifest-row small")?.textContent).toBe("audit - evidence/audit-export.json");
    expect(container.querySelector(".admin-artifact-row small")?.textContent).toBe("256 B - sha bbbbbbbbbbbb");
    await press(button("Download evidence artifact evidence/manifest.json"));
    expect(actions.onDownloadEvidenceArtifact).toHaveBeenCalledWith("evidence/manifest.json");
  });

  it("locks the evidence controls while a bundle is being exported", async () => {
    const job = {
      artifactCount: 0, artifactPaths: "", createdAt: "t0", evidenceArtifactCount: 0, error: null,
      executionModel: "thread", finishedAt: "t1", jobId: "sj_1", kind: "compiler", owner: "operator-1", status: "completed" as const,
    };
    await render({
      jobRecords: [job],
      evidenceBundle: {
        ...EMPTY.evidenceBundle,
        loading: true,
        artifacts: [{ relativePath: "evidence/manifest.json", sha256: "", sha256Label: "", sizeBytes: null, sizeLabel: "unknown" }],
      },
    });
    expect(container.querySelector(".admin-evidence-section .admin-section-header span")?.textContent).toBe("exporting");
    const form = [...container.querySelectorAll<HTMLInputElement | HTMLTextAreaElement | HTMLButtonElement>(
      ".admin-evidence-section form input, .admin-evidence-section form textarea, .admin-evidence-section form button",
    )];
    expect(form).toHaveLength(16);
    const controls = [
      button("Add sj_1 to evidence bundle"),
      button("Download evidence artifact evidence/manifest.json"),
      ...form,
    ];
    expect(controls.filter((element) => !element.disabled)).toEqual([]);
  });

  it("names each unhealthy capability with its reason", async () => {
    await render({
      capabilities: { ...EMPTY.capabilities, registered: 2 },
      unhealthyCapabilities: [{
        capability_id: "studio.synthesis_dashboard", title: "Synthesis Dashboard", summary: "", status: "unavailable",
        healthy: false, message: "Yosys unavailable.", requirements: [], evidence: [], ui_placement: "Admin", docs_path: null,
      }],
    });
    expect(container.textContent).toContain("2 registered");
    expect(container.textContent).not.toContain("All registered capabilities healthy");
    const row = [...container.querySelectorAll(".admin-audit-row")].find((item) => item.querySelector("strong")?.textContent === "Synthesis Dashboard");
    expect(row?.querySelector("span")?.textContent).toBe("unavailable");
    expect(row?.querySelector("small")?.textContent).toBe("Yosys unavailable.");
  });
});
