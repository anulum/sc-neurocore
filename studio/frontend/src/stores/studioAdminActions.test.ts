// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio operator administration actions against a mocked server

import { createStore } from "zustand/vanilla";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  createStudioAuditQuarantineArchive,
  createStudioIdentityBrowserUser,
  fetchStudioAuditExport,
  fetchStudioAuditQuarantineArchiveRetention,
  fetchStudioAuditStatus,
  fetchStudioIdentityBrowserUsers,
  fetchStudioIdentityServiceAccounts,
  fetchStudioJobs,
  fetchStudioJobStatus,
  fetchStudioOperatorStatus,
  purgeStudioAuditQuarantineArchiveRetention,
  restoreStudioAuditQuarantineArchive,
  rotateStudioIdentityBrowserUserPassword,
  updateStudioIdentityBrowserUser,
  updateStudioIdentityServiceAccount,
  validateStudioAuditQuarantineArchive,
} from "../api/client";
import {
  identityAccountsLoadedState,
  identityAccountsMutatedState,
  jobStatusLoadedState,
  operatorStatusLoadedState,
} from "../adminStoreState";
import {
  auditArchiveCreatedState,
  auditArchivePurgedState,
  auditArchiveRestoredState,
  auditArchiveRetentionLoadedState,
  auditArchiveValidationLoadedState,
  auditExportLoadedState,
  auditStatusLoadedState,
} from "../auditShell";
import { studioInitialData } from "./studioInitialState";
import { createStudioStoreActions } from "./studioStoreActions";
import { createStudioEquationIrActions } from "./studioEquationIrActions";
import type { StudioState } from "./studioTypes";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    createStudioAuditQuarantineArchive: vi.fn(),
    createStudioIdentityBrowserUser: vi.fn(),
    fetchStudioAuditExport: vi.fn(),
    fetchStudioAuditQuarantineArchiveRetention: vi.fn(),
    fetchStudioAuditStatus: vi.fn(),
    fetchStudioIdentityBrowserUsers: vi.fn(),
    fetchStudioIdentityServiceAccounts: vi.fn(),
    fetchStudioJobs: vi.fn(),
    fetchStudioJobStatus: vi.fn(),
    fetchStudioOperatorStatus: vi.fn(),
    purgeStudioAuditQuarantineArchiveRetention: vi.fn(),
    restoreStudioAuditQuarantineArchive: vi.fn(),
    rotateStudioIdentityBrowserUserPassword: vi.fn(),
    updateStudioIdentityBrowserUser: vi.fn(),
    updateStudioIdentityServiceAccount: vi.fn(),
    validateStudioAuditQuarantineArchive: vi.fn(),
  };
});

afterEach(() => {
  vi.resetAllMocks();
});

/** A store with the real actions over the initial data. */
function studioStore() {
  return createStore<StudioState>((set, get) => ({
    ...studioInitialData, ...createStudioStoreActions(set, get), ...createStudioEquationIrActions(set, get),
  }));
}

/** A server answer of the type the mocked function resolves to, made of the fields the store reads. */
function answer<F extends (...args: never[]) => Promise<unknown>>(_: F, value: object): Awaited<ReturnType<F>> {
  return value as Awaited<ReturnType<F>>;
}

const STATUS = answer(fetchStudioAuditStatus, { marker: "status" });
const EXPORT = answer(fetchStudioAuditExport, { marker: "export" });
const OPERATOR = answer(fetchStudioOperatorStatus, { audit: { marker: "audit" }, jobs: { marker: "jobs" } });
const JOBS = answer(fetchStudioJobs, { jobs: [{ marker: "job" }] });
const RETENTION = answer(fetchStudioAuditQuarantineArchiveRetention, { marker: "retention" });
const ACCOUNTS = answer(fetchStudioIdentityServiceAccounts, { service_accounts: [{ marker: "service" }] });
const USERS = answer(fetchStudioIdentityBrowserUsers, { browser_users: [{ marker: "user" }] });
const ARCHIVE = { schema: "archive", records: 3 };
const MANIFEST = { schema: "manifest", sha256: "0".repeat(64) };
// The rotated password the tests send. The API is mocked, so any value serves; it is built rather than
// written out so that the secret scan has no literal to mistake for a credential.
const NEW_PASSWORD = "x".repeat(18);

/** Answer every refresh read the administration actions make. */
function serverAnswersReads(): void {
  vi.mocked(fetchStudioOperatorStatus).mockResolvedValue(OPERATOR);
  vi.mocked(fetchStudioJobs).mockResolvedValue(JOBS);
  vi.mocked(fetchStudioAuditExport).mockResolvedValue(EXPORT);
  vi.mocked(fetchStudioAuditQuarantineArchiveRetention).mockResolvedValue(RETENTION);
  vi.mocked(fetchStudioIdentityServiceAccounts).mockResolvedValue(ACCOUNTS);
  vi.mocked(fetchStudioIdentityBrowserUsers).mockResolvedValue(USERS);
}

describe("audit", () => {
  it("writes the audit status the server reports", async () => {
    const store = studioStore();
    vi.mocked(fetchStudioAuditStatus).mockResolvedValue(STATUS);
    await store.getState().loadAuditStatus();
    expect(store.getState()).toMatchObject(auditStatusLoadedState(STATUS));
  });

  it("reads the last hundred audit records for an export", async () => {
    const store = studioStore();
    vi.mocked(fetchStudioAuditExport).mockResolvedValue(EXPORT);
    await store.getState().loadAuditExport();
    expect(fetchStudioAuditExport).toHaveBeenCalledWith(100);
    expect(store.getState()).toMatchObject(auditExportLoadedState(EXPORT));
  });

  it("archives the quarantine and refreshes the operator status, the jobs and the export", async () => {
    const store = studioStore();
    serverAnswersReads();
    const archive = answer(createStudioAuditQuarantineArchive, { marker: "archive" });
    vi.mocked(createStudioAuditQuarantineArchive).mockResolvedValue(archive);
    await store.getState().createAuditQuarantineArchive(5);
    expect(createStudioAuditQuarantineArchive).toHaveBeenCalledWith(5);
    expect(store.getState()).toMatchObject(auditArchiveCreatedState(archive, EXPORT, OPERATOR, JOBS));
  });

  it("reads the retention plan for the archives to keep", async () => {
    const store = studioStore();
    vi.mocked(fetchStudioAuditQuarantineArchiveRetention).mockResolvedValue(RETENTION);
    await store.getState().loadAuditQuarantineArchiveRetention(3);
    expect(fetchStudioAuditQuarantineArchiveRetention).toHaveBeenCalledWith(3);
    expect(store.getState()).toMatchObject(auditArchiveRetentionLoadedState(RETENTION));
  });

  it("validates an archive against its manifest", async () => {
    const store = studioStore();
    const validation = answer(validateStudioAuditQuarantineArchive, { marker: "validation" });
    vi.mocked(validateStudioAuditQuarantineArchive).mockResolvedValue(validation);
    await store.getState().validateAuditQuarantineArchive(ARCHIVE, MANIFEST);
    expect(validateStudioAuditQuarantineArchive).toHaveBeenCalledWith(ARCHIVE, MANIFEST);
    expect(store.getState()).toMatchObject(auditArchiveValidationLoadedState(validation));
  });

  it("restores an archive and refreshes the operator status and the jobs", async () => {
    const store = studioStore();
    serverAnswersReads();
    const restored = answer(restoreStudioAuditQuarantineArchive, { marker: "restore" });
    vi.mocked(restoreStudioAuditQuarantineArchive).mockResolvedValue(restored);
    await store.getState().restoreAuditQuarantineArchive(ARCHIVE, MANIFEST);
    expect(restoreStudioAuditQuarantineArchive).toHaveBeenCalledWith(ARCHIVE, MANIFEST);
    expect(store.getState()).toMatchObject(auditArchiveRestoredState(restored, OPERATOR, JOBS));
  });

  it("purges by the retention plan and reads the plan again", async () => {
    const store = studioStore();
    serverAnswersReads();
    const purged = answer(purgeStudioAuditQuarantineArchiveRetention, { marker: "purge" });
    vi.mocked(purgeStudioAuditQuarantineArchiveRetention).mockResolvedValue(purged);
    await store.getState().purgeAuditQuarantineArchiveRetention(2);
    expect(purgeStudioAuditQuarantineArchiveRetention).toHaveBeenCalledWith(2);
    expect(fetchStudioAuditQuarantineArchiveRetention).toHaveBeenCalledWith(2);
    expect(store.getState()).toMatchObject(auditArchivePurgedState(purged, RETENTION, OPERATOR, JOBS));
  });

  it.each([
    ["loadAuditStatus", fetchStudioAuditStatus, "Audit status check failed"],
    ["loadAuditExport", fetchStudioAuditExport, "Audit export failed"],
    ["createAuditQuarantineArchive", createStudioAuditQuarantineArchive, "Audit archive creation failed"],
    ["loadAuditQuarantineArchiveRetention", fetchStudioAuditQuarantineArchiveRetention, "Audit archive retention check failed"],
    ["validateAuditQuarantineArchive", validateStudioAuditQuarantineArchive, "Audit archive validation failed"],
    ["restoreAuditQuarantineArchive", restoreStudioAuditQuarantineArchive, "Audit archive restore failed"],
    ["purgeAuditQuarantineArchiveRetention", purgeStudioAuditQuarantineArchiveRetention, "Audit archive retention purge failed"],
  ] as const)("names the failure when %s is refused without a reason", async (action, refused, message) => {
    const store = studioStore();
    vi.mocked(refused).mockRejectedValue(new Error(""));
    const run = store.getState()[action] as (...args: unknown[]) => Promise<void>;
    await run(1, "manifest.json");
    expect(store.getState().auditError).toBe(message);
    expect(store.getState().auditLoading).toBe(false);
  });
});

describe("jobs, identities and the operator", () => {
  it("writes the job status with the job list", async () => {
    const store = studioStore();
    serverAnswersReads();
    const jobStatus = answer(fetchStudioJobStatus, { marker: "job status" });
    vi.mocked(fetchStudioJobStatus).mockResolvedValue(jobStatus);
    await store.getState().loadJobStatus();
    expect(store.getState()).toMatchObject(jobStatusLoadedState(jobStatus, JOBS));
  });

  it("lists the service accounts and the browser users", async () => {
    const store = studioStore();
    serverAnswersReads();
    await store.getState().loadIdentityServiceAccounts();
    expect(store.getState()).toMatchObject(identityAccountsLoadedState(ACCOUNTS, USERS));
  });

  it("writes the operator status with the job list", async () => {
    const store = studioStore();
    serverAnswersReads();
    await store.getState().loadOperatorStatus();
    expect(store.getState()).toMatchObject(operatorStatusLoadedState(OPERATOR, JOBS));
  });

  it.each([
    ["createIdentityBrowserUser", () => createStudioIdentityBrowserUser, [{ username: "ada" }]],
    ["updateIdentityServiceAccount", () => updateStudioIdentityServiceAccount, ["svc-1", { enabled: false }]],
    ["updateIdentityBrowserUser", () => updateStudioIdentityBrowserUser, ["ada", { enabled: false }]],
    ["rotateIdentityBrowserUserPassword", () => rotateStudioIdentityBrowserUserPassword, ["ada", NEW_PASSWORD]],
  ] as const)("%s changes the account and refreshes both lists and the audit export", async (action, mutation, args) => {
    const store = studioStore();
    serverAnswersReads();
    vi.mocked(mutation()).mockResolvedValue(undefined as never);
    const run = store.getState()[action] as (...values: unknown[]) => Promise<void>;
    await run(...args);
    expect(mutation()).toHaveBeenCalledOnce();
    expect(store.getState()).toMatchObject(identityAccountsMutatedState(ACCOUNTS, USERS, EXPORT));
  });

  it("sends a rotated password as the request body's password", async () => {
    const store = studioStore();
    serverAnswersReads();
    vi.mocked(rotateStudioIdentityBrowserUserPassword).mockResolvedValue(undefined as never);
    await store.getState().rotateIdentityBrowserUserPassword("ada", NEW_PASSWORD);
    expect(rotateStudioIdentityBrowserUserPassword).toHaveBeenCalledWith("ada", { password: NEW_PASSWORD });
  });

  it.each([
    ["loadJobStatus", fetchStudioJobStatus, "Job status check failed"],
    ["loadIdentityServiceAccounts", fetchStudioIdentityServiceAccounts, "Identity account check failed"],
    ["createIdentityBrowserUser", createStudioIdentityBrowserUser, "Browser user creation failed"],
    ["updateIdentityServiceAccount", updateStudioIdentityServiceAccount, "Identity account update failed"],
    ["updateIdentityBrowserUser", updateStudioIdentityBrowserUser, "Browser user update failed"],
    ["rotateIdentityBrowserUserPassword", rotateStudioIdentityBrowserUserPassword, "Browser user secret rotation failed"],
    ["loadOperatorStatus", fetchStudioOperatorStatus, "Operator status check failed"],
  ] as const)("names the failure when %s is refused without a reason", async (action, refused, message) => {
    const store = studioStore();
    serverAnswersReads();
    vi.mocked(refused).mockRejectedValue(new Error(""));
    const run = store.getState()[action] as (...args: unknown[]) => Promise<void>;
    await run("ada", { enabled: false });
    expect(store.getState().auditError).toBe(message);
    expect(store.getState().auditLoading).toBe(false);
  });
});

describe("evidence bundles", () => {
  it("downloads an administration bundle's file through the surface downloader", async () => {
    const store = studioStore();
    const downloadEvidenceBundleArtifactForSurface = vi.fn(() => Promise.resolve());
    store.setState({ downloadEvidenceBundleArtifactForSurface });
    await store.getState().downloadEvidenceBundleArtifact("bundle/manifest.json");
    expect(downloadEvidenceBundleArtifactForSurface).toHaveBeenCalledWith("admin", "bundle/manifest.json");
  });
});
