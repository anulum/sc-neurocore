// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio sign-in actions against a mocked server and the browser's session storage

import { createStore } from "zustand/vanilla";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  fetchStudioAuthSession,
  loginStudioBrowserUser,
  logoutStudioBrowserUser,
  setStudioAuthToken,
} from "../api/client";
import type { StudioAuthSession, StudioLoginResponse, StudioLogoutResponse } from "../api/types";
import { STUDIO_AUTH_STORAGE_KEY, unauthenticatedStudioAuthSession } from "../studioAuthSession";
import { createStudioAuthActions } from "./studioAuthActions";
import { studioInitialData } from "./studioInitialState";
import { createStudioEquationIrActions } from "./studioEquationIrActions";
import { createStudioStoreActions } from "./studioStoreActions";
import type { StudioState } from "./studioTypes";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    fetchStudioAuthSession: vi.fn(),
    loginStudioBrowserUser: vi.fn(),
    logoutStudioBrowserUser: vi.fn(),
    setStudioAuthToken: vi.fn(),
  };
});

beforeEach(() => { sessionStorage.clear(); });
afterEach(() => {
  vi.resetAllMocks();
  sessionStorage.clear();
});

const SIGNED_IN: StudioAuthSession = { authenticated: true, principal_id: "ada", roles: ["operator"] };
const LOGIN: StudioLoginResponse = {
  access_token: "tok-1", expires_at_utc: "2026-11-18T00:00:00Z", principal_id: "ada", roles: ["operator"],
  token_type: "bearer",
};

/**
 * A store with the real actions over the initial data, its sign-in actions bound to a recorded training
 * binding, and the operator status load recorded instead of run.
 */
function studioStore() {
  const training = { releaseTrainingSession: vi.fn() };
  const store = createStore<StudioState>((set, get) => ({
    ...studioInitialData, ...createStudioStoreActions(set, get), ...createStudioEquationIrActions(set, get),
    ...createStudioAuthActions(set, get, training),
  }));
  const loadOperatorStatus = vi.fn(() => Promise.resolve());
  store.setState({ loadOperatorStatus });
  return { store, training, loadOperatorStatus };
}

/** The token the browser's session storage holds. */
const storedToken = () => sessionStorage.getItem(STUDIO_AUTH_STORAGE_KEY);

/** The token most recently handed to the API client. */
const clientToken = () => vi.mocked(setStudioAuthToken).mock.calls.at(-1)?.[0];

describe("the session check", () => {
  it("asks the server nothing when no token is stored, and reads the user as signed out", async () => {
    const { store } = studioStore();
    store.setState({ authSession: SIGNED_IN });
    await store.getState().loadAuthSession();
    expect(fetchStudioAuthSession).not.toHaveBeenCalled();
    expect(clientToken()).toBeNull();
    expect(store.getState().authSession).toEqual(unauthenticatedStudioAuthSession());
  });

  it("hands a stored token to the client and loads the session it opens", async () => {
    const { store } = studioStore();
    sessionStorage.setItem(STUDIO_AUTH_STORAGE_KEY, "tok-0");
    vi.mocked(fetchStudioAuthSession).mockResolvedValue(SIGNED_IN);
    await store.getState().loadAuthSession();
    expect(vi.mocked(setStudioAuthToken).mock.calls).toEqual([["tok-0"]]);
    expect(store.getState().authSession).toEqual(SIGNED_IN);
    expect(store.getState().authLoading).toBe(false);
    expect(store.getState().authError).toBeNull();
  });

  it("drops a stored token the server refuses, and names the refusal", async () => {
    const { store } = studioStore();
    sessionStorage.setItem(STUDIO_AUTH_STORAGE_KEY, "tok-0");
    vi.mocked(fetchStudioAuthSession).mockRejectedValueOnce(new Error("token expired"));
    await store.getState().loadAuthSession();
    expect(storedToken()).toBeNull();
    expect(clientToken()).toBeNull();
    expect(store.getState().authError).toBe("token expired");
    expect(store.getState().authSession).toEqual(unauthenticatedStudioAuthSession());
    expect(store.getState().authLoading).toBe(false);
    sessionStorage.setItem(STUDIO_AUTH_STORAGE_KEY, "tok-0");
    vi.mocked(fetchStudioAuthSession).mockRejectedValueOnce(new Error(""));
    await store.getState().loadAuthSession();
    expect(store.getState().authError).toBe("Session check failed");
  });
});

describe("signing in", () => {
  it("stores the token, hands it to the client, loads the session and then the operator status", async () => {
    const { store, loadOperatorStatus } = studioStore();
    vi.mocked(loginStudioBrowserUser).mockResolvedValue(LOGIN);
    vi.mocked(fetchStudioAuthSession).mockResolvedValue(SIGNED_IN);
    await store.getState().loginBrowserUser("ada", "s3cret");
    expect(loginStudioBrowserUser).toHaveBeenCalledWith("ada", "s3cret");
    expect(storedToken()).toBe("tok-1");
    expect(clientToken()).toBe("tok-1");
    expect(store.getState().authSession).toEqual(SIGNED_IN);
    expect(store.getState().authLoading).toBe(false);
    expect(loadOperatorStatus).toHaveBeenCalledOnce();
    expect(vi.mocked(fetchStudioAuthSession).mock.invocationCallOrder[0])
      .toBeLessThan(loadOperatorStatus.mock.invocationCallOrder[0] ?? 0);
  });

  it("shows the request as loading until the server answers", async () => {
    const { store } = studioStore();
    let answer!: (value: StudioLoginResponse) => void;
    vi.mocked(loginStudioBrowserUser).mockReturnValue(new Promise((resolve) => { answer = resolve; }));
    vi.mocked(fetchStudioAuthSession).mockResolvedValue(SIGNED_IN);
    store.setState({ authError: "earlier" });
    const signingIn = store.getState().loginBrowserUser("ada", "s3cret");
    expect(store.getState().authLoading).toBe(true);
    expect(store.getState().authError).toBeNull();
    answer(LOGIN);
    await signingIn;
    expect(store.getState().authLoading).toBe(false);
  });

  it("reports a refused sign-in by its reason, or as a failed login, and keeps no token", async () => {
    const { store, loadOperatorStatus } = studioStore();
    vi.mocked(loginStudioBrowserUser).mockRejectedValueOnce(new Error("bad password"));
    await store.getState().loginBrowserUser("ada", "wrong");
    expect(store.getState().authError).toBe("bad password");
    expect(storedToken()).toBeNull();
    expect(clientToken()).toBeNull();
    vi.mocked(loginStudioBrowserUser).mockRejectedValueOnce(new Error(""));
    await store.getState().loginBrowserUser("ada", "wrong");
    expect(store.getState().authError).toBe("Login failed");
    expect(store.getState().authSession).toEqual(unauthenticatedStudioAuthSession());
    expect(loadOperatorStatus).not.toHaveBeenCalled();
  });

  it("removes the new token again when the session it should open cannot be read", async () => {
    const { store, loadOperatorStatus } = studioStore();
    vi.mocked(loginStudioBrowserUser).mockResolvedValue(LOGIN);
    vi.mocked(fetchStudioAuthSession).mockRejectedValue(new Error("session store down"));
    await store.getState().loginBrowserUser("ada", "s3cret");
    expect(storedToken()).toBeNull();
    expect(clientToken()).toBeNull();
    expect(store.getState().authError).toBe("session store down");
    expect(loadOperatorStatus).not.toHaveBeenCalled();
  });
});

describe("signing out", () => {
  /** A store signed in, following a training job, with a graph run on screen and a recovery pending. */
  function signedInStore() {
    const made = studioStore();
    sessionStorage.setItem(STUDIO_AUTH_STORAGE_KEY, "tok-1");
    made.store.setState({
      authSession: SIGNED_IN,
      trainingJobs: [{ job_id: "sj_1", status: "running", config: null }] as StudioState["trainingJobs"],
      trainingJobsError: "earlier", trainingJobsLoading: true,
      trainingJobId: "sj_1", trainingStatus: "running",
      trainingEpochs: [{ epoch: 1 }] as unknown as StudioState["trainingEpochs"],
      trainingObservedConfig: {} as StudioState["trainingObservedConfig"], trainingExperimentKey: "key",
      graphJobRecovery: {} as StudioState["graphJobRecovery"], graphSimResult: {} as StudioState["graphSimResult"],
    });
    return made;
  }

  /** Every state the sign-out must leave, whatever the server answered. */
  function expectSignedOut(store: ReturnType<typeof studioStore>["store"]) {
    expect(storedToken()).toBeNull();
    expect(clientToken()).toBeNull();
    expect(store.getState().authSession).toEqual(unauthenticatedStudioAuthSession());
    expect(store.getState().authLoading).toBe(false);
    expect(store.getState()).toMatchObject({
      trainingJobs: [], trainingJobsError: null, trainingJobsLoading: false,
      trainingJobId: null, trainingStatus: "idle", trainingEpochs: [],
      trainingObservedConfig: null, trainingExperimentKey: null,
      graphJobRecovery: null, graphSimResult: null,
    });
  }

  it("releases the training before it asks the server, and forgets the session, the token and the jobs", async () => {
    const { store, training } = signedInStore();
    let answer!: (value: StudioLogoutResponse) => void;
    vi.mocked(logoutStudioBrowserUser).mockReturnValue(new Promise((resolve) => { answer = resolve; }));
    const signingOut = store.getState().logoutBrowserUser();
    expect(training.releaseTrainingSession).toHaveBeenCalledOnce();
    expect(training.releaseTrainingSession.mock.invocationCallOrder[0])
      .toBeLessThan(vi.mocked(logoutStudioBrowserUser).mock.invocationCallOrder[0] ?? 0);
    expect(store.getState().authLoading).toBe(true);
    expect(store.getState().graphJobRecovery).toBeNull();
    expect(store.getState().graphSimResult).toBeNull();
    answer({ revoked: true });
    await signingOut;
    expectSignedOut(store);
    expect(store.getState().authError).toBeNull();
  });

  it("signs out locally even when the server refuses, and names the refusal", async () => {
    const { store } = signedInStore();
    vi.mocked(logoutStudioBrowserUser).mockRejectedValueOnce(new Error("server unreachable"));
    await store.getState().logoutBrowserUser();
    expectSignedOut(store);
    expect(store.getState().authError).toBe("server unreachable");
    sessionStorage.setItem(STUDIO_AUTH_STORAGE_KEY, "tok-2");
    vi.mocked(logoutStudioBrowserUser).mockRejectedValueOnce(new Error(""));
    await store.getState().logoutBrowserUser();
    expectSignedOut(store);
    expect(store.getState().authError).toBe("Logout failed");
  });
});
