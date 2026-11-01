// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Account-bound graph recovery state

import type { StateCreator } from "zustand";
import type { StudioState } from "./stores/studioTypes";

/**
 * Clear account-owned graph state before subscribers or saves can observe another account.
 *
 * @param initialize - Actual Studio store initializer and actions.
 * @returns Initializer guarding actions and the public setState boundary.
 */
export function withGraphAccountState(initialize: StateCreator<StudioState>): StateCreator<StudioState> {
  return (set, get, api) => {
    const setForAccount: typeof set = (partial, replace) => {
      const before = get();
      const next = typeof partial === "function" ? partial(before) : partial;
      const session = next.authSession === undefined ? before.authSession : next.authSession;
      const ownerChanged = before.authSession?.principal_id !== session?.principal_id;
      const recovery = next.graphJobRecovery === undefined ? before.graphJobRecovery : next.graphJobRecovery;
      const foreignRecovery = recovery !== null && recovery.owner !== (session?.principal_id ?? "local");
      const patch = ownerChanged || foreignRecovery ? { ...next, graphJobRecovery: null, graphSimResult: null } : next;
      if (replace === true) set({ ...before, ...patch }, true);
      else set(patch);
    };
    api.setState = setForAccount;
    return initialize(setForAccount, get, api);
  };
}
