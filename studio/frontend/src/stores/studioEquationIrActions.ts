// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Native equation compiler store actions

import { readEquationIrConfiguration } from "../equationIrConfig";
import { compilerConfigurationInvalidatedState } from "../compilerStoreState";
import { runStoreCompile } from "./studioCompile";
import type { StudioState, StudioStoreActions } from "./studioTypes";

/** Native equation actions composed separately from unrelated Studio workflows. */
export type StudioEquationIrActions = Pick<StudioStoreActions,
  "setEquationIrConfig" | "runBuildIR" | "runEmitSV" | "runVerifyIR" | "runReplayIR">;

/**
 * Bind compiler drafts and native actions to the actual Studio store.
 *
 * @param set - Apply owned state transitions.
 * @param get - Read current inputs and request ownership.
 * @returns Complete native compiler actions.
 */
export function createStudioEquationIrActions(
  set: (patch: Partial<StudioState>) => void, get: () => StudioState,
): StudioEquationIrActions {
  return {
    setEquationIrConfig: (config) => { set({
      ...compilerConfigurationInvalidatedState(),
      equationIrConfig: readEquationIrConfiguration(config),
    }); },
    runBuildIR: () => runStoreCompile("ir", get, set),
    runEmitSV: () => runStoreCompile("sv", get, set),
    runVerifyIR: () => runStoreCompile("verify-ir", get, set),
    runReplayIR: () => runStoreCompile("replay-ir", get, set),
  };
}
