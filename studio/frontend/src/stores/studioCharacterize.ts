// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Characterisation transport ownership

import { connectProgress, fetchCharacterize, type ProgressMessage } from "../api/client";
import { characterizeFailureState, characterizeProgressMessageState,
  characterizeRequestConfig, characterizeRunStartState } from "../characterizeStoreState";
import { studioExperimentKey } from "../studioExperimentKey";
import { studioSimulationConfigInput } from "../studioSimulationConfigInput";
import type { StudioState } from "./studioTypes";

/**
 * Characterise one captured experiment with at most one HTTP fallback.
 *
 * Terminal or obsolete requests detach their socket; both transports validate
 * results identically and can publish only to the captured experiment.
 *
 * @param get - Read live experiment state.
 * @param set - Patch state without replacing unrelated results.
 */
export function runStoreCharacterize(
  get: () => StudioState,
  set: (patch: Partial<StudioState>) => void,
): void {
  const state = get();
  if (state.isSimulating || !state.selectedModelName) return;
  set({ ...characterizeRunStartState(), analysisExperimentKey: null });
  let socket: WebSocket | undefined;
  let done = false;
  let fallbackStarted = false;
  const detach = (): void => {
    if (!socket) return;
    socket.onopen = socket.onmessage = socket.onerror = socket.onclose = null;
    socket.close();
    socket = undefined;
  };
  try {
    const key = studioExperimentKey(studioSimulationConfigInput(state));
    const config = characterizeRequestConfig(state);
    const current = (): boolean => {
      if (done) return false;
      try {
        if (studioExperimentKey(studioSimulationConfigInput(get())) === key) return true;
        set({ isSimulating: false, progressPct: 0, progressMsg: "" });
      } catch (error: unknown) {
        set(characterizeFailureState(error));
      }
      done = true;
      detach();
      return false;
    };
    const receive = (message: ProgressMessage): void => {
      if (!current()) return;
      const patch = characterizeProgressMessageState(message);
      if (!patch) return;
      if ("isSimulating" in patch) {
        done = true;
        detach();
        set({ ...patch, analysisExperimentKey: "charResult" in patch ? key : null });
      } else set(patch);
    };
    const fallback = (): void => {
      if (fallbackStarted || !current()) return;
      fallbackStarted = true;
      detach();
      void fetchCharacterize(config).then(
        (result) => { receive({ type: "complete", result }); },
        (error: unknown) => {
          receive({ type: "error", msg: characterizeFailureState(error).error });
        },
      );
    };
    try {
      socket = connectProgress("characterize", config, (message) => {
        if (!fallbackStarted) receive(message);
      });
      socket.onerror = fallback;
      socket.onclose = fallback;
    } catch { fallback(); }
  } catch (error: unknown) {
    done = true;
    detach();
    set(characterizeFailureState(error));
  }
}
