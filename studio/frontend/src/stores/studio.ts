// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio Zustand store
// Studio Zustand store composition root (public entry for App/hooks).

import { create } from "zustand";
import { setStudioAuthToken } from "../api/client";
import { syncStoredStudioAuthToken } from "../studioAuthSession";
import { readStudioStartupHashState } from "../studioStartupRuntime";
import { studioInitialData } from "./studioInitialState";
import { createStudioStoreActions } from "./studioStoreActions";
import type { StudioState } from "./studioTypes";

export type { SourceMode, ViewTab, StudioState } from "./studioTypes";
export type { EvidenceBundleSurface } from "../evidenceBundles";

syncStoredStudioAuthToken(setStudioAuthToken);

export const useStudioStore = create<StudioState>((set, get) => {
  const actions = createStudioStoreActions(
    set as (partial: Partial<StudioState> | ((state: StudioState) => Partial<StudioState>)) => void,
    get,
  );
  return {
    ...studioInitialData,
    ...actions,
  } as StudioState;
});


const startupHashState = readStudioStartupHashState();
if (startupHashState !== null) {
  useStudioStore.getState().selectModel(startupHashState.selectedModelName);
  useStudioStore.setState({
    current: startupHashState.current,
    duration: startupHashState.duration,
    protocol: startupHashState.protocol,
  });
}
