// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Identity of the experiment a result was produced under

/**
 * Telling whether a result still belongs to the configuration on screen.
 *
 * A result is only evidence of the run that produced it. Once the reader
 * changes the model, a parameter, the step or the protocol, every result they
 * are looking at describes an experiment that is no longer configured — and a
 * workflow that keeps calling those steps complete is claiming something it
 * cannot support.
 *
 * The key is a **local** identity, not the server's. It deliberately does not
 * try to reproduce `input_sha256`: the server digests its own canonical form
 * and a second digest taken over a JavaScript object would differ for reasons
 * that have nothing to do with the experiment. What this key has to answer is
 * narrower and answerable — *is this the same configuration as the one that
 * produced that result?* — so it is a canonical rendering of the resolved
 * experiment and nothing more.
 *
 * It uses the evidence seal's canonical form, so key equality means the same
 * thing here as digest equality does elsewhere: sorted keys, one normal form
 * per number, and no dependence on the order fields happened to be written.
 */

import { canonicalSealText } from "./evidenceSeal";
import type { StudioProjectTrainingConfig } from "./studioProjectState";
import type { StudioSimulationConfigInput } from "./studioSimulationConfig";

/**
 * The identity of the experiment a configuration resolves to.
 *
 * Both branches' fields are included whatever the source mode is. A workspace
 * switched from a model to an ODE and back is the same experiment only if the
 * fields it came back to are the same, and reading only the active branch
 * would call those two states identical.
 *
 * @param input - The configuration as the panel holds it.
 * @returns Its identity. Two configurations that would submit the same run
 *   produce the same string; any change the server would see produces a
 *   different one.
 */
export function studioExperimentKey(input: StudioSimulationConfigInput): string {
  return canonicalSealText({
    current: input.current,
    dt: input.dt,
    duration: input.duration,
    equations: input.equations,
    frequencyHz: input.frequencyHz,
    modelParams: input.modelParams,
    odeInit: input.odeInit,
    odeParams: input.odeParams,
    protocol: input.protocol,
    reset: input.reset,
    seed: input.seed,
    selectedModelName: input.selectedModelName,
    sourceMode: input.sourceMode,
    threshold: input.threshold,
    trial: input.trial,
  });
}

/**
 * Whether a result recorded under `recorded` still describes `current`.
 *
 * A result with no recorded key is not current: it predates this mechanism, or
 * it arrived by a path that does not record one, and in both cases the
 * workflow has no evidence that it belongs to the configuration on screen.
 *
 * @param recorded - The key recorded when the result was applied.
 * @param current - The key the configuration resolves to now.
 * @returns Whether the result may be treated as describing the current run.
 */
export function studioResultIsCurrent(recorded: string | null, current: string): boolean {
  return recorded !== null && recorded === current;
}

/**
 * The identity of a training run's configuration.
 *
 * Separate from the experiment key, because training is separate from the
 * experiment: a network trained on a dataset does not stop being trained
 * because the reader changed the integration step on another panel. What
 * invalidates a training run is a change to what was trained -- the dataset,
 * the architecture, the epochs, the surrogate.
 *
 * @param config - The training configuration as the panel holds it.
 * @returns Its identity.
 */
export function studioTrainingKey(config: StudioProjectTrainingConfig): string {
  return canonicalSealText(config);
}
