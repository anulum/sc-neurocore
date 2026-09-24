// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — What a share link asks the Studio to do

import type { StudioStartupHashState } from "./studioUrlState";

/** Nothing to apply: the page was not opened with a link. */
export interface NoShareLink {
  kind: "none";
}

/** The link names a model this catalogue does not hold. */
export interface UnknownShareLinkModel {
  kind: "unknown-model";
  modelName: string;
  message: string;
}

/** The link is applicable: these are the settings it asks for. */
export interface ApplicableShareLink {
  kind: "apply";
  modelName: string;
  current: number;
  duration: number;
  protocol: string;
}

/** A model link names a model this catalogue holds: select it and change nothing else. */
export interface SelectableModelLink {
  kind: "select";
  modelName: string;
}

/** What a share link asks for, once judged against the catalogue. */
export type ShareLinkDecision = NoShareLink | UnknownShareLinkModel | ApplicableShareLink;

/**
 * Decide what a share link asks the Studio to do.
 *
 * Separated from the store so the judgement is testable without a browser, a
 * fetch or a store: the action around it only carries the decision out.
 *
 * @param link - What the address fragment decoded to, or `null` for no link.
 * @param knownModels - The catalogue identities currently loaded.
 * @returns Whether to apply the link, refuse it by name, or do nothing.
 */
export function studioShareLinkDecision(
  link: StudioStartupHashState | null,
  knownModels: readonly string[],
): ShareLinkDecision {
  if (link === null) return { kind: "none" };
  if (!knownModels.includes(link.selectedModelName)) {
    // The corpus renames identities and holds an alias of another model, so a
    // link can outlive the name it carries. Saying which name failed beats
    // selecting nothing and leaving the reader to guess whether the link or
    // the Studio is at fault.
    return unknownModel(link.selectedModelName);
  }
  return {
    kind: "apply",
    modelName: link.selectedModelName,
    current: link.current,
    duration: link.duration,
    protocol: link.protocol,
  };
}

/**
 * Say which name a link carried that this catalogue does not hold.
 *
 * @param modelName - The name the link carried.
 * @returns The refusal.
 */
function unknownModel(modelName: string): UnknownShareLinkModel {
  return {
    kind: "unknown-model",
    modelName,
    message:
      `This link opens "${modelName}", which this catalogue does not hold. `
      + "It may have been renamed since the link was made.",
  };
}

/**
 * Decide what a model link asks the Studio to do.
 *
 * @param modelName - The name the link carries, or `null` for no model link.
 * @param knownModels - The catalogue identities currently loaded.
 * @returns Select the model, refuse the name, or do nothing.
 */
export function studioModelLinkDecision(
  modelName: string | null,
  knownModels: readonly string[],
): NoShareLink | UnknownShareLinkModel | SelectableModelLink {
  if (modelName === null) return { kind: "none" };
  if (!knownModels.includes(modelName)) return unknownModel(modelName);
  return { kind: "select", modelName };
}
