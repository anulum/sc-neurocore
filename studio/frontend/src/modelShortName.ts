// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — short model labels that never make two models look alike

/**
 * Short labels for catalogue class names.
 *
 * Lists drop the `Neuron` or `Model` suffix to save width. Two models
 * differing only in that suffix, `AstrocyteModel` and `AstrocyteNeuron`,
 * then both read "Astrocyte" and could not be told apart; the library also
 * removed the first "Neuron" or "Model" anywhere in a name, not only at its
 * end. A label is shortened only where the short form is unique.
 */

const SUFFIX = /(?:Neuron|Model)$/;

/**
 * Map every name to the label a list shows for it.
 *
 * @param names - Every class name the list can show, so a label is unique
 *   across the whole catalogue rather than within one filtered view.
 * @returns The label for each name: the name without a trailing `Neuron` or
 *   `Model`, or the full name when that short form is empty or shared.
 */
export function shortModelLabels(names: readonly string[]): Map<string, string> {
  const owners = new Map<string, number>();
  for (const name of new Set(names)) {
    const short = name.replace(SUFFIX, "");
    owners.set(short, (owners.get(short) ?? 0) + 1);
  }
  const labels = new Map<string, string>();
  for (const name of names) {
    const short = name.replace(SUFFIX, "");
    labels.set(name, short === "" || (owners.get(short) ?? 0) > 1 ? name : short);
  }
  return labels;
}
