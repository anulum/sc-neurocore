// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/**
 * Indexed reads that say what happened when the element is not there.
 *
 * With `noUncheckedIndexedAccess` on, `items[index]` is `T | undefined`, which
 * is the truth: an index can be out of range and a sparse array can have a
 * hole. The wrong way to satisfy the compiler is `!`, which asserts the
 * opposite of what the type says and turns a missing element into a crash
 * three frames away from its cause. These read the element and, where the
 * caller's own invariant says it must exist, fail immediately and name the
 * index and the length.
 */

/**
 * Read an element that the caller's invariant says must exist.
 *
 * Use this where a missing element is a defect rather than a case: a loop over
 * a list's own indices, a lookup guarded by a length check just above, a test
 * that has just built the list it is reading. Where a missing element is an
 * ordinary outcome, use {@link atOr} or handle the `undefined` directly.
 *
 * An array holding a literal `undefined` is indistinguishable from a hole
 * here, and this throws for both. That is deliberate: the lists this is used
 * on do not hold `undefined`, and treating a stored `undefined` as present
 * would let the assertion pass on data the caller cannot use anyway.
 *
 * @param items - The list to read.
 * @param index - The position to read.
 * @returns The element at that position.
 * @throws {RangeError} When there is no element there.
 */
export function at<T>(items: readonly T[], index: number): T {
  const item = items[index];
  if (item === undefined) {
    throw new RangeError(`index ${String(index)} is outside a list of ${String(items.length)}`);
  }
  return item;
}

/**
 * Read an element, falling back when it is not there.
 *
 * @param items - The list to read.
 * @param index - The position to read.
 * @param fallback - What to use when there is no element there.
 * @returns The element, or the fallback.
 */
export function atOr<T>(items: readonly T[], index: number, fallback: T): T {
  return items[index] ?? fallback;
}
