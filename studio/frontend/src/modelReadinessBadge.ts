// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — What a model's readiness lets Studio call it

import type { ModelReadiness } from "./api/client";

/** A readiness badge: its text, whether it is a verified claim, and its tooltip. */
export interface ReadinessBadge {
  label: string;
  verified: boolean;
  title: string;
}

/**
 * Decide what the model panel may say about a model being perfect.
 *
 * Only tiers bound to fresh facet receipts may be called perfect. A model
 * whose descriptor flags claim perfection without such receipts is labelled
 * as a declaration, so the claim is visible but not presented as verified.
 *
 * @param readiness - The model's readiness, or undefined when it has none.
 * @returns The badge, or null when nothing is claimed.
 */
export function perfectBadge(readiness: ModelReadiness | undefined): ReadinessBadge | null {
  if (readiness?.is_perfect_verified === true) {
    return {
      label: "perfect",
      verified: true,
      title: "Verified: fresh facet receipts reach science S5 and the terminal silicon tier",
    };
  }
  if (readiness?.is_perfect === true) {
    return {
      label: "declared perfect · unverified",
      verified: false,
      title:
        "The descriptor's flags claim S5 and the terminal silicon tier; no fresh facet "
        + "receipts verify it",
    };
  }
  return null;
}

/**
 * Describe the verified tiers next to the declared ones.
 *
 * @param readiness - The model's readiness, or undefined when it has none.
 * @returns For example "verified S3 / H0", or null without a verification block.
 */
export function verifiedTiers(readiness: ModelReadiness | undefined): string | null {
  const verified = readiness?.verified;
  if (verified === undefined) {
    return null;
  }
  return `verified ${verified.science_label} / ${verified.silicon_label}`;
}
