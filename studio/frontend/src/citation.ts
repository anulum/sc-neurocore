// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import type { ModelProvenance } from "./api/client";

/**
 * Build the citation for a model's source publication.
 *
 * @param provenance - What the catalogue records about where the model came
 *   from, if anything.
 * @param modelName - The model's name, used when the provenance has no title.
 * @returns The citation, or an empty string when there is nothing citeable.
 *   Callers hide the "How to cite" affordance on an empty string rather than
 *   showing an incomplete reference, which would be worse than none.
 */
export function formatCitation(
  provenance: ModelProvenance | null,
  modelName: string,
): string {
  if (!provenance || (!provenance.doi && provenance.authors.length === 0)) {
    return "";
  }
  const parts: string[] = [];
  if (provenance.authors.length > 0) {
    const authors = provenance.authors.join(", ");
    parts.push(provenance.year ? `${authors} (${provenance.year}).` : `${authors}.`);
  }
  if (provenance.paper_title) {
    parts.push(`${provenance.paper_title}.`);
  }
  const link = provenance.doi
    ? `https://doi.org/${provenance.doi}`
    : provenance.url;
  if (link) {
    parts.push(link);
  }
  parts.push(`Implemented as ${modelName} in SC-NeuroCore.`);
  return parts.join(" ");
}
