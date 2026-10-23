// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Reproducible federation declaration archive

/** Remove build-clock timestamps from the producer's complete type archive. */
import AdmZip from "adm-zip";
import { resolve } from "node:path";

const archiveName = process.argv[2];
if (!archiveName) {
  throw new Error("Provide the generated federation types ZIP path.");
}
const archivePath = resolve(archiveName);
const archive = new AdmZip(archivePath);
for (const entry of archive.getEntries()) {
  // ZIP stores local calendar fields: local midnight encodes the same DOS
  // date in every timezone. Preserve each entry's body, attributes and comment.
  entry.header.time = new Date(1980, 0, 1);
}
archive.writeZip(archivePath);
