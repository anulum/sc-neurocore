// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — built federation artefact verifier

import { access, readFile, readdir } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const frontendRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const distRoot = join(frontendRoot, "dist");
const remoteEntryPath = join(distRoot, "remoteEntry.js");

async function filesBelow(directory, prefix = "") {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const relative = join(prefix, entry.name);
    if (entry.isDirectory()) {
      files.push(...await filesBelow(join(directory, entry.name), relative));
    } else {
      files.push(relative);
    }
  }
  return files;
}

function assertContract(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

await access(remoteEntryPath);
const remoteEntry = await readFile(remoteEntryPath, "utf8");
assertContract(remoteEntry.length > 0, "remoteEntry.js is empty");
assertContract(/export\{[^}]*\bas get\b[^}]*\bas init\b|export\{[^}]*\bas init\b[^}]*\bas get\b/.test(remoteEntry),
  "remoteEntry.js does not export the Module Federation get/init contract");

const entryImports = [...remoteEntry.matchAll(/from["']\.\/([^"']+)["']/g)].map((match) => match[1]);
assertContract(entryImports.length > 0, "remoteEntry.js has no local runtime import");
for (const importedPath of entryImports) {
  assertContract(!importedPath.includes(".."), `remoteEntry.js import escapes dist: ${importedPath}`);
  await access(join(distRoot, importedPath));
}

const files = await filesBelow(distRoot);
const textFiles = files.filter((file) => /\.(?:css|html|js|json|d\.ts)$/.test(file));
const builtText = (await Promise.all(
  textFiles.map((file) => readFile(join(distRoot, file), "utf8")),
)).join("\n");
assertContract(builtText.includes("sc_neurocore"), "built remote omits its federation identity");
assertContract(builtText.includes("SnnStudioPanel"), "built remote omits the SnnStudioPanel expose");
assertContract(!builtText.includes("demo_studio"), "built remote still contains the demo_studio identity");

const indexHtml = await readFile(join(distRoot, "index.html"), "utf8");
assertContract(indexHtml.includes('/studios/sc-neurocore/'),
  "standalone index does not use the /studios/sc-neurocore/ asset base");

console.log(`verified sc_neurocore federation build (${files.length} files)`);
