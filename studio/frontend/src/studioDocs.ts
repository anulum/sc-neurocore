// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — where a capability's documentation is published

/**
 * Links from the Studio to its documentation.
 *
 * Capabilities name their page as a repository path such as
 * `docs/studio/network-canvas.md`. The Studio linked to `/<that path>` on its
 * own server, which serves no documentation: every such link returned 404, on
 * the development server and on an installed Studio alike. The pages are
 * published with MkDocs (directory URLs) at {@link DOCS_SITE}.
 */

/** The published documentation site; equal to `site_url` in `mkdocs.yml`. */
export const DOCS_SITE = "https://anulum.github.io/sc-neurocore/";

/**
 * The published URL of one documentation page.
 *
 * @param docsPath - The page's repository path, `docs/…/<page>.md`.
 * @returns Its URL on {@link DOCS_SITE}: `index.md` maps to its directory and
 *   any other page to `<page>/`, as MkDocs directory URLs do; a path outside
 *   `docs/` or not ending in `.md` maps to the site's front page.
 */
export function publishedDocsUrl(docsPath: string): string {
  const match = /^docs\/(.+)\.md$/.exec(docsPath.trim());
  const page = match?.[1];
  if (page === undefined) return DOCS_SITE;
  const route = page === "index" ? "" : page.endsWith("/index") ? page.slice(0, -"index".length) : `${page}/`;
  return `${DOCS_SITE}${route}`;
}
