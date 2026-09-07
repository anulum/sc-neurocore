// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio development-preview shell labelling

/**
 * Canonical label shown on the Studio shell so lab/experimental builds are
 * never silently presented as production-validated product surfaces.
 */
export const DEVELOPMENT_PREVIEW_LABEL = "Development preview" as const;

/** Short honesty note paired with the banner label. */
export const DEVELOPMENT_PREVIEW_DETAIL =
  "Experimental lab Studio — not a production-validated release surface." as const;

/**
 * What the operator status says the deployment is.
 *
 * Any string, because the value arrives from the server and this build must
 * not decide that an unrecognised profile is impossible. `development` and
 * `production` are the two it can say something specific about; everything
 * else, including absent, is treated as unknown and shown the loud banner.
 */
export type StudioDeploymentProfile = string | null | undefined;

/**
 * Whether the shell must show the development-preview banner.
 *
 * The banner shows until an explicit release criterion says otherwise. A
 * production profile does not turn it off: the profile describes the
 * deployment's own defaults, not whether this interface has passed release
 * criteria, and a preview that presents itself as a validated product is the
 * failure this exists to prevent.
 *
 * The profile is therefore not read at all here -- it only changes the banner's
 * wording, in `developmentPreviewBannerModel`.
 *
 * @param _deploymentProfile - What the deployment says it is. Deliberately
 *   unused: see above.
 * @param options - Whether release criteria have passed.
 * @returns Whether to show the banner.
 */
export function shouldShowDevelopmentPreviewBanner(
  _deploymentProfile: StudioDeploymentProfile,
  options: { releaseValidated?: boolean } = {},
): boolean {
  if (options.releaseValidated === true) {
    return false;
  }
  return true;
}

/**
 * Build the banner's copy.
 *
 * @param deploymentProfile - What the deployment says it is.
 * @param options - Whether release criteria have passed.
 * @returns The label, the detail line, and whether to show them. A production
 *   profile gets an extra sentence saying the profile is production and the
 *   interface is still preview, because that is the pairing most likely to be
 *   misread.
 */
export function developmentPreviewBannerModel(
  deploymentProfile: StudioDeploymentProfile,
  options: { releaseValidated?: boolean } = {},
): { detail: string; label: string; visible: boolean } {
  const visible = shouldShowDevelopmentPreviewBanner(deploymentProfile, options);
  const profile =
    deploymentProfile === "production" || deploymentProfile === "development"
      ? deploymentProfile
      : "unknown";
  return {
    detail:
      profile === "production"
        ? `${DEVELOPMENT_PREVIEW_DETAIL} Operator profile reports production defaults, but this UI remains preview until release criteria pass.`
        : DEVELOPMENT_PREVIEW_DETAIL,
    label: DEVELOPMENT_PREVIEW_LABEL,
    visible,
  };
}
