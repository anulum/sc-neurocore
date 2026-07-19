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

export type StudioDeploymentProfile = "development" | "production" | string | null | undefined;

/**
 * Whether the shell must show the development-preview banner.
 *
 * Production profile still shows the banner until an explicit release criterion
 * flips ``releaseValidated`` (default false). Missing operator status is treated
 * as non-production (safe default: loud preview).
 */
export function shouldShowDevelopmentPreviewBanner(
  _deploymentProfile: StudioDeploymentProfile,
  options: { releaseValidated?: boolean } = {},
): boolean {
  if (options.releaseValidated === true) {
    return false;
  }
  // Always show until releaseValidated; profile only changes tone in banner copy.
  void _deploymentProfile;
  return true;
}

/**
 * Build the banner copy for the current deployment profile.
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
