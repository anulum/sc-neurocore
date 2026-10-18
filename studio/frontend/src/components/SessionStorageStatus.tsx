// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Browser session persistence feedback

import type { StudioSessionWriteResult } from "../studioSavedSessions";

/**
 * Distinguish page-only session changes from a successful browser write.
 *
 * @param props - The last attempt to persist the session list, if any.
 * @returns An accessible write result, or nothing before the first attempt.
 */
export default function SessionStorageStatus({ result }: {
  result: StudioSessionWriteResult | null;
}) {
  if (result === null) return null;
  const saved = result.status === "ok";
  return (
    <p role={saved ? "status" : "alert"} data-testid="session-storage-status"
      style={{ fontSize: "var(--fs-body)", color: saved ? "var(--text-secondary)" : "var(--warning)" }}>
      {saved ? "Session changes saved in this browser." : (
        <>{result.message} These session changes are kept in this page only.
          {" "}Reloading may restore the last saved list. Save a server project
          {" "}to keep a separate copy of the current experiment.</>
      )}
    </p>
  );
}
