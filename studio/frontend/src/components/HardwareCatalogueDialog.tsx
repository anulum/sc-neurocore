// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Tool-independent hardware catalogue dialog

import { useCallback } from "react";
import HardwareCatalogue from "./HardwareCatalogue";

/**
 * Open registration discovery independently of optional synthesis tools.
 *
 * Native modal behaviour owns focus, Escape and close-button submission.
 * Key events stay inside the dialog so workbench shortcuts cannot start jobs.
 *
 * @param props - Existing target/source state and close/selection actions.
 * @returns A native modal containing the same hardware catalogue as synthesis.
 */
export default function HardwareCatalogueDialog({ selectedTarget, modelMode, onSelectTarget, onClose }: {
  selectedTarget: string;
  modelMode: boolean;
  onSelectTarget: (target: string) => void;
  onClose: () => void;
}) {
  const open = useCallback((dialog: HTMLDialogElement | null) => {
    if (dialog !== null) dialog.showModal();
  }, []);
  return (
    <dialog ref={open} aria-labelledby="hardware-catalogue-title" className="text-prompt"
      onClose={onClose} onKeyDown={(event) => { event.stopPropagation(); }}
      style={{ width: "min(960px, 95vw)", maxHeight: "85vh", overflow: "auto" }}>
      <h2 id="hardware-catalogue-title">Hardware catalogue</h2>
      <form method="dialog">
        <button type="submit" className="btn btn--ghost">Close hardware catalogue</button>
      </form>
      <HardwareCatalogue selectedTarget={selectedTarget} modelMode={modelMode}
        onSelectTarget={onSelectTarget} defaultExpanded />
    </dialog>
  );
}
