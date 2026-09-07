// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { useEffect } from "react";
import { useStudioStore } from "../stores/studio";

/**
 * The editable ODE templates a session can start from.
 *
 * @returns The panel.
 */
export default function TemplateLibrary() {
  const { templates, loadTemplates, selectTemplate } = useStudioStore();

  useEffect(() => { void loadTemplates(); }, [loadTemplates]);

  return (
    <select
      defaultValue=""
      onChange={(e) => { if (e.target.value) selectTemplate(e.target.value); }}
    >
      <option value="" disabled>ODE templates...</option>
      {templates.map((t) => (
        <option key={t.name} value={t.name}>{t.description}</option>
      ))}
    </select>
  );
}
