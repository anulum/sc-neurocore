import { useEffect } from "react";
import { useStudioStore } from "../stores/studio";

export default function TemplateLibrary() {
  const { templates, loadTemplates, selectTemplate } = useStudioStore();

  useEffect(() => { loadTemplates(); }, [loadTemplates]);

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
