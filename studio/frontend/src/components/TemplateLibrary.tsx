import { useEffect } from "react";
import { useStudioStore } from "../stores/studio";

export default function TemplateLibrary() {
  const { templates, selectedTemplate, loadTemplates, selectTemplate } =
    useStudioStore();

  useEffect(() => {
    loadTemplates();
  }, [loadTemplates]);

  return (
    <select
      value={selectedTemplate}
      onChange={(e) => selectTemplate(e.target.value)}
      style={{ fontSize: 14, padding: "4px 8px" }}
    >
      {templates.map((t) => (
        <option key={t.name} value={t.name}>
          {t.description}
        </option>
      ))}
    </select>
  );
}
