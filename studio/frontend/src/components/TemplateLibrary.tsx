import { useEffect } from "react";
import { useStudioStore } from "../stores/studio";

export default function TemplateLibrary() {
  const { templates, selectedTemplate, loadTemplates, selectTemplate } =
    useStudioStore();

  useEffect(() => {
    loadTemplates().then(() => {
      useStudioStore.getState().runSimulation();
    });
  }, [loadTemplates]);

  return (
    <select
      value={selectedTemplate}
      onChange={(e) => selectTemplate(e.target.value)}
    >
      {templates.map((t) => (
        <option key={t.name} value={t.name}>
          {t.description}
        </option>
      ))}
    </select>
  );
}
