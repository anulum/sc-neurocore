// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — the workbench view switcher: grouped tabs with the ARIA tabs pattern

import { useRef } from "react";
import type { KeyboardEvent } from "react";

import type { ViewTab } from "../stores/studio";
import { nextViewTabIndex, viewTabId, VIEW_PANEL_ID } from "../viewTabs";
import type { ViewTabGroup } from "../viewTabs";

/** What the switcher needs to know about one view's availability. */
export interface ViewTabAvailability {
  available: boolean;
  /** Why the view is unavailable, or what it offers; shown as its tooltip. */
  message: string;
}

/**
 * The view switcher.
 *
 * One `tablist` of real tabs: the selected view is announced as selected,
 * only it sits in the Tab order, and the arrow keys, Home and End move
 * between the views (WAI-ARIA tabs pattern, automatic activation). Each group
 * name is shown before its first view and is read as part of that view's
 * description, so a reader hears where they are.
 *
 * @param props - The groups to show, the active view, each view's
 *   availability, and what to do when a view is chosen.
 * @returns The switcher.
 */
export default function ViewTabs({
  active,
  availability,
  groups,
  onSelect,
}: {
  active: ViewTab;
  availability: (view: ViewTab) => ViewTabAvailability;
  groups: ViewTabGroup[];
  onSelect: (view: ViewTab) => void;
}) {
  const tabs = useRef(new Map<ViewTab, HTMLButtonElement>());
  const ordered = groups.flatMap((group) => group.views.map((entry) => entry.view));
  const enabled = ordered.map((view) => availability(view).available);
  const activeIsListed = ordered.includes(active);

  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>, view: ViewTab) => {
    const target = nextViewTabIndex(event.key, ordered.indexOf(view), enabled);
    if (target === null) return;
    event.preventDefault();
    const next = ordered[target];
    if (next === undefined) return;
    tabs.current.get(next)?.focus();
    onSelect(next);
  };

  return (
    <nav className="view-tabs" aria-label="Workbench views">
      <div className="view-tabs-list" role="tablist" aria-label="Workbench views">
        {groups.map((group) => {
          const groupId = `view-group-${group.label.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
          return (
          <div className="view-tabs-group" key={group.label}>
            <span className="view-tabs-group-label" id={groupId} aria-hidden="true">
              {group.label}
            </span>
            {group.views.map((entry, index) => {
              const state = availability(entry.view);
              const selected = entry.view === active;
              // Exactly one tab is in the Tab order: the selected one, or the
              // first enabled tab when the active view is not listed.
              const focusable = activeIsListed
                ? selected
                : entry.view === ordered[enabled.indexOf(true)];
              return (
                <button
                  key={entry.view}
                  ref={(element) => {
                    if (element) tabs.current.set(entry.view, element);
                    else tabs.current.delete(entry.view);
                  }}
                  type="button"
                  role="tab"
                  id={viewTabId(entry.view)}
                  className="view-tab"
                  aria-selected={selected}
                  aria-controls={VIEW_PANEL_ID}
                  aria-describedby={index === 0 ? groupId : undefined}
                  disabled={!state.available}
                  tabIndex={focusable ? 0 : -1}
                  title={state.message}
                  onClick={() => { onSelect(entry.view); }}
                  onKeyDown={(event) => { onKeyDown(event, entry.view); }}
                >
                  {entry.label}
                </button>
              );
            })}
          </div>
          );
        })}
      </div>
    </nav>
  );
}
