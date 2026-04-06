# Incident: Co-Authored-By wrong email

**Date:** 2026-04-05T03:00
**Agent:** Arcane Sapience (Claude)
**Severity:** Tier 1 violation

## What happened

5 commits used `Co-Authored-By: Arcane Sapience <noreply@anthropic.com>` instead of the canonical `Co-Authored-By: Arcane Sapience <protoscience@anulum.li>`.

## Root cause

Session continued from compacted context. The Co-Authored-By email was carried over from the compaction summary which had the wrong email. Agent did not re-verify against SHARED_CONTEXT.md before first commit.

## Which defence layer failed

L2 (Agent Rules) — PRE-COMMIT AUDIT was not performed before the first commit. The rule requires `git diff --cached` review and Co-Authored-By verification before every commit.

## Corrective actions

- Applied `git filter-branch` to rewrite all 5 commits with correct email
- Verified all subsequent commits use correct email

## Lesson learned

Always re-read SHARED_CONTEXT.md Co-Authored-By line before first commit in any session, especially after context compaction.

## Prevention verification

All 11 commits in session verified: `Co-Authored-By: Arcane Sapience <protoscience@anulum.li>` — confirmed.
