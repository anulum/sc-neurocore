// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Review comments on the open workspace revision

/**
 * Discuss the workspace revision that is open, and only that revision.
 *
 * Comments are bound on the server to the revision and its digest; this panel
 * shows them for the revision the editor loaded or last saved, says when a
 * revision is no longer what was reviewed, and threads replies under the
 * comment they answer.
 */

import { useCallback, useEffect, useState, type CSSProperties } from "react";

import { addReviewComment, listReviewComments, type ReviewComment } from "../api/projectApi";
import { useStudioStore } from "../stores/studio";

const button: CSSProperties = {
  background: "transparent", border: "1px solid var(--control-border)", borderRadius: 3,
  color: "var(--text-secondary)", cursor: "pointer", fontSize: 10, padding: "2px 8px",
};

const STATUS_WORDS: Record<ReviewComment["revision_status"], string> = {
  matches: "",
  changed: " — the revision no longer matches what was reviewed",
  missing: " — the reviewed revision is gone",
};

/**
 * The review panel.
 *
 * @returns The panel.
 */
export default function ReviewPanel() {
  const { projectRevision } = useStudioStore();
  const [comments, setComments] = useState<ReviewComment[]>([]);
  const [draft, setDraft] = useState("");
  const [replyTo, setReplyTo] = useState<ReviewComment | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (projectRevision === null) return;
    try {
      const listed = await listReviewComments(projectRevision.name, projectRevision.revision);
      setComments(listed.comments);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }, [projectRevision]);

  useEffect(() => { void refresh(); }, [refresh]);

  if (projectRevision === null) {
    return (
      <section aria-label="Review" style={{ padding: "8px 12px", fontSize: 10 }}>
        <h2 style={{ fontSize: 13, margin: 0 }}>Review</h2>
        <p>Save or open a project: comments are made on one saved revision.</p>
      </section>
    );
  }

  const roots = comments.filter((comment) => comment.reply_to === null);
  const repliesTo = (id: string) => comments.filter((comment) => comment.reply_to === id);

  /** Send the draft as a comment, or as a reply. */
  async function submit(): Promise<void> {
    if (projectRevision === null) return;
    try {
      await addReviewComment(projectRevision.name, projectRevision.revision, draft, replyTo?.comment_id ?? null);
      setDraft("");
      setReplyTo(null);
      setMessage(null);
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  /**
   * One comment and the replies under it.
   *
   * @param props - The comment.
   * @returns The list item.
   */
  function Comment({ comment }: { comment: ReviewComment }) {
    return (
      <li>
        <strong>{comment.author}</strong>{" "}
        <span style={{ color: "var(--text-muted)" }}>
          {new Date(comment.created_at * 1000).toISOString().replace("T", " ").slice(0, 19)}
          {STATUS_WORDS[comment.revision_status]}
        </span>
        <p style={{ margin: "2px 0" }}>{comment.body}</p>
        <button type="button" style={button} aria-label={`Reply to ${comment.author}: ${comment.body.slice(0, 40)}`}
          onClick={() => { setReplyTo(comment); }}>Reply</button>
        {repliesTo(comment.comment_id).length > 0 && (
          <ul style={{ paddingLeft: 16 }}>
            {repliesTo(comment.comment_id).map((reply) => <Comment key={reply.comment_id} comment={reply} />)}
          </ul>
        )}
      </li>
    );
  }

  return (
    <section aria-label="Review" style={{ flex: 1, overflow: "auto", padding: "8px 12px", fontSize: 10, display: "flex", flexDirection: "column", gap: 6 }}>
      <h2 style={{ fontSize: 13, margin: 0 }}>
        Review of {projectRevision.name}, revision {projectRevision.revision}
      </h2>
      {roots.length === 0 ? (
        <p style={{ margin: 0 }}>No comments on this revision yet.</p>
      ) : (
        <ul aria-label="Comments" style={{ margin: 0, paddingLeft: 16 }}>
          {roots.map((comment) => <Comment key={comment.comment_id} comment={comment} />)}
        </ul>
      )}
      <label htmlFor="review-draft">
        {replyTo === null ? "Comment on this revision" : `Reply to ${replyTo.author}`}
      </label>
      <textarea id="review-draft" rows={3} value={draft} onChange={(event) => { setDraft(event.target.value); }} />
      <div style={{ display: "flex", gap: 6 }}>
        <button type="button" style={button} onClick={() => { void submit(); }}>
          {replyTo === null ? "Add comment" : "Add reply"}
        </button>
        {replyTo !== null && (
          <button type="button" style={button} onClick={() => { setReplyTo(null); }}>Cancel reply</button>
        )}
      </div>
      {message !== null && <p role="status" style={{ margin: 0 }}>{message}</p>}
    </section>
  );
}
