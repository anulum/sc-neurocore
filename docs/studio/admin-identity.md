# Admin identity & audit access

The Studio admin, audit, and identity-management endpoints are disabled until an
identity store is configured. With no store, those routes return
`409 identity_store_unavailable` — a deliberate secure default so a fresh Studio
exposes no privileged surface. Read-only domain features (model catalogue,
behaviour facet, simulation, characterisation, DCLS, benchmark databank) work
without any identity store.

This page covers enabling the privileged surface for a real deployment.

## What is gated

These endpoints require a configured identity store (otherwise `409`):

- `GET /api/studio/identity/browser-users`
- `GET /api/studio/identity/service-accounts` and `…/{principal_id}`
- `GET /api/studio/audit/export`
- `GET /api/studio/audit/quarantine/export`

and the write/admin operations on the same prefixes.

## 1. Bootstrap an admin identity

Create the identity file and a first `studio.admin` service account with the CLI:

```bash
sc-neurocore studio-bootstrap-admin --identity-file /etc/sc-neurocore/studio-identities.json
```

The command prints a JSON result (`schema_version: sc-neurocore.studio.identity.v1`)
containing the one-time `bearer_token`, the `principal_id` (default
`svc-studio-admin`), the granted `roles` (`studio.admin`, `studio.viewer`), and the
`token_sha256`. The file is written with hardened `0600` permissions. **Capture the
`bearer_token` immediately — only its SHA-256 is stored, so it cannot be recovered.**

Useful flags:

- `--principal-id <id>` — name the service account (default `svc-studio-admin`).
- `--roles <role> …` — override the granted roles.
- `--expires-at-utc <ISO-8601>` — set an expiry for the bootstrap identity.
- `--overwrite` — atomically replace an existing identity file.

## 2. Point the Studio at the store

Set the environment variable before launching the Studio:

```bash
export SC_NEUROCORE_STUDIO_IDENTITY_FILE=/etc/sc-neurocore/studio-identities.json
```

When this is set to a readable store, the gated endpoints return `200` and serve
the configured principals; an unreadable or malformed store returns
`503 identity_store_unhealthy` instead.

### The store stays private, and one process serves it

The store holds credential hashes, so the Studio keeps it owner-only. Loading
refuses a store another account owns, since that account could change who may
sign in; a store its owner left readable by group or others is narrowed to
`0600` before it is read, and the widening is logged, because its hashes were
exposed until then. Every write leaves the file `0600`, whatever mode it had.
POSIX modes do not apply on Windows, where the file's access-control list
governs.

Browser sessions and login throttles are kept in the API process. A second API
process serving the same store — a second server worker, or a second server
started on the same files — would keep its own: a session signed out in one
would stay valid in the other, and the login-attempt limit would multiply. The
first API process to open a store therefore holds it (an exclusive lock on
`<store>.api-lock` beside it) for its lifetime, and another process that tries
is refused at startup with the reason. The supported deployment is one lab
process; `sc-neurocore studio` starts exactly one.

## 3. Authenticate requests

Send the captured token as a bearer credential:

```bash
curl -H "Authorization: Bearer <bearer_token>" \
  http://127.0.0.1:8000/api/studio/identity/service-accounts
```

## Route-policy enforcement

Setting `SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES=1` makes the Studio reject any
request whose route lacks a registered policy, and enforces the per-route
visibility (`public`, `authenticated`, `admin`) declared in the route-policy
registry. Leave it unset for local development; enable it for shared deployments.

## Operator status

`GET /api/studio/operator/status` reports the live route-policy counts
(`public_count`, `authenticated_count`, `admin_count`, `total_count`) and the
identity-store health, which is the quickest way to confirm a deployment is wired
correctly.
