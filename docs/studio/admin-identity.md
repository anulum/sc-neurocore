# Admin identity & audit access

The Studio admin, audit, and identity-management endpoints are disabled until an
identity store is configured. With no store, those routes return
`409 identity_store_unavailable` — a deliberate secure default so a fresh Studio
exposes no privileged surface. Read-only domain features (model catalogue,
behaviour facet, simulation, characterisation, DCLS, benchmark databank) work
without any identity store.

The supported lab configuration serves one operator on loopback with one API
process, an identity store, enforced route policies and a rotating audit log.
It works offline and requires no cloud login. Unauthenticated local development
is a separate, unprotected configuration.

The built frontend includes Monaco's editor assets and license under `vs/`;
the standalone and federated entry points load that local runtime. The profile
allows its local scripts and worker assets through Content-Security-Policy.

Shared multi-process sessions, per-workspace multi-user job access, OIDC, and
LAN or VPN exposure with TLS termination are outside this lab profile. Those
deployments require a separately reviewed operating contract.

Identity mutations return `422` with the identity contract's validation message.
Unexpected validation faults return `Studio identity request could not be
validated.`; Python conversion or codec diagnostics are never returned. Duplicate
browser usernames return `409`, as do updates that would remove the last active,
unexpired `studio.admin` principal. Validation refusals leave the identity records
unchanged.

Browser usernames and passwords support Unicode. New and rotated passwords
require at least 12 characters; login can still verify an existing shorter
password so it can be rotated. Browser principals must differ from service
principals. Schema errors never echo credential fields or their input context.
Expiry timestamps must include
a timezone and convert to a UTC date within years 1–9999; an offset that moves
an otherwise valid timestamp outside that range returns `422`.

## What is gated

These endpoints require a configured identity store (otherwise `409`):

- `GET /api/studio/identity/browser-users`
- `GET /api/studio/identity/service-accounts` and `…/{principal_id}`
- `GET /api/studio/audit/export`
- `GET /api/studio/audit/quarantine/export`

and the write/admin operations on the same prefixes.

## 1. Bootstrap an admin identity

Create a private state directory and the first `studio.admin` service account.
Run these commands as the account that will run Studio:

```bash
studio_state="${XDG_STATE_HOME:-$HOME/.local/state}/sc-neurocore-studio"
install -d -m 700 "$studio_state" "$studio_state/jobs" "$studio_state/audit"
sc-neurocore studio-bootstrap-admin --identity-file "$studio_state/identities.json"
```

The command prints a JSON result (`schema_version: sc-neurocore.studio.identity.v1`)
containing the one-time `bearer_token`, the `principal_id` (default
`svc-studio-admin`), the granted `roles` (`studio.admin`, `studio.viewer`), and the
`token_sha256`. The file is written with hardened `0600` permissions. **Capture the
`bearer_token` immediately — only its SHA-256 is stored, so it cannot be recovered.**

Useful flags:

- `--principal-id <id>` — name the service account (default `svc-studio-admin`).
- `--role <role>` — override the granted roles; repeat for each role.
- `--expires-at-utc <ISO-8601>` — set an expiry for the bootstrap identity.
- `--allow-overwrite` — atomically replace an existing identity file.

For the browser login form, add a separate operator identity before starting
the server. Supply its password from your secret manager through standard
input; keep the service account's bearer token separate:

```bash
printf '%s\n' "${STUDIO_OPERATOR_PASSWORD:?Load the operator password from your secret manager}" | \
  sc-neurocore studio-add-browser-user \
    --identity-file "$studio_state/identities.json" \
    --username operator --principal-id human-operator \
    --role studio.admin --role studio.viewer --password-stdin
```

## 2. Point the Studio at the store

Configure the complete loopback lab profile before launching Studio:

```bash
export SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE=production
export SC_NEUROCORE_STUDIO_ALLOWED_HOSTS=127.0.0.1,localhost
export SC_NEUROCORE_STUDIO_CORS_ORIGINS=http://127.0.0.1:8001,http://localhost:8001
export SC_NEUROCORE_STUDIO_WEBSOCKET_ALLOWED_ORIGINS="$SC_NEUROCORE_STUDIO_CORS_ORIGINS"
export SC_NEUROCORE_STUDIO_IDENTITY_FILE="$studio_state/identities.json"
export SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES=true
export SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL=false
export SC_NEUROCORE_STUDIO_JOB_ROOT="$studio_state/jobs"
export SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH="$studio_state/audit/events.jsonl"
export SC_NEUROCORE_STUDIO_AUDIT_ROTATION_BYTES=104857600
export SC_NEUROCORE_STUDIO_AUDIT_RETAINED_FILES=14
export SC_NEUROCORE_STUDIO_BENCHMARK_DATABANK="$studio_state/contributions.jsonl"
sc-neurocore studio --port 8001
```

An identity file requires route-policy enforcement and disables development
header principals. Inconsistent combinations are refused when settings are
created, before the application starts. With a readable store, the gated
endpoints require a bearer identity with the route's roles; merely configuring
the file does not grant access. An unreadable or malformed store prevents
startup; a later store read failure returns `503 identity_store_unhealthy`.

### The store stays private, and one process serves it

The parent directory must belong to the running account and must not be
writable by group or other accounts. Symbolic-link identity files are refused;
parent aliases are resolved. A file is opened once, without following a final
symlink where the operating system supports that flag, and owner and mode are
checked on that same descriptor.

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
`<store>.api-lock` beside its resolved path, plus a device/inode hold) for its lifetime, and another process that tries
is refused at startup with the reason. The lock file is created by the
account running Studio, so the store's directory must be writable by it; when
the lock cannot be created or opened, startup is refused with that reason,
never reported as another process. The supported deployment is one lab
process; `sc-neurocore studio` starts exactly one.

CLI and API changes hold the same writer lock across the complete read,
validation and atomic replacement. API changes also reload authentication and
revoke affected sessions before the update returns. **Restart the API after an
offline CLI change**, including bootstrap replacement and browser-user
provisioning. The CLI prints this reminder. An existing process uses its loaded
identities until that restart; old service tokens stop working after restart.

All state-changing HTTP requests require an allowed `Origin`, or no `Origin`
and `Sec-Fetch-Site: same-origin` or `none`. JSON routes require
`Content-Type: application/json`; a charset parameter is permitted. This also
applies to unprotected development. For a custom CLI port, both explicit CORS and WebSocket origin lists
must contain its loopback URL; otherwise startup refuses. When origins are
unset, the CLI adds its chosen loopback port. For example, native JSON clients
send `Sec-Fetch-Site: none` explicitly alongside the bearer and content type.

Every failed browser login has the same `401 invalid_browser_login` response,
including unknown, disabled and expired users. Unknown users run a default-cost
dummy password verifier. Each attempt completes the larger of the current
default and the largest stored PBKDF2 iteration count. Older digests retain
their password acceptance while verification completes that same hash-work
floor; rotation writes a fresh verifier. Failed attempts are reserved before verification and
bounded per username and directly connected client. Proxy forwarding headers
do not supply that client identity. The registry retains at most 1024 keys by
default (`SC_NEUROCORE_STUDIO_BROWSER_LOGIN_MAX_BUCKETS`). Live failure, lockout
and in-flight state is never evicted for a new key. If every slot is protected,
new keys, including legitimate users without an existing bucket, receive `429`
until a slot expires or its failure state is cleared.
Password rotation clears that user's failure state.

Lab uses the production runtime profile. Interactive API documentation at
`/docs` and `/redoc` is disabled in that profile, so it cannot load Swagger or
ReDoc from an external CDN. The JSON schema at `/openapi.json` remains available.
Development retains the interactive documentation. A launch without a built UI
opens the schema in lab/production, or the interactive docs in development.

Benchmark contributions require an authenticated identity under enforcement.
The databank defaults to 16 MiB and 10,000 records; set
`SC_NEUROCORE_STUDIO_BENCHMARK_MAX_BYTES` and
`SC_NEUROCORE_STUDIO_BENCHMARK_MAX_RECORDS` to change those budgets. A record
over its individual byte budget returns `413`; a full store returns `507`
without changing existing bytes. Databank reads accept `limit` (1–1000, default
100) and `offset`; the leaderboard counts and orders the returned page.

## 3. Authenticate requests

Updating a browser user's roles, active state or expiry through the
administration API revokes every existing browser session for that principal.
The user must sign in again; the next login
uses the updated account state. Password rotation also revokes those sessions.

Send the captured token as a bearer credential:

```bash
curl -H "Authorization: Bearer <bearer_token>" \
  http://127.0.0.1:8001/api/studio/identity/service-accounts
```

## Route-policy enforcement

Setting `SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES=1` makes the Studio reject any
request whose route lacks a registered policy, and enforces the per-route
visibility (`public`, `authenticated`, `admin`) declared in the route-policy
registry. The lab profile requires it. Leave it unset only for unprotected
development without an identity store.

## Operator status

`GET /api/studio/operator/status` reports the live route-policy counts
(`public_count`, `authenticated_count`, `admin_count`, `total_count`) and the
identity-store health, which is the quickest way to confirm a deployment is wired
correctly.
