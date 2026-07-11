# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio identity and browser-session routes

"""Authenticate browser sessions and administer persistent Studio identities."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.api.schemas import (
    StudioBrowserLoginRequest,
    StudioBrowserUserCreateRequest,
    StudioBrowserUserPasswordRotateRequest,
    StudioBrowserUserUpdateRequest,
    StudioIdentityServiceAccountUpdateRequest,
)
from sc_neurocore.studio.api.security import _studio_timestamp_utc
from sc_neurocore.studio.platform import (
    THROTTLED_BROWSER_LOGIN_REASON,
    AuditEvent,
    AuditSinkError,
    Principal,
    StudioIdentityAuthenticator,
    StudioIdentityLifecycleError,
    add_studio_browser_user_record,
    list_studio_browser_user_public_records,
    list_studio_identity_public_records,
    load_studio_identity_store,
    rotate_studio_browser_user_password,
    update_studio_browser_user_record,
    update_studio_identity_record,
)


def build_identity_router(context: StudioApiContext) -> APIRouter:
    """Build the identity and browser-session router over shared Studio runtime state."""
    router = APIRouter()
    app = context.app
    settings = context.settings
    studio_audit_sink = context.studio_audit_sink
    studio_browser_login_throttle = context.studio_browser_login_throttle
    studio_browser_session_manager = context.studio_browser_session_manager

    @router.post("/api/studio/auth/login")
    def api_studio_auth_login(
        login: StudioBrowserLoginRequest,
        request: Request,
    ) -> dict[str, list[str] | str]:
        """Authenticate a browser user and issue an expiring bearer session."""
        if context.studio_identity_authenticator is None:
            raise HTTPException(status_code=409, detail="identity_store_unavailable")
        throttle_decision = studio_browser_login_throttle.check(login.username)
        if not throttle_decision.allowed:
            request_id = getattr(request.state, "studio_request_id", None)
            try:
                studio_audit_sink.record(
                    AuditEvent(
                        action="studio.auth.login",
                        decision="deny",
                        principal_id=None,
                        reason=throttle_decision.reason or THROTTLED_BROWSER_LOGIN_REASON,
                        request_id=request_id if isinstance(request_id, str) else None,
                        route="/api/studio/auth/login",
                        timestamp_utc=_studio_timestamp_utc(),
                    )
                )
            except AuditSinkError as exc:
                raise HTTPException(status_code=503, detail="audit_append_failed") from exc
            headers = (
                {}
                if throttle_decision.retry_after_seconds is None
                else {"Retry-After": str(throttle_decision.retry_after_seconds)}
            )
            raise HTTPException(
                status_code=429,
                detail=throttle_decision.reason or THROTTLED_BROWSER_LOGIN_REASON,
                headers=headers,
            )
        identity_result = context.studio_identity_authenticator.authenticate_browser_user(
            login.username,
            login.password,
        )
        request_id = getattr(request.state, "studio_request_id", None)
        if identity_result.principal is None:
            throttle_after_failure = (
                studio_browser_login_throttle.record_failure(login.username)
                if identity_result.failure_reason == "invalid_browser_login"
                else None
            )
            reason = identity_result.failure_reason or "invalid_browser_login"
            if (
                throttle_after_failure is not None
                and not throttle_after_failure.allowed
                and throttle_after_failure.reason is not None
            ):
                reason = throttle_after_failure.reason
            try:
                studio_audit_sink.record(
                    AuditEvent(
                        action="studio.auth.login",
                        decision="deny",
                        principal_id=None,
                        reason=reason,
                        request_id=request_id if isinstance(request_id, str) else None,
                        route="/api/studio/auth/login",
                        timestamp_utc=_studio_timestamp_utc(),
                    )
                )
            except AuditSinkError as exc:
                raise HTTPException(status_code=503, detail="audit_append_failed") from exc
            if throttle_after_failure is not None and not throttle_after_failure.allowed:
                headers = (
                    {}
                    if throttle_after_failure.retry_after_seconds is None
                    else {"Retry-After": str(throttle_after_failure.retry_after_seconds)}
                )
                raise HTTPException(status_code=429, detail=reason, headers=headers)
            raise HTTPException(
                status_code=401,
                detail=reason,
            )
        studio_browser_login_throttle.record_success(login.username)
        issued = studio_browser_session_manager.issue(identity_result.principal)
        try:
            studio_audit_sink.record(
                AuditEvent(
                    action="studio.auth.login",
                    decision="allow",
                    principal_id=identity_result.principal.principal_id,
                    reason="authenticated",
                    request_id=request_id if isinstance(request_id, str) else None,
                    route="/api/studio/auth/login",
                    timestamp_utc=_studio_timestamp_utc(),
                )
            )
        except AuditSinkError as exc:
            raise HTTPException(status_code=503, detail="audit_append_failed") from exc
        return issued.to_public_dict()

    @router.get("/api/studio/auth/session")
    def api_studio_auth_session(request: Request) -> dict[str, bool | list[str] | str | None]:
        """Return the current browser bearer-session principal."""
        return studio_browser_session_manager.public_session(request.headers.get("authorization"))

    @router.post("/api/studio/auth/logout")
    def api_studio_auth_logout(request: Request) -> dict[str, bool]:
        """Revoke the current browser bearer session."""
        revoked = studio_browser_session_manager.revoke_authorization_header(
            request.headers.get("authorization")
        )
        actor = getattr(request.state, "studio_principal", None)
        request_id = getattr(request.state, "studio_request_id", None)
        try:
            studio_audit_sink.record(
                AuditEvent(
                    action="studio.auth.logout",
                    decision="allow",
                    principal_id=actor.principal_id if isinstance(actor, Principal) else None,
                    reason="revoked" if revoked else "not_found",
                    request_id=request_id if isinstance(request_id, str) else None,
                    route="/api/studio/auth/logout",
                    timestamp_utc=_studio_timestamp_utc(),
                )
            )
        except AuditSinkError as exc:
            raise HTTPException(status_code=503, detail="audit_append_failed") from exc
        return {"revoked": revoked}

    @router.get("/api/studio/identity/service-accounts")
    def api_studio_identity_service_accounts() -> dict[str, object]:
        """Return token-free persistent service accounts for administrators."""
        if settings.identity_file_path is None:
            raise HTTPException(status_code=409, detail="identity_store_unavailable")
        try:
            records = list_studio_identity_public_records(Path(settings.identity_file_path))
        except ValueError as exc:
            raise HTTPException(status_code=503, detail="identity_store_unhealthy") from exc
        return {
            "schema_version": "sc-neurocore.studio.identity.service-accounts.v1",
            "service_accounts": [record.to_public_dict() for record in records],
        }

    @router.get("/api/studio/identity/browser-users")
    def api_studio_identity_browser_users() -> dict[str, object]:
        """Return password-free persistent browser users for administrators."""
        if settings.identity_file_path is None:
            raise HTTPException(status_code=409, detail="identity_store_unavailable")
        try:
            records = list_studio_browser_user_public_records(Path(settings.identity_file_path))
        except ValueError as exc:
            raise HTTPException(status_code=503, detail="identity_store_unhealthy") from exc
        return {
            "browser_users": [record.to_public_dict() for record in records],
            "schema_version": "sc-neurocore.studio.identity.browser-users.v1",
        }

    @router.post("/api/studio/identity/browser-users")
    def api_create_studio_identity_browser_user(
        create: StudioBrowserUserCreateRequest,
        request: Request,
    ) -> dict[str, bool | list[str] | str | None]:
        """Create one persistent browser user for administrators."""
        if settings.identity_file_path is None:
            raise HTTPException(status_code=409, detail="identity_store_unavailable")
        identity_path = Path(settings.identity_file_path)
        try:
            created = add_studio_browser_user_record(
                identity_path,
                active=create.active,
                expires_at_utc=create.expires_at_utc,
                password=create.password,
                principal_id=create.principal_id,
                roles=create.roles,
                username=create.username,
            )
            context.studio_identity_authenticator = StudioIdentityAuthenticator(
                load_studio_identity_store(identity_path)
            )
            app.state.studio_identity_authenticator = context.studio_identity_authenticator
            actor = getattr(request.state, "studio_principal", None)
            request_id = getattr(request.state, "studio_request_id", None)
            studio_audit_sink.record(
                AuditEvent(
                    action="studio.identity.browser_user.create",
                    decision="allow",
                    principal_id=(actor.principal_id if isinstance(actor, Principal) else None),
                    reason=f"created:{created.username}",
                    request_id=request_id if isinstance(request_id, str) else None,
                    route="/api/studio/identity/browser-users",
                    timestamp_utc=_studio_timestamp_utc(),
                )
            )
        except ValueError as exc:
            detail = str(exc)
            status_code = 409 if "already exists" in detail else 422
            raise HTTPException(status_code=status_code, detail=detail) from exc
        except AuditSinkError as exc:
            raise HTTPException(status_code=503, detail="audit_append_failed") from exc
        return created.to_public_dict()

    @router.get("/api/studio/identity/browser-users/{username}")
    def api_studio_identity_browser_user(
        username: str,
    ) -> dict[str, bool | list[str] | str | None]:
        """Return one password-free persistent browser user for administrators."""
        if settings.identity_file_path is None:
            raise HTTPException(status_code=409, detail="identity_store_unavailable")
        try:
            records = list_studio_browser_user_public_records(Path(settings.identity_file_path))
        except ValueError as exc:
            raise HTTPException(status_code=503, detail="identity_store_unhealthy") from exc
        for record in records:
            if record.username == username:
                return record.to_public_dict()
        raise HTTPException(status_code=404, detail="identity_browser_user_not_found")

    @router.get("/api/studio/identity/service-accounts/{principal_id}")
    def api_studio_identity_service_account(
        principal_id: str,
    ) -> dict[str, bool | list[str] | str | None]:
        """Return one token-free persistent service account for administrators."""
        if settings.identity_file_path is None:
            raise HTTPException(status_code=409, detail="identity_store_unavailable")
        try:
            records = list_studio_identity_public_records(Path(settings.identity_file_path))
        except ValueError as exc:
            raise HTTPException(status_code=503, detail="identity_store_unhealthy") from exc
        for record in records:
            if record.principal_id == principal_id:
                return record.to_public_dict()
        raise HTTPException(status_code=404, detail="identity_service_account_not_found")

    @router.patch("/api/studio/identity/service-accounts/{principal_id}")
    def api_update_studio_identity_service_account(
        principal_id: str,
        update: StudioIdentityServiceAccountUpdateRequest,
        request: Request,
    ) -> dict[str, bool | list[str] | str | None]:
        """Update persistent service-account role metadata for administrators."""
        if settings.identity_file_path is None:
            raise HTTPException(status_code=409, detail="identity_store_unavailable")
        identity_path = Path(settings.identity_file_path)
        try:
            updated = update_studio_identity_record(
                identity_path,
                active=update.active,
                expires_at_utc=update.expires_at_utc,
                principal_id=principal_id,
                roles=update.roles,
            )
            context.studio_identity_authenticator = StudioIdentityAuthenticator(
                load_studio_identity_store(identity_path)
            )
            app.state.studio_identity_authenticator = context.studio_identity_authenticator
            actor = getattr(request.state, "studio_principal", None)
            request_id = getattr(request.state, "studio_request_id", None)
            studio_audit_sink.record(
                AuditEvent(
                    action="studio.identity.service_account.update",
                    decision="allow",
                    principal_id=(actor.principal_id if isinstance(actor, Principal) else None),
                    reason=f"updated:{updated.principal_id}",
                    request_id=request_id if isinstance(request_id, str) else None,
                    route="/api/studio/identity/service-accounts/{principal_id}",
                    timestamp_utc=_studio_timestamp_utc(),
                )
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="identity_service_account_not_found",
            ) from exc
        except StudioIdentityLifecycleError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except AuditSinkError as exc:
            raise HTTPException(status_code=503, detail="audit_append_failed") from exc
        return updated.to_public_dict()

    @router.patch("/api/studio/identity/browser-users/{username}")
    def api_update_studio_identity_browser_user(
        username: str,
        update: StudioBrowserUserUpdateRequest,
        request: Request,
    ) -> dict[str, bool | list[str] | str | None]:
        """Update persistent browser-user role metadata for administrators."""
        if settings.identity_file_path is None:
            raise HTTPException(status_code=409, detail="identity_store_unavailable")
        identity_path = Path(settings.identity_file_path)
        try:
            updated = update_studio_browser_user_record(
                identity_path,
                active=update.active,
                expires_at_utc=update.expires_at_utc,
                roles=update.roles,
                username=username,
            )
            context.studio_identity_authenticator = StudioIdentityAuthenticator(
                load_studio_identity_store(identity_path)
            )
            app.state.studio_identity_authenticator = context.studio_identity_authenticator
            actor = getattr(request.state, "studio_principal", None)
            request_id = getattr(request.state, "studio_request_id", None)
            studio_audit_sink.record(
                AuditEvent(
                    action="studio.identity.browser_user.update",
                    decision="allow",
                    principal_id=(actor.principal_id if isinstance(actor, Principal) else None),
                    reason=f"updated:{updated.username}",
                    request_id=request_id if isinstance(request_id, str) else None,
                    route="/api/studio/identity/browser-users/{username}",
                    timestamp_utc=_studio_timestamp_utc(),
                )
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="identity_browser_user_not_found",
            ) from exc
        except StudioIdentityLifecycleError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except AuditSinkError as exc:
            raise HTTPException(status_code=503, detail="audit_append_failed") from exc
        return updated.to_public_dict()

    @router.post("/api/studio/identity/browser-users/{username}/password")
    def api_rotate_studio_identity_browser_user_password(
        username: str,
        update: StudioBrowserUserPasswordRotateRequest,
        request: Request,
    ) -> dict[str, bool | list[str] | str | None]:
        """Rotate one persistent browser-user password verifier."""
        if settings.identity_file_path is None:
            raise HTTPException(status_code=409, detail="identity_store_unavailable")
        identity_path = Path(settings.identity_file_path)
        try:
            updated = rotate_studio_browser_user_password(
                identity_path,
                password=update.password,
                username=username,
            )
            context.studio_identity_authenticator = StudioIdentityAuthenticator(
                load_studio_identity_store(identity_path)
            )
            app.state.studio_identity_authenticator = context.studio_identity_authenticator
            revoked_sessions = studio_browser_session_manager.revoke_principal(updated.principal_id)
            studio_browser_login_throttle.record_success(username)
            actor = getattr(request.state, "studio_principal", None)
            request_id = getattr(request.state, "studio_request_id", None)
            studio_audit_sink.record(
                AuditEvent(
                    action="studio.identity.browser_user.password.rotate",
                    decision="allow",
                    principal_id=(actor.principal_id if isinstance(actor, Principal) else None),
                    reason=f"rotated:{updated.username}:sessions_revoked:{revoked_sessions}",
                    request_id=request_id if isinstance(request_id, str) else None,
                    route="/api/studio/identity/browser-users/{username}/password",
                    timestamp_utc=_studio_timestamp_utc(),
                )
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="identity_browser_user_not_found",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except AuditSinkError as exc:
            raise HTTPException(status_code=503, detail="audit_append_failed") from exc
        return updated.to_public_dict()

    return router
