// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import type { SyntheticEvent } from "react";

import type { StudioAuthSession } from "../api/client";
import { formText } from "../adminFormParsers";
import { useStudioStore } from "../stores/studio";

/**
 * Sign-in, wired to the store.
 *
 * A container over `AuthControlView`, which takes its actions as props.
 *
 * @returns The control.
 */
export default function AuthControl() {
  const { authError, authLoading, authSession, loginBrowserUser, logoutBrowserUser } =
    useStudioStore();

  return (
    <AuthControlView
      authError={authError}
      authLoading={authLoading}
      authSession={authSession}
      onLogin={loginBrowserUser}
      onLogout={logoutBrowserUser}
    />
  );
}

/** The session to show and the actions that change it. */
export interface AuthControlViewProps {
  authError: string | null;
  authLoading: boolean;
  authSession: StudioAuthSession | null;
  onLogin: (username: string, password: string) => Promise<void>;
  onLogout: () => Promise<void>;
}

/**
 * Sign-in and sign-out, with every action supplied as a prop.
 *
 * @param props - The session and the actions.
 * @returns The control.
 */
export function AuthControlView({
  authError,
  authLoading,
  authSession,
  onLogin,
  onLogout,
}: AuthControlViewProps) {

  /**
   * Sign in with the typed credentials.
   *
   * @param event - The submission.
   */
  function submitLogin(event: SyntheticEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const username = formText(form.get("username"));
    const password = formText(form.get("password"));
    void onLogin(username, password);
  }

  if (authSession?.authenticated) {
    return (
      <div className="auth-control">
        <span>{authSession.principal_id}</span>
        <button
          aria-label="Logout browser session"
          disabled={authLoading}
          onClick={() => void onLogout()}
        >
          Logout
        </button>
      </div>
    );
  }

  return (
    <form className="auth-control" onSubmit={submitLogin}>
      <input
        aria-label="Studio username"
        autoComplete="username"
        disabled={authLoading}
        name="username"
        placeholder="user"
      />
      <input
        aria-label="Studio password"
        autoComplete="current-password"
        disabled={authLoading}
        name="password"
        placeholder="password"
        type="password"
      />
      <button aria-label="Login browser session" disabled={authLoading} type="submit">
        Login
      </button>
      {authError && <span className="auth-error">{authError}</span>}
    </form>
  );
}
