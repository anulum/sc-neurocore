import type { FormEvent } from "react";

import type { StudioAuthSession } from "../api/client";
import { useStudioStore } from "../stores/studio";

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

export interface AuthControlViewProps {
  authError: string | null;
  authLoading: boolean;
  authSession: StudioAuthSession | null;
  onLogin: (username: string, password: string) => Promise<void>;
  onLogout: () => Promise<void>;
}

export function AuthControlView({
  authError,
  authLoading,
  authSession,
  onLogin,
  onLogout,
}: AuthControlViewProps) {

  function submitLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const username = String(form.get("username") ?? "");
    const password = String(form.get("password") ?? "");
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
