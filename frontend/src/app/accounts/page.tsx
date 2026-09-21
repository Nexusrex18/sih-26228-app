"use client";

import { KeyRound, Loader2, UserPlus, UserX, UserCheck } from "lucide-react";
import * as React from "react";
import { toast } from "sonner";
import { Shell, useApp } from "@/components/shell";
import {
  Banner,
  Button,
  Chip,
  Empty,
  Panel,
  Section,
  SectionHead,
} from "@/components/ui/primitives";
import { api, ApiError, type AccountRow, type AccountsPayload } from "@/lib/api";
import type { Role } from "@/lib/types";
import { cn } from "@/lib/ui";

/**
 * Account management (plan D-E9).
 *
 * `admin` manages accounts and holds no workflow rights: one person controlling both
 * identity and decisions is exactly what the separation exists to prevent. The server
 * refuses every call from any other role; this page only saves the round trip.
 *
 * No optimistic updates: every change re-renders from the server's answer. Passwords
 * travel in a POST body only — never in a URL, where they would land in history and logs.
 */
export default function AccountsPage() {
  const { session } = useApp();
  const [data, setData] = React.useState<AccountsPayload | null>(null);
  const isAdmin = session?.role === "admin";

  const load = React.useCallback(async () => {
    try {
      setData(await api.accounts());
    } catch (e) {
      toast.error((e as Error).message);
    }
  }, []);

  React.useEffect(() => {
    if (session?.authenticated && isAdmin) void load();
  }, [session?.authenticated, isAdmin, load]);

  const apply = async (call: Promise<AccountsPayload>) => {
    try {
      const r = await call;
      setData(r);
      if (r.message) toast.success(r.message);
      return true;
    } catch (e) {
      const err = e as ApiError;
      toast.error(err.title ?? "Refused", {
        description: `${err.message} Nothing was recorded.`,
      });
      return false;
    }
  };

  return (
    <Shell title="Accounts" subtitle="local accounts · issued by the administrator" wide>
      {!isAdmin ? (
        <Empty title="This page is for the admin role">
          <p>
            Accounts are managed by an administrator. Your account holds the{" "}
            <span className="font-mono">{session?.role}</span> role, which can read and
            decide but not issue identities.
          </p>
        </Empty>
      ) : (
        <>
          <Section>
            <SectionHead
              title="Issued accounts"
              note="There is no self sign-up: with no identity provider on an air gap, an account anyone could create would make every signature in the ledger meaningless."
            />
            {data === null ? (
              <div className="flex items-center gap-3 py-10 text-ink-muted">
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                <span className="font-mono text-sm">loading accounts</span>
              </div>
            ) : (
              <div className="space-y-2">
                {data.accounts.map((a) => (
                  <AccountCard
                    key={a.actor_id}
                    account={a}
                    roles={data.roles}
                    minChars={data.min_password_chars}
                    self={a.actor_id === session?.actor_id}
                    apply={apply}
                  />
                ))}
              </div>
            )}
          </Section>

          <Section delay={0.04}>
            <SectionHead
              title="Issue an account"
              note="Roles: analyst proposes a decision; approver confirms a lowering by someone else; viewer reads; admin manages accounts and cannot decide."
            />
            {data ? (
              <CreateForm roles={data.roles} minChars={data.min_password_chars} apply={apply} />
            ) : null}
          </Section>

          <Section delay={0.08}>
            <Banner tone="info" title="What a change here does">
              <p>
                A role change, a disable or a password reset ends that person&rsquo;s live
                sessions, so it binds on their next request. Disabling never removes
                anything from the ledger: their past decisions stay recorded, which is the
                point of an append-only record. Account events go to the application log,
                not the ledger, in this version.
              </p>
            </Banner>
          </Section>
        </>
      )}
    </Shell>
  );
}

function AccountCard({
  account,
  roles,
  minChars,
  self,
  apply,
}: {
  account: AccountRow;
  roles: Role[];
  minChars: number;
  self: boolean;
  apply: (call: Promise<AccountsPayload>) => Promise<boolean>;
}) {
  const [password, setPassword] = React.useState("");
  const [busy, setBusy] = React.useState(false);

  const run = async (call: () => Promise<AccountsPayload>) => {
    setBusy(true);
    await apply(call());
    setBusy(false);
  };

  return (
    <Panel
      className={cn(
        "flex flex-wrap items-center gap-x-5 gap-y-3 p-4",
        account.disabled && "hatched",
      )}
    >
      <div className="min-w-[10rem]">
        <div className="break-all font-mono text-sm text-ink">{account.actor_id}</div>
        <div className="mt-1 flex flex-wrap gap-2">
          {account.disabled ? (
            <Chip tone="absent" icon={UserX}>
              disabled
            </Chip>
          ) : (
            <Chip tone="accept" icon={UserCheck}>
              active
            </Chip>
          )}
          {self ? <Chip tone="accent">you</Chip> : null}
        </div>
      </div>

      <label className="flex items-center gap-2 text-2xs text-ink-faint">
        <span className="font-mono uppercase tracking-wider">Role</span>
        <select
          value={account.role}
          disabled={busy}
          aria-label={`Role for ${account.actor_id}`}
          onChange={(e) =>
            void run(() =>
              api.updateAccount(account.actor_id, { role: e.target.value as Role }),
            )
          }
          className="h-11 rounded-lg border border-line-strong bg-surface-panel px-3 font-mono text-xs text-ink sm:h-9"
        >
          {roles.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
      </label>

      <form
        className="flex flex-wrap items-center gap-2"
        onSubmit={async (e) => {
          e.preventDefault();
          await run(() => api.updateAccount(account.actor_id, { password }));
          setPassword("");
        }}
      >
        <input
          type="password"
          autoComplete="new-password"
          value={password}
          minLength={minChars}
          required
          onChange={(e) => setPassword(e.target.value)}
          placeholder={`new password (${minChars}+ chars)`}
          aria-label={`New password for ${account.actor_id}`}
          className="h-11 w-52 rounded-lg border border-line-strong bg-surface-deep px-3 text-sm text-ink placeholder:text-ink-faint sm:h-9"
        />
        <Button
          type="submit"
          size="sm"
          icon={KeyRound}
          disabled={busy || password.length < minChars}
          className="h-11 sm:h-7"
        >
          Set password
        </Button>
      </form>

      <span className="flex-1" />

      <Button
        size="sm"
        tone={account.disabled ? "default" : "danger"}
        disabled={busy || self}
        title={self ? "You cannot disable your own account" : undefined}
        className="h-11 sm:h-7"
        onClick={() =>
          void run(() =>
            api.updateAccount(account.actor_id, { disabled: !account.disabled }),
          )
        }
      >
        {account.disabled ? "Enable" : "Disable"}
      </Button>
    </Panel>
  );
}

function CreateForm({
  roles,
  minChars,
  apply,
}: {
  roles: Role[];
  minChars: number;
  apply: (call: Promise<AccountsPayload>) => Promise<boolean>;
}) {
  const [actor, setActor] = React.useState("");
  const [role, setRole] = React.useState<Role>("analyst");
  const [password, setPassword] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const validId = /^[A-Za-z0-9._:-]{1,64}$/.test(actor);

  return (
    <Panel className="p-4">
      <form
        className="grid gap-4 sm:grid-cols-[1fr_10rem_1fr_auto] sm:items-end"
        onSubmit={async (e) => {
          e.preventDefault();
          setBusy(true);
          const ok = await apply(api.createAccount({ actor_id: actor, role, password }));
          setBusy(false);
          if (ok) {
            setActor("");
            setPassword("");
          }
        }}
      >
        <label className="grid gap-1.5 text-xs text-ink-muted">
          Account name
          <input
            value={actor}
            onChange={(e) => setActor(e.target.value)}
            autoCapitalize="none"
            autoCorrect="off"
            spellCheck={false}
            required
            placeholder="e.g. Tan.00"
            className="h-11 rounded-lg border border-line-strong bg-surface-deep px-3 font-mono text-sm text-ink placeholder:text-ink-faint"
          />
          <span className={cn("text-2xs", actor && !validId ? "text-review" : "text-ink-faint")}>
            Letters, digits and . _ : - only. Case-sensitive: the ledger records it exactly.
          </span>
        </label>
        <label className="grid gap-1.5 text-xs text-ink-muted">
          Role
          <select
            value={role}
            onChange={(e) => setRole(e.target.value as Role)}
            className="h-11 rounded-lg border border-line-strong bg-surface-panel px-3 font-mono text-sm text-ink"
          >
            {roles.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
          <span className="text-2xs text-ink-faint">&nbsp;</span>
        </label>
        <label className="grid gap-1.5 text-xs text-ink-muted">
          Password
          <input
            type="password"
            autoComplete="new-password"
            value={password}
            minLength={minChars}
            required
            onChange={(e) => setPassword(e.target.value)}
            className="h-11 rounded-lg border border-line-strong bg-surface-deep px-3 text-sm text-ink"
          />
          <span className="text-2xs text-ink-faint">At least {minChars} characters.</span>
        </label>
        <Button
          type="submit"
          tone="primary"
          icon={busy ? Loader2 : UserPlus}
          disabled={busy || !validId || password.length < minChars}
          className="h-11 sm:mb-5"
        >
          Issue account
        </Button>
      </form>
    </Panel>
  );
}
