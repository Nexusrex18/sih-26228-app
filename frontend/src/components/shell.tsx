"use client";

import { AnimatePresence, MotionConfig, motion } from "framer-motion";
import {
  Activity,
  FileSearch,
  Loader2,
  LogOut,
  Moon,
  RefreshCw,
  ScrollText,
  Search,
  ShieldAlert,
  ShieldCheck,
  ShieldQuestion,
  Sun,
  Users,
  Zap,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import * as React from "react";
import { api, setCsrfToken } from "@/lib/api";
import type { Health, Session } from "@/lib/types";
import { cn } from "@/lib/ui";
import { CommandPalette } from "@/components/command-palette";
import { LedgerPulse } from "@/components/ledger-pulse";
import { Banner, Button } from "@/components/ui/primitives";

/* --------------------------------------------------------------- context */

interface AppState {
  session: Session | null;
  health: Health | null;
  loading: boolean;
  refreshHealth: () => Promise<void>;
}

const Ctx = React.createContext<AppState>({
  session: null,
  health: null,
  loading: true,
  refreshHealth: async () => undefined,
});

export const useApp = () => React.useContext(Ctx);

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [session, setSession] = React.useState<Session | null>(null);
  const [health, setHealth] = React.useState<Health | null>(null);
  const [loading, setLoading] = React.useState(true);

  const refreshHealth = React.useCallback(async () => {
    try {
      setHealth(await api.health());
    } catch {
      setHealth(null);
    }
  }, []);

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const s = await api.session();
        if (cancelled) return;
        setSession(s);
        setCsrfToken(s.csrf_token);
        if (s.authenticated) await refreshHealth();
      } catch {
        if (!cancelled) setSession(null);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [refreshHealth]);

  return (
    // `reducedMotion="user"`: under prefers-reduced-motion, Framer drops every transform
    // animation and keeps opacity — gentler, not zero.
    <MotionConfig reducedMotion="user">
      <Ctx.Provider value={{ session, health, loading, refreshHealth }}>
        {children}
      </Ctx.Provider>
    </MotionConfig>
  );
}

/* ----------------------------------------------------------------- theme */

function useTheme() {
  const [theme, setTheme] = React.useState<"dark" | "light">("dark");
  React.useEffect(() => {
    let stored: string | null = null;
    try {
      stored = localStorage.getItem("cva-theme");
    } catch {
      /* private window */
    }
    const next = stored === "light" ? "light" : "dark";
    setTheme(next);
    document.documentElement.classList.toggle("light", next === "light");
  }, []);
  const toggle = React.useCallback(() => {
    setTheme((prev) => {
      const next = prev === "light" ? "dark" : "light";
      document.documentElement.classList.toggle("light", next === "light");
      try {
        localStorage.setItem("cva-theme", next);
      } catch {
        /* private window */
      }
      return next;
    });
  }, []);
  return { theme, toggle };
}

/* ------------------------------------------------------------------ rail */

const NAV = [
  { href: "/", label: "Scans", icon: FileSearch },
  { href: "/audit", label: "Audit trail", icon: ScrollText },
  { href: "/verification", label: "Verification", icon: Activity },
];

function Rail() {
  const pathname = usePathname();
  const { theme, toggle } = useTheme();
  const { session } = useApp();
  // Accounts is the admin's page and only the admin's: the admin role manages identities
  // and holds no workflow rights, and nobody else may manage identities (D-E9).
  const nav =
    session?.role === "admin"
      ? [...NAV, { href: "/accounts", label: "Accounts", icon: Users }]
      : NAV;
  return (
    <nav
      aria-label="Sections"
      className="glass sticky top-0 z-30 flex h-16 shrink-0 items-center gap-2 border-b border-line px-3 md:h-screen md:w-16 md:flex-col md:border-b-0 md:border-r md:py-4"
    >
      <Link
        href="/"
        aria-label="CVA home"
        className="relative grid h-9 w-9 shrink-0 place-items-center overflow-hidden rounded border border-accent-dim font-mono text-[0.6rem] font-bold tracking-wide text-accent"
      >
        CVA
        <span
          aria-hidden
          className="pointer-events-none absolute inset-0 animate-sweep bg-gradient-to-br from-transparent via-accent/25 to-transparent"
        />
      </Link>
      <div className="flex gap-1 md:flex-col">
        {nav.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              title={label}
              aria-current={active ? "page" : undefined}
              className={cn(
                "relative grid h-11 w-11 place-items-center rounded-lg transition-colors md:h-10 md:w-10",
                active ? "text-accent" : "text-ink-faint hover:text-ink",
              )}
            >
              {active ? (
                // The indicator slides to the new section: on-screen movement, so the
                // strong ease-in-out, well under 300ms. No spring — nothing is thrown.
                <motion.span
                  layoutId="rail-active"
                  aria-hidden
                  className="absolute inset-0 rounded-lg border border-accent/40 bg-accent/10"
                  transition={{ duration: 0.25, ease: [0.77, 0, 0.175, 1] }}
                />
              ) : null}
              <Icon className="relative h-4 w-4" aria-hidden />
              <span className="sr-only">{label}</span>
            </Link>
          );
        })}
      </div>
      <div className="ml-auto flex gap-1 md:ml-0 md:mt-auto md:flex-col">
        <SignOut />
        <button
          type="button"
          onClick={toggle}
          aria-label={theme === "light" ? "Switch to dark theme" : "Switch to light theme"}
          className="relative grid h-11 w-11 place-items-center overflow-hidden rounded-lg text-ink-faint transition-colors hover:bg-surface-panel hover:text-ink md:h-10 md:w-10"
        >
          {/* The icon turns over like a dial: the old one rotates out as the new one
              rotates in. Rare action, so a little motion is earned — ease-out, 200ms,
              no bounce, and the exit is quicker than the entry. */}
          <AnimatePresence mode="popLayout" initial={false}>
            <motion.span
              key={theme}
              initial={{ opacity: 0, transform: "rotate(-90deg) scale(0.9)" }}
              animate={{
                opacity: 1,
                transform: "rotate(0deg) scale(1)",
                transition: { duration: 0.2, ease: [0.23, 1, 0.32, 1] },
              }}
              exit={{
                opacity: 0,
                transform: "rotate(90deg) scale(0.9)",
                transition: { duration: 0.12, ease: [0.23, 1, 0.32, 1] },
              }}
              className="grid place-items-center"
            >
              {theme === "light" ? (
                <Moon className="h-4 w-4" aria-hidden />
              ) : (
                <Sun className="h-4 w-4" aria-hidden />
              )}
            </motion.span>
          </AnimatePresence>
        </button>
      </div>
    </nav>
  );
}

/** A real form POST, not a fetch: the server ends the session and redirects to its own
 *  sign-in page, and the browser follows that as a navigation. The CSRF token travels as
 *  the form field the server's before-request check reads. The action is absolute — it
 *  lives outside the SPA's `/app` base path. */
function SignOut() {
  const { session } = useApp();
  if (!session?.authenticated) return null;
  return (
    <form method="post" action="/logout">
      <input type="hidden" name="csrf_token" value={session.csrf_token} />
      <button
        type="submit"
        title="Sign out"
        aria-label="Sign out"
        className="grid h-11 w-11 place-items-center rounded-lg text-ink-faint transition-colors hover:bg-surface-panel hover:text-ink md:h-10 md:w-10"
      >
        <LogOut className="h-4 w-4" aria-hidden />
      </button>
    </form>
  );
}

/* --------------------------------------------------------- trust banner
 * On EVERY page. The fold is only as trustworthy as the ledger, so the ledger's state is
 * never one click away — it is above the content, always (plan §5.4). */

export function TrustBanner() {
  const { health, refreshHealth } = useApp();
  const [busy, setBusy] = React.useState(false);
  if (!health) return null;

  const tone = (
    {
      verified: "ok",
      unverified: "warn",
      failed: "alarm",
      unavailable: "absent",
    } as const
  )[health.kind];
  const icon = (
    {
      verified: ShieldCheck,
      unverified: ShieldQuestion,
      failed: ShieldAlert,
      unavailable: Zap,
    } as const
  )[health.kind];
  const title = {
    verified: "Audit ledger verified",
    unverified: "Audit ledger not verified",
    failed: "Audit ledger verification FAILED",
    unavailable: "Workflow unavailable",
  }[health.kind];

  return (
    <div className="mb-6">
      <Banner
        tone={tone}
        icon={icon}
        title={title}
        alarm={health.kind === "failed"}
        sweep={health.kind === "verified"}
        actions={
          <div className="flex items-center gap-3">
            <div className="hidden md:block">
              <LedgerPulse health={health} />
            </div>
            <Button
              size="sm"
              icon={busy ? Loader2 : RefreshCw}
              disabled={busy}
              className={busy ? "[&_svg]:animate-spin" : undefined}
              onClick={async () => {
                setBusy(true);
                await refreshHealth();
                setBusy(false);
              }}
            >
              Verify now
            </Button>
          </div>
        }
      >
        <p>{health.text}</p>
        {!health.writable ? (
          <p className="mt-1 text-xs">
            Recorded decisions are still shown. New ones cannot be taken: a decision that
            exists in this page but not in the ledger is exactly what this design refuses to
            produce.
          </p>
        ) : null}
      </Banner>
    </div>
  );
}

/* ----------------------------------------------------------------- shell */

export function Shell({
  title,
  subtitle,
  actions,
  children,
  wide,
}: {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
  wide?: boolean;
}) {
  const { session, loading } = useApp();
  const pathname = usePathname();
  const [paletteOpen, setPaletteOpen] = React.useState(false);

  if (loading) {
    return (
      <div className="grid min-h-screen place-items-center">
        <div className="flex items-center gap-3 text-ink-muted">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
          <span className="font-mono text-sm">loading</span>
        </div>
      </div>
    );
  }

  if (session && !session.authenticated) {
    return <SignInWall />;
  }

  return (
    <div className="relative flex min-h-screen flex-col md:flex-row">
      {/* The room: a slow aurora and a drifting instrument grid. Decoration only — it
          is removed under reduced transparency and stopped under reduced motion. */}
      <div aria-hidden className="aurora pointer-events-none fixed inset-0" />
      <div aria-hidden className="grid-field pointer-events-none fixed inset-0 opacity-30" />
      <Rail />
      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} />
      <div className="relative flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex flex-wrap items-center gap-4 border-b border-line bg-surface/70 px-5 py-3 backdrop-blur-md md:top-0">
          <div className="min-w-0">
            <h1 className="truncate font-display text-base font-semibold tracking-tight">{title}</h1>
            {subtitle ? (
              <div className="truncate font-mono text-2xs text-ink-faint">{subtitle}</div>
            ) : null}
          </div>
          <div className="ml-auto flex items-center gap-2">
            <button
              type="button"
              onClick={() => setPaletteOpen(true)}
              className="inline-flex h-11 w-11 items-center justify-center gap-2 rounded-lg border border-line-strong bg-surface-raised/50 font-mono text-2xs text-ink-muted transition-colors hover:border-accent/40 hover:text-ink sm:h-8 sm:w-auto sm:px-3"
              aria-label="Jump to a scan or page"
            >
              <Search className="h-3.5 w-3.5" aria-hidden />
              <span className="hidden sm:inline">Jump to</span>
              <kbd className="hidden rounded border border-line-strong px-1 text-ink-faint sm:inline">
                ⌘K
              </kbd>
            </button>
            {actions}
            {session?.authenticated ? (
              <span className="hidden items-center gap-2 rounded-full border border-line-strong bg-surface-raised/60 px-3 py-1 font-mono text-2xs text-ink-muted sm:inline-flex">
                <span aria-hidden className="pulse-dot h-1.5 w-1.5 rounded-full bg-accept text-accept" />
                {session.actor_id} · {session.role}
              </span>
            ) : null}
          </div>
        </header>

        <main
          className={cn(
            "relative w-full flex-1 px-5 py-6",
            wide ? "max-w-[1440px]" : "max-w-[1280px]",
          )}
        >
          <AnimatePresence mode="wait">
            <motion.div
              key={pathname}
              initial={{ opacity: 0, transform: "translateY(8px)" }}
              animate={{
                opacity: 1,
                transform: "translateY(0px)",
                transition: { duration: 0.24, ease: [0.23, 1, 0.32, 1] },
              }}
              exit={{
                opacity: 0,
                transform: "translateY(-4px)",
                transition: { duration: 0.14, ease: [0.23, 1, 0.32, 1] },
              }}
            >
              <TrustBanner />
              {children}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}

/** Not a login form: the session cookie is issued by Flask's own form, which keeps one
 *  credential path rather than two. */
function SignInWall() {
  return (
    <div className="grid min-h-screen place-items-center px-6">
      <div className="w-full max-w-md rounded-lg border border-line bg-surface-panel p-6">
        <h1 className="text-lg font-semibold">Sign in required</h1>
        <p className="mt-2 text-sm text-ink-muted">
          This dashboard reads an audit ledger and records analyst decisions against it.
          Both need an identified account.
        </p>
        <a
          className="mt-5 inline-flex h-9 items-center rounded border border-accent bg-accent px-4 text-sm font-medium text-white"
          href={`/login?next=${encodeURIComponent("/app/")}`}
        >
          Go to sign in
        </a>
      </div>
    </div>
  );
}
