"use client";

import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  FileSearch,
  Loader2,
  Moon,
  RefreshCw,
  ScrollText,
  ShieldAlert,
  ShieldCheck,
  ShieldQuestion,
  Sun,
  Zap,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import * as React from "react";
import { api, setCsrfToken } from "@/lib/api";
import type { Health, Session } from "@/lib/types";
import { cn } from "@/lib/ui";
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
    <Ctx.Provider value={{ session, health, loading, refreshHealth }}>
      {children}
    </Ctx.Provider>
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
  return (
    <nav
      aria-label="Sections"
      className="sticky top-0 z-30 flex h-16 shrink-0 items-center gap-2 border-b border-line bg-surface-deep px-3 md:h-screen md:w-16 md:flex-col md:border-b-0 md:border-r md:py-4"
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
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              title={label}
              aria-current={active ? "page" : undefined}
              className={cn(
                "grid h-9 w-9 place-items-center rounded border transition-colors",
                active
                  ? "border-accent/40 bg-surface-panel text-accent"
                  : "border-transparent text-ink-faint hover:bg-surface-panel hover:text-ink",
              )}
            >
              <Icon className="h-4 w-4" aria-hidden />
              <span className="sr-only">{label}</span>
            </Link>
          );
        })}
      </div>
      <div className="ml-auto md:ml-0 md:mt-auto">
        <button
          type="button"
          onClick={toggle}
          aria-label={theme === "light" ? "Switch to dark theme" : "Switch to light theme"}
          className="grid h-9 w-9 place-items-center rounded border border-transparent text-ink-faint transition-colors hover:bg-surface-panel hover:text-ink"
        >
          {theme === "light" ? (
            <Moon className="h-4 w-4" aria-hidden />
          ) : (
            <Sun className="h-4 w-4" aria-hidden />
          )}
        </button>
      </div>
    </nav>
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
        actions={
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
      <div aria-hidden className="grid-field pointer-events-none fixed inset-0 opacity-30" />
      <Rail />
      <div className="relative flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex flex-wrap items-center gap-4 border-b border-line bg-surface/80 px-5 py-3 backdrop-blur-md md:top-0">
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold tracking-tight">{title}</div>
            {subtitle ? (
              <div className="truncate text-xs text-ink-faint">{subtitle}</div>
            ) : null}
          </div>
          <div className="ml-auto flex items-center gap-2">
            {actions}
            {session?.authenticated ? (
              <span className="hidden items-center gap-2 rounded-full border border-line-strong bg-surface-raised/60 px-3 py-1 font-mono text-2xs text-ink-muted sm:inline-flex">
                <span aria-hidden className="h-1.5 w-1.5 rounded-full bg-accept" />
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
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.25, ease: [0.2, 0.9, 0.3, 1] }}
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
