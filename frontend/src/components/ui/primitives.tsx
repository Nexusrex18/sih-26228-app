"use client";

import { motion, type HTMLMotionProps } from "framer-motion";
import type { LucideIcon } from "lucide-react";
import * as React from "react";
import { cn } from "@/lib/ui";

/* ------------------------------------------------------------------ Panel */

export function Panel({
  className,
  hatched,
  children,
  ...rest
}: React.HTMLAttributes<HTMLDivElement> & { hatched?: boolean }) {
  return (
    <div
      className={cn(
        "glass rounded-xl border border-line/80",
        hatched && "hatched border-absent/30 bg-absent/[0.06]",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}

export function PanelHeader({
  className,
  children,
  ...rest
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "flex items-center gap-3 border-b border-line bg-surface-raised/50 px-4 py-3",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}

/* ------------------------------------------------------------------ Eyebrow */

export function Eyebrow({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "font-mono text-eyebrow font-semibold uppercase text-ink-faint",
        className,
      )}
    >
      {children}
    </span>
  );
}

/* ------------------------------------------------------------------ Chip
 * Colour is never the only carrier of state: every chip is icon + word + colour. */

export function Chip({
  tone = "neutral",
  icon: Icon,
  children,
  className,
  title,
}: {
  tone?: "neutral" | "quarantine" | "review" | "accept" | "pending" | "absent" | "accent";
  icon?: LucideIcon;
  children: React.ReactNode;
  className?: string;
  title?: string;
}) {
  const tones: Record<string, string> = {
    neutral: "border-line-strong bg-surface-raised/60 text-ink-muted",
    quarantine: "border-quarantine/35 bg-quarantine/10 text-quarantine",
    review: "border-review/35 bg-review/10 text-review",
    accept: "border-accept/35 bg-accept/10 text-accept",
    pending: "border-pending/35 bg-pending/10 text-pending",
    absent: "border-absent/30 bg-absent/[0.08] text-absent",
    accent: "border-accent/40 bg-accent/10 text-accent",
  };
  return (
    <span
      title={title}
      className={cn(
        "inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2.5 py-0.5",
        "font-mono text-2xs font-semibold",
        tones[tone],
        className,
      )}
    >
      {Icon ? <Icon className="h-3 w-3 shrink-0" aria-hidden /> : null}
      {children}
    </span>
  );
}

/* ------------------------------------------------------------------ Button */

const BUTTON_TONES: Record<string, string> = {
  default:
    "border-line-strong bg-surface-raised text-ink hover:border-accent/50 hover:bg-surface-raised/70",
  primary:
    "border-accent bg-accent text-white hover:bg-accent/85 shadow-[0_6px_20px_-10px_hsl(var(--accent))]",
  danger:
    "border-quarantine/40 bg-quarantine/10 text-quarantine hover:bg-quarantine/20",
  ghost: "border-transparent bg-transparent text-ink-muted hover:bg-surface-raised hover:text-ink",
};

export function Button({
  tone = "default",
  size = "md",
  icon: Icon,
  className,
  children,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  tone?: keyof typeof BUTTON_TONES;
  size?: "sm" | "md";
  icon?: LucideIcon;
}) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg border font-medium",
        // Press feedback: a 0.97 scale over 160ms with the strong ease-out — the interface
        // confirming it heard the press. Colour changes on their own, un-eased.
        "transition-[background-color,border-color,transform] duration-press ease-out-strong active:scale-[0.97]",
        "disabled:pointer-events-none disabled:opacity-45",
        size === "sm" ? "h-7 px-2.5 text-xs" : "h-9 px-3.5 text-sm",
        BUTTON_TONES[tone],
        className,
      )}
      {...rest}
    >
      {Icon ? <Icon className={size === "sm" ? "h-3.5 w-3.5" : "h-4 w-4"} aria-hidden /> : null}
      {children}
    </button>
  );
}

/* ------------------------------------------------------------------ Stat
 * The label sits above the number, because the label is what you scan for and the number is
 * what you read. Numbers are mono and tabular so a count that ticks up does not reflow. */

export function Stat({
  label,
  value,
  note,
  tone = "neutral",
  countUp = false,
}: {
  label: string;
  value: number | string;
  note?: React.ReactNode;
  tone?: "neutral" | "quarantine" | "review" | "accept" | "pending" | "absent";
  countUp?: boolean;
}) {
  const tones: Record<string, string> = {
    neutral: "text-ink",
    quarantine: "text-quarantine",
    review: "text-review",
    accept: "text-accept",
    pending: "text-pending",
    absent: "text-absent",
  };
  return (
    <div className="flex flex-col gap-0.5">
      <Eyebrow>{label}</Eyebrow>
      <span
        className={cn(
          "font-mono text-3xl font-semibold leading-none tracking-tight tnum",
          tones[tone],
        )}
      >
        {countUp && typeof value === "number" ? <CountUp value={value} /> : value}
      </span>
      {note ? <span className="text-xs text-ink-muted">{note}</span> : null}
    </div>
  );
}

/** Counts up to a value that is already correct in the markup, so nothing is invented. */
export function CountUp({ value, duration = 520 }: { value: number; duration?: number }) {
  const [shown, setShown] = React.useState(value);
  const reduced = usePrefersReducedMotion();

  React.useEffect(() => {
    if (reduced || value <= 0 || value > 100_000) {
      setShown(value);
      return;
    }
    let raf = 0;
    let start: number | null = null;
    const step = (ts: number) => {
      if (start === null) start = ts;
      const p = Math.min(1, (ts - start) / duration);
      setShown(Math.round(value * (1 - Math.pow(1 - p, 3))));
      if (p < 1) raf = requestAnimationFrame(step);
    };
    setShown(0);
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [value, duration, reduced]);

  return <>{shown}</>;
}

export function usePrefersReducedMotion() {
  const [reduced, setReduced] = React.useState(false);
  React.useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(mq.matches);
    const on = () => setReduced(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return reduced;
}

/* ------------------------------------------------------------------ Banner */

export function Banner({
  tone,
  icon: Icon,
  title,
  children,
  actions,
  alarm,
  sweep,
}: {
  tone: "ok" | "warn" | "alarm" | "absent" | "info";
  icon?: LucideIcon;
  title: React.ReactNode;
  children?: React.ReactNode;
  actions?: React.ReactNode;
  alarm?: boolean;
  /** A scanning light across the icon — the "verified" shield being re-read. */
  sweep?: boolean;
}) {
  // Tone is the tinted surface, a tinted 1px edge and the icon tile — never a thick side
  // stripe, which on a rounded box reads as a pasted-on decoration.
  const tones: Record<string, string> = {
    ok: "border-accept/30 bg-accept/[0.07] [&_.banner-icon]:text-accept",
    warn: "border-review/30 bg-review/[0.07] [&_.banner-icon]:text-review",
    alarm: "border-quarantine/40 bg-quarantine/[0.08] [&_.banner-icon]:text-quarantine",
    absent: "border-absent/30 bg-absent/[0.06] [&_.banner-icon]:text-absent",
    info: "border-accent/30 bg-accent/[0.07] [&_.banner-icon]:text-accent",
  };
  return (
    <div
      role={tone === "alarm" || tone === "absent" ? "alert" : "status"}
      className={cn(
        "glass relative flex flex-wrap items-start gap-3 overflow-hidden rounded-xl border px-4 py-3",
        tones[tone],
      )}
    >
      {alarm ? (
        <span
          aria-hidden
          className="pointer-events-none absolute inset-0 animate-sweep bg-gradient-to-r from-transparent via-quarantine/15 to-transparent"
        />
      ) : null}
      {Icon ? (
        <span
          className={cn(
            "banner-icon relative mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-lg border border-line-strong bg-surface-raised/50",
            sweep && "sweep-mask",
          )}
        >
          <Icon className="h-4 w-4" aria-hidden />
        </span>
      ) : null}
      <div className="relative min-w-0 flex-1">
        <strong className="block text-sm font-semibold">{title}</strong>
        {children ? (
          <div className="text-sm text-ink-muted">{children}</div>
        ) : null}
      </div>
      {actions ? <div className="relative flex shrink-0 gap-2">{actions}</div> : null}
    </div>
  );
}

/* ------------------------------------------------------------------ Empty
 * Empty and error states are CONTENT. "0 findings" must say which detectors ran. */

export function Empty({
  title,
  children,
  hatched,
}: {
  title: string;
  children?: React.ReactNode;
  hatched?: boolean;
}) {
  return (
    <div
      className={cn(
        "rounded-lg border border-dashed border-line-strong bg-surface-raised/40 px-6 py-10",
        hatched && "hatched",
      )}
    >
      <h3 className="mb-2 text-base font-semibold">{title}</h3>
      <div className="max-w-[70ch] space-y-2 text-sm text-ink-muted">{children}</div>
    </div>
  );
}

/* ------------------------------------------------------------------ Hash */

export function Hash({ value, label }: { value?: string | null; label?: string }) {
  const [copied, setCopied] = React.useState(false);
  if (!value) return <span className="text-ink-faint">—</span>;
  return (
    <span className="inline-flex items-center gap-2">
      {label ? <span className="text-xs text-ink-faint">{label}</span> : null}
      <code className="rounded-sm border border-line bg-surface-deep px-1.5 py-0.5 font-mono text-xs break-all">
        {value.length > 24 ? `${value.slice(0, 24)}…` : value}
      </code>
      <button
        type="button"
        className="text-xs text-ink-faint transition-colors hover:text-accent"
        onClick={() => {
          navigator.clipboard?.writeText(value).then(
            () => {
              setCopied(true);
              setTimeout(() => setCopied(false), 1200);
            },
            () => undefined,
          );
        }}
      >
        {copied ? "copied" : "copy"}
      </button>
    </span>
  );
}

/* ------------------------------------------------------------------ motion */

export const fadeUp = {
  initial: { opacity: 0, transform: "translateY(8px)" },
  animate: { opacity: 1, transform: "translateY(0px)" },
  transition: { duration: 0.28, ease: [0.23, 1, 0.32, 1] as const },
};

export function Section({
  children,
  delay = 0,
  className,
  ...rest
}: HTMLMotionProps<"section"> & { delay?: number }) {
  return (
    <motion.section
      initial={{ opacity: 0, transform: "translateY(8px)" }}
      animate={{ opacity: 1, transform: "translateY(0px)" }}
      transition={{ duration: 0.28, delay, ease: [0.23, 1, 0.32, 1] }}
      className={cn("mb-12 scroll-mt-24", className)}
      {...rest}
    >
      {children}
    </motion.section>
  );
}

export function SectionHead({
  eyebrow,
  title,
  note,
  actions,
}: {
  eyebrow?: string;
  title: string;
  note?: React.ReactNode;
  actions?: React.ReactNode;
}) {
  return (
    <div className="mb-4 flex flex-wrap items-baseline gap-3 border-b border-line pb-2">
      {eyebrow ? <Eyebrow>{eyebrow}</Eyebrow> : null}
      <h2 className="font-display text-xl font-semibold tracking-tight">{title}</h2>
      {note ? (
        <span className="max-w-[60ch] text-sm text-ink-muted">{note}</span>
      ) : null}
      {actions ? <div className="ml-auto flex gap-2">{actions}</div> : null}
    </div>
  );
}
