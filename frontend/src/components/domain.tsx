"use client";

import { motion } from "framer-motion";
import {
  AlertOctagon,
  Check,
  CircleDashed,
  Clock4,
  Flag,
  KeyRound,
  ShieldCheck,
  ShieldOff,
  ShieldQuestion,
} from "lucide-react";
import * as React from "react";
import type {
  Availability,
  ContributorRow,
  Disposition,
  FindingState,
  PlanRow,
  SealState,
  Severity,
} from "@/lib/types";
import { cn, fixed, hostTime, isAbsent, SEVERITY_TONE } from "@/lib/ui";
import { Chip, Eyebrow, Panel } from "@/components/ui/primitives";

/* ------------------------------------------------------- disposition chip */

const DISPOSITION_ICON = {
  quarantine: AlertOctagon,
  review: Flag,
  accept: Check,
  pending: Clock4,
} as const;

export function DispositionChip({
  value,
  label,
  className,
}: {
  value: Disposition | "pending";
  label?: string;
  className?: string;
}) {
  return (
    <Chip tone={value} icon={DISPOSITION_ICON[value]} className={className}>
      {label ?? value}
    </Chip>
  );
}

export function SeverityText({ value }: { value: Severity }) {
  return (
    <span
      className={cn(
        "font-mono text-2xs font-bold uppercase tracking-wider",
        SEVERITY_TONE[value],
      )}
    >
      {value}
    </span>
  );
}

/* ---------------------------------------------------------- seal badge
 * Three states, never two. "not sealed" and "differs" are different facts, and collapsing
 * them would let an altered report present as an unsealed one. */

export function SealBadge({ seal }: { seal: SealState | null }) {
  if (!seal) {
    return (
      <Chip tone="neutral" icon={ShieldQuestion}>
        ledger unreachable
      </Chip>
    );
  }
  const map = {
    sealed: { tone: "accept", icon: ShieldCheck, pulse: "animate-breathe" },
    differs: { tone: "quarantine", icon: ShieldOff, pulse: "animate-alarm" },
    not_sealed: { tone: "absent", icon: ShieldQuestion, pulse: "" },
    unknown: { tone: "neutral", icon: ShieldQuestion, pulse: "" },
  } as const;
  const cfg = map[seal.state];
  return (
    <span className="inline-flex items-center gap-2" title={seal.detail}>
      <span
        aria-hidden
        className={cn(
          "h-1.5 w-1.5 shrink-0 rounded-full",
          seal.state === "sealed" && "bg-accept",
          seal.state === "differs" && "bg-quarantine",
          seal.state === "not_sealed" && "bg-absent",
          seal.state === "unknown" && "bg-ink-faint",
          cfg.pulse,
        )}
      />
      <Chip tone={cfg.tone} icon={cfg.icon}>
        {seal.label}
      </Chip>
    </span>
  );
}

/* --------------------------------------------------------- availability */

export function AvailabilityChip({
  state,
  exclusionReason,
}: {
  state: Availability;
  exclusionReason?: string | null;
}) {
  if (!isAbsent(state)) return null;
  return (
    <Chip tone="absent" icon={CircleDashed}>
      {state}
      {exclusionReason ? ` · ${exclusionReason}` : ""}
    </Chip>
  );
}

/* ----------------------------------------------------- effective state
 * The tool's disposition stays visible beside the human one. Tamper-evidence needs an
 * append-only story: what the tool said, then what humans did (plan D-E5). */

export function EffectiveState({
  state,
  original,
  originalRule,
}: {
  state: FindingState | null;
  original: Disposition;
  originalRule: string;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-xs text-ink-faint">tool said</span>
      <DispositionChip value={original} />
      <span className="font-mono text-2xs text-ink-faint">{originalRule}</span>
      {state && state.effective !== original ? (
        <>
          <span aria-hidden className="text-ink-faint">
            →
          </span>
          <span className="text-xs text-ink-faint">now</span>
          <DispositionChip value={state.effective} />
          <span className="text-xs text-ink-faint">
            cited to ledger seq{" "}
            <span className="font-mono text-ink-muted">{state.cited_seq}</span>
          </span>
        </>
      ) : null}
      {state?.pending ? (
        <DispositionChip value="pending" label="pending a second approver" />
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------ seq rail
 * The signature element. A vertical spine of ledger sequence numbers with a highlight
 * travelling downward, the direction records are appended. It encodes the one genuinely
 * unusual fact here: this ledger is ordered by seq, never by the clock. */

export function SeqRail({ children }: { children: React.ReactNode }) {
  return <div className="seqrail">{children}</div>;
}

export function SeqItem({
  seq,
  tone = "neutral",
  index = 0,
  children,
}: {
  seq: number;
  tone?: "neutral" | "decision" | "flagged";
  index?: number;
  children: React.ReactNode;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, transform: "translateX(-8px)" }}
      animate={{ opacity: 1, transform: "translateX(0px)" }}
      transition={{
        duration: 0.24,
        delay: Math.min(index * 0.035, 0.3),
        ease: [0.23, 1, 0.32, 1],
      }}
      className="relative mb-4 last:mb-0"
    >
      <span
        className={cn(
          "absolute -left-14 top-0.5 w-11 text-right font-mono text-xs font-semibold tnum",
          tone === "decision" && "text-accent",
          tone === "flagged" && "text-quarantine",
          tone === "neutral" && "text-ink-faint",
        )}
      >
        {seq}
        <span
          aria-hidden
          className={cn(
            "absolute -right-[1.125rem] top-2 h-[7px] w-[7px] rounded-full border-2 bg-surface",
            tone === "decision" && "border-accent",
            tone === "flagged" && "border-quarantine",
            tone === "neutral" && "border-line-strong",
          )}
        />
      </span>
      {children}
    </motion.div>
  );
}

/* ------------------------------------------------------------- plan rows
 * All four states print at equal prominence. A shorter report must never look like a
 * cleaner result, so an absent check is hatched rather than faded. */

export function PlanRows({ rows }: { rows: PlanRow[] }) {
  if (!rows.length) {
    return (
      <p className="text-sm text-ink-muted">
        This report carries no check plan, so what was attempted cannot be shown.
      </p>
    );
  }
  return (
    <div className="space-y-1">
      {rows.map((r) => (
        <div
          key={r.check_id}
          className={cn(
            "flex flex-wrap items-baseline gap-3 rounded border border-line px-3 py-2 text-sm",
            isAbsent(r.state) && "hatched border-absent/30 bg-absent/[0.06]",
          )}
        >
          <span className="min-w-[11rem] font-mono text-xs">{r.check_id}</span>
          {r.state === "OK" ? (
            <Chip tone="accept" icon={Check}>
              ran
            </Chip>
          ) : (
            <AvailabilityChip state={r.state} exclusionReason={r.exclusion_reason} />
          )}
          <span className="text-ink-muted">
            {r.reason}
            {r.estimated_cost ? ` · estimated ${r.estimated_cost}` : ""}
          </span>
        </div>
      ))}
    </div>
  );
}

/* --------------------------------------------------- provenance summary
 * Always shown, including when clean. A report with no provenance section is
 * indistinguishable from a scan where provenance was never checked. */

export function ProvenanceSummary({
  summary,
}: {
  summary: import("@/lib/types").ProvenanceSummary | null;
}) {
  if (!summary) {
    return (
      <div className="hatched rounded-lg border border-dashed border-line-strong bg-surface-raised/40 px-6 py-8">
        <h3 className="mb-2 text-base font-semibold">
          No provenance section in this report
        </h3>
        <p className="max-w-[70ch] text-sm text-ink-muted">
          This scan carries no <code className="font-mono">provenance_summary</code>. No
          inference ledger was supplied, or no <code className="font-mono">prov.*</code>{" "}
          check resolved, so nothing about the field record&rsquo;s integrity was
          established either way.
        </p>
        <p className="mt-2 max-w-[70ch] text-sm text-ink-muted">
          That is not a clean provenance result. It is the absence of one.
        </p>
      </div>
    );
  }
  const cells: {
    label: string;
    value: number;
    note: string;
    tone?: "absent";
    hatch?: boolean;
  }[] = [
    {
      label: "Records verified",
      value: summary.records_verified,
      note: "chain and signatures checked",
    },
    {
      label: "Anchors checked",
      value: summary.anchors_checked,
      note: "external witnesses",
    },
    {
      label: "Unwitnessed window",
      value: summary.records_after_last_anchor,
      note: "records after the last anchor — this span still trusts the key holder",
      tone: "absent",
      hatch: summary.records_after_last_anchor > 0,
    },
    {
      label: "Declared gaps",
      value: summary.declared_degraded_intervals,
      note: "intervals the field unit admitted it could not seal",
      hatch: summary.declared_degraded_intervals > 0,
    },
  ];

  return (
    <div className="space-y-4">
      {summary.not_assessed_reason ? (
        <div className="hatched rounded border border-absent/30 bg-absent/[0.06] px-4 py-3">
          <strong className="block text-sm">
            An inference ledger was supplied, and nothing in it was checked
          </strong>
          <span className="text-sm text-ink-muted">{summary.not_assessed_reason}</span>
        </div>
      ) : null}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {cells.map((c) => (
          <Panel key={c.label} hatched={c.hatch} className="p-4">
            <Eyebrow>{c.label}</Eyebrow>
            <div
              className={cn(
                "font-mono text-3xl font-semibold leading-none tnum",
                c.tone === "absent" ? "text-absent" : "text-ink",
              )}
            >
              {c.value}
            </div>
            <p className="mt-1 text-xs text-ink-muted">{c.note}</p>
          </Panel>
        ))}
      </div>
      <div className="flex flex-wrap gap-2">
        <Chip tone="neutral" icon={KeyRound}>
          custody: {summary.custody_type}
        </Chip>
        <Chip tone="neutral" icon={Clock4}>
          durability: {summary.durability_window}
        </Chip>
        {summary.ledger_state ? (
          <Chip
            tone={
              summary.ledger_state === "sealed"
                ? "accept"
                : summary.ledger_state === "differs"
                  ? "quarantine"
                  : "absent"
            }
            icon={ShieldCheck}
          >
            ledger {summary.ledger_state.replace("_", " ")}
          </Chip>
        ) : null}
      </div>
    </div>
  );
}

/* ------------------------------------------------- contributor card (phone)
 * A seven-column table is unreadable at 390px and horizontal scrolling hides the column
 * that matters, so below `md` every contributor table becomes these cards. One component,
 * used by both the overview and the contributor view, so the two cannot drift. */

export function Attribution({ row }: { row: ContributorRow }) {
  if (row.contributor_source === "exif_cluster") {
    return (
      <div className="space-y-1">
        <Chip tone="absent">hypothesis</Chip>
        <p className="text-2xs text-ink-faint">
          camera-serial cluster, not a declared identity
        </p>
      </div>
    );
  }
  if (row.contributor_source) {
    return (
      <span className="font-mono text-2xs text-ink-muted">{row.contributor_source}</span>
    );
  }
  return <span className="text-ink-faint">—</span>;
}

export function TargetStatus({
  target,
}: {
  target?: import("@/lib/types").TargetState;
}) {
  if (target?.status === "quarantined") {
    return (
      <div className="space-y-1">
        <Chip tone="quarantine">held, seq {target.cited_seq}</Chip>
        {target.pending_release ? <Chip tone="pending">release pending</Chip> : null}
      </div>
    );
  }
  return <span className="text-2xs text-ink-faint">active</span>;
}

export function ContributorCard({
  row,
  target,
  showGroupKey,
  showStatus,
}: {
  row: ContributorRow;
  target?: import("@/lib/types").TargetState;
  showGroupKey?: boolean;
  /** The quarantine state is only known where `/targets` was fetched; where it was not,
   *  the card says nothing rather than implying "active". */
  showStatus?: boolean;
}) {
  return (
    <Panel className="p-4">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="break-all font-mono text-sm">{row.group_value}</span>
        {showGroupKey ? <Chip tone="neutral">{row.group_key}</Chip> : null}
        <DispositionChip value={row.disposition ?? "accept"} />
      </div>
      <div className="mb-3 flex flex-wrap gap-4 text-2xs text-ink-faint">
        <span className="tnum">
          n <span className="font-mono text-ink-muted">{row.n_samples}</span>
        </span>
        <span className="tnum">
          flagged <span className="font-mono text-ink-muted">{row.n_flagged}</span>
        </span>
        <span className="tnum">
          posterior{" "}
          <span className="font-mono text-ink-muted">{fixed(row.posterior_mean, 4)}</span>
        </span>
      </div>
      <p className="mt-1 font-mono text-2xs text-ink-faint tnum">
        {fixed(row.ci_low, 4)}–{fixed(row.ci_high, 4)}
        {row.cohort_rate_used !== undefined
          ? ` vs ${fixed(row.cohort_rate_used, 4)}`
          : ""}
      </p>
      <div className="mt-2 flex flex-wrap gap-2">
        <Attribution row={row} />
        {row.excludes_cohort_rate ? (
          <Chip tone="quarantine">excludes the cohort rate</Chip>
        ) : null}
        {showStatus ? <TargetStatus target={target} /> : null}
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------ host clock */

export function HostTime({ value }: { value: string | null | undefined }) {
  return (
    <span
      className="font-mono text-2xs text-ink-faint"
      title="host clock, untrusted — ordering is by ledger seq"
    >
      {hostTime(value)}
    </span>
  );
}
