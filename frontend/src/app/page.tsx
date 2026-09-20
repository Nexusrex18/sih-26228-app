"use client";

import { motion } from "framer-motion";
import { AlertTriangle, ArrowRight, FileWarning, Loader2 } from "lucide-react";
import Link from "next/link";
import * as React from "react";
import { DispositionChip, HostTime, SealBadge } from "@/components/domain";
import { Shell, useApp } from "@/components/shell";
import {
  Chip,
  Empty,
  Eyebrow,
  Panel,
  Section,
  SectionHead,
  Stat,
} from "@/components/ui/primitives";
import { api } from "@/lib/api";
import type { ScanSummary } from "@/lib/types";
import { cn } from "@/lib/ui";

export default function ScansPage() {
  const { session } = useApp();
  const [rows, setRows] = React.useState<ScanSummary[] | null>(null);
  const [ledgerReadable, setLedgerReadable] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!session?.authenticated) return;
    api
      .scans()
      .then((r) => {
        setRows(r.scans);
        setLedgerReadable(r.ledger_readable);
      })
      .catch((e: Error) => setError(e.message));
  }, [session?.authenticated]);

  const readable = (rows ?? []).filter((r) => r.readable);
  const totals = readable.reduce(
    (acc, r) => ({
      quarantine: acc.quarantine + r.counts.quarantine,
      review: acc.review + r.counts.review,
      accept: acc.accept + r.counts.accept,
    }),
    { quarantine: 0, review: 0, accept: 0 },
  );

  return (
    <Shell
      title="Scans"
      subtitle={
        rows
          ? `${rows.length} scan${rows.length === 1 ? "" : "s"} indexed`
          : "reading the report directory"
      }
      wide
    >
      {error ? (
        <Empty title="The scan list could not be read">
          <p>{error}</p>
        </Empty>
      ) : null}

      {rows === null && !error ? (
        <div className="flex items-center gap-3 py-16 text-ink-muted">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
          <span className="font-mono text-sm">loading scans</span>
        </div>
      ) : null}

      {rows && rows.length === 0 ? (
        <Empty title="No scans here yet">
          <p>
            This dashboard reads the report directory and expects the layout{" "}
            <code className="font-mono">cva scan</code> writes:{" "}
            <code className="font-mono">&lt;scan_id&gt;/report.json</code> beside a shared{" "}
            <code className="font-mono">evidence/</code> directory.
          </p>
          <p>
            An empty queue means nothing has been scanned. It does not mean anything has
            been cleared.
          </p>
        </Empty>
      ) : null}

      {rows && rows.length > 0 ? (
        <>
          <Section>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <Panel className="p-4">
                <Stat label="Scans" value={rows.length} countUp note="indexed on disk" />
              </Panel>
              <Panel className="p-4">
                <Stat
                  label="Quarantine"
                  value={totals.quarantine}
                  tone="quarantine"
                  countUp
                  note="findings held across all scans"
                />
              </Panel>
              <Panel className="p-4">
                <Stat
                  label="Review"
                  value={totals.review}
                  tone="review"
                  countUp
                  note="awaiting an analyst"
                />
              </Panel>
              <Panel className="p-4">
                <Stat
                  label="Accept"
                  value={totals.accept}
                  tone="accept"
                  countUp
                  note="routed by rule D7"
                />
              </Panel>
            </div>
          </Section>

          <Section delay={0.05}>
            <SectionHead
              title="Triage queue"
              note="Newest first. The seal badge compares the file on disk with the digest the ledger holds for it."
            />

            {!ledgerReadable ? (
              <div className="mb-4 flex items-center gap-2 rounded border border-absent/30 bg-absent/[0.06] px-4 py-2 text-sm text-ink-muted">
                <FileWarning className="h-4 w-4 shrink-0" aria-hidden />
                The ledger is unreachable, so no seal state could be checked. That is not
                the same as &ldquo;not sealed&rdquo;.
              </div>
            ) : null}

            <div className="space-y-2">
              {rows.map((r, i) => (
                <ScanRow key={r.scan_id} row={r} index={i} />
              ))}
            </div>
          </Section>
        </>
      ) : null}
    </Shell>
  );
}

function ScanRow({ row, index }: { row: ScanSummary; index: number }) {
  if (!row.readable) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: Math.min(index * 0.03, 0.3) }}
      >
        <Panel hatched className="p-4">
          <div className="flex flex-wrap items-center gap-3">
            <span className="font-mono text-sm">{row.scan_id}</span>
            <Chip tone="absent" icon={AlertTriangle}>
              unreadable
            </Chip>
            <span className="text-sm text-ink-muted">{row.unreadable_reason}</span>
          </div>
        </Panel>
      </motion.div>
    );
  }

  const total = Math.max(1, row.n_findings);
  const bars = [
    { key: "q", n: row.counts.quarantine, cls: "bg-quarantine" },
    { key: "r", n: row.counts.review, cls: "bg-review" },
    { key: "a", n: row.counts.accept, cls: "bg-accept" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(index * 0.03, 0.3) }}
    >
      <Link href={`/scan?id=${row.scan_id}`} className="group block">
        <Panel className="p-4 transition-colors group-hover:border-accent/40">
          <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
            <div className="min-w-[13rem]">
              <Eyebrow>scan</Eyebrow>
              <div className="font-mono text-sm text-ink">{row.scan_id}</div>
              <HostTime value={row.created_at_utc} />
            </div>

            <DispositionChip
              value={row.verdict.toLowerCase() as "accept" | "review" | "quarantine"}
              label={row.verdict}
            />

            <Chip tone="neutral">
              {row.profile_name} · {row.budget_tier}
            </Chip>

            <div className="min-w-[10rem] flex-1">
              <div className="mb-1 flex h-1.5 overflow-hidden rounded bg-line">
                {bars.map((b) =>
                  b.n > 0 ? (
                    <motion.i
                      key={b.key}
                      initial={{ width: 0 }}
                      animate={{ width: `${(100 * b.n) / total}%` }}
                      transition={{ duration: 0.5, delay: 0.1 }}
                      className={cn("block h-full", b.cls)}
                    />
                  ) : null,
                )}
              </div>
              <span className="font-mono text-2xs text-ink-faint tnum">
                {row.counts.quarantine} quarantine · {row.counts.review} review ·{" "}
                {row.counts.accept} accept
              </span>
            </div>

            <SealBadge seal={row.seal} />

            <ArrowRight
              className="h-4 w-4 shrink-0 text-ink-faint transition-transform group-hover:translate-x-0.5 group-hover:text-accent"
              aria-hidden
            />
          </div>
        </Panel>
      </Link>
    </motion.div>
  );
}
