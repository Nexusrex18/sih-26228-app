"use client";

import { Flag, Loader2, ScrollText } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import * as React from "react";
import { toast } from "sonner";
import { ActorBars, ChartCard, SeqActivity } from "@/components/charts";
import { DispositionChip, SeqItem, SeqRail } from "@/components/domain";
import { Shell, useApp } from "@/components/shell";
import {
  Chip,
  Empty,
  Panel,
  Section,
  SectionHead,
} from "@/components/ui/primitives";
import { api } from "@/lib/api";
import type { Disposition, ScanSummary, TimelineRow } from "@/lib/types";
import { cn, hostTime } from "@/lib/ui";

export default function AuditPage() {
  return (
    <React.Suspense fallback={null}>
      <Body />
    </React.Suspense>
  );
}

const DISPOSITIONS = new Set(["accept", "review", "quarantine"]);

function Body() {
  const params = useSearchParams();
  const queryScan = params.get("id") ?? "";
  const { session, health } = useApp();
  const [scanId, setScanId] = React.useState(queryScan);
  const [data, setData] = React.useState<{
    readable: boolean;
    decisions: number;
    rows: TimelineRow[];
  } | null>(null);
  const [scans, setScans] = React.useState<ScanSummary[]>([]);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => setScanId(queryScan), [queryScan]);

  React.useEffect(() => {
    if (!session?.authenticated) return;
    setLoading(true);
    api
      .audit(scanId || undefined)
      .then(setData, (e: Error) => toast.error(e.message))
      .finally(() => setLoading(false));
  }, [scanId, session?.authenticated]);

  React.useEffect(() => {
    if (!session?.authenticated) return;
    api.scans().then((r) => setScans(r.scans.filter((s) => s.readable)), () => undefined);
  }, [session?.authenticated]);

  const rows = data?.rows ?? [];

  return (
    <Shell
      title="Audit trail"
      subtitle={
        data
          ? `${rows.length} record${rows.length === 1 ? "" : "s"} · ${data.decisions} analyst decision${data.decisions === 1 ? "" : "s"}`
          : undefined
      }
      wide
    >
      <Section>
        <p className="max-w-[78ch] text-sm text-ink-muted">
          Every <code className="font-mono text-xs">scan_record</code> and{" "}
          <code className="font-mono text-xs">analyst_event</code> in ledger order.{" "}
          <strong className="text-ink">
            Ordering is by <span className="font-mono">seq</span>, not by time
          </strong>
          : the host clock is untrusted, and a clock that steps backwards changes what is
          displayed and never what is ordered.
        </p>
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => setScanId("")}
            aria-pressed={!scanId}
            className={filterCls(!scanId)}
          >
            All scans
          </button>
          {scans.map((s) => (
            <button
              key={s.scan_id}
              type="button"
              onClick={() => setScanId(s.scan_id)}
              aria-pressed={scanId === s.scan_id}
              className={filterCls(scanId === s.scan_id)}
            >
              {s.scan_id}
            </button>
          ))}
          <span className="flex-1" />
          <Link
            href="/verification"
            className={cn(filterCls(false), "gap-2 no-underline")}
          >
            <ScrollText className="h-3 w-3" aria-hidden />
            Verification detail
          </Link>
        </div>
      </Section>

      {data && data.readable && rows.length ? (
        <Section delay={0.02}>
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
            <ChartCard title="Decisions per person" note="analyst events, by recording account">
              {(() => {
                const byActor = new Map<string, number>();
                for (const r of rows) {
                  if (r.kind === "analyst_event") byActor.set(r.actor, (byActor.get(r.actor) ?? 0) + 1);
                }
                const actors = [...byActor.entries()]
                  .map(([actor, n]) => ({ actor, n }))
                  .sort((a, b) => b.n - a.n);
                return actors.length ? (
                  <ActorBars data={actors} />
                ) : (
                  <p className="py-8 text-center text-sm text-ink-muted">
                    No analyst has recorded a decision yet.
                  </p>
                );
              })()}
            </ChartCard>
            <ChartCard title="Activity along the ledger" note="x is ledger seq, never time">
              <SeqActivity rows={rows} />
            </ChartCard>
          </div>
        </Section>
      ) : null}

      <Section delay={0.04}>
        {loading && !data ? (
          <div className="flex items-center gap-3 py-16 text-ink-muted">
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
            <span className="font-mono text-sm">reading the ledger</span>
          </div>
        ) : data && !data.readable ? (
          <Empty title="The ledger could not be read" hatched>
            <p>{health?.detail || health?.text || "The ledger was not reachable."}</p>
            <p>
              This is not an empty audit trail. It is an audit trail that could not be
              opened, and the difference matters.
            </p>
          </Empty>
        ) : rows.length === 0 ? (
          <Empty title="The ledger holds no scan or decision records yet">
            <p>
              A scan seals a <code className="font-mono text-xs">scan_record</code> when it
              finishes and a signing key is configured. An analyst decision appends an{" "}
              <code className="font-mono text-xs">analyst_event</code>. Neither has happened
              here.
            </p>
          </Empty>
        ) : (
          <SeqRail>
            {rows.map((r, i) => (
              <SeqItem
                key={r.seq}
                seq={r.seq}
                index={i}
                tone={
                  r.audit_flagged
                    ? "flagged"
                    : r.kind === "analyst_event"
                      ? "decision"
                      : "neutral"
                }
              >
                <Row row={r} />
              </SeqItem>
            ))}
          </SeqRail>
        )}
      </Section>

      <Section delay={0.08}>
        <SectionHead title="What this trail does not record" />
        <Panel hatched className="space-y-3 p-4">
          <p className="max-w-[80ch] text-sm text-ink-muted">
            <strong className="text-ink">
              Attribution is by the recording tool, not by the analyst&rsquo;s own key.
            </strong>{" "}
            <code className="font-mono text-xs">actor_id</code> is asserted by{" "}
            <code className="font-mono text-xs">cva-web</code> after local authentication.
            Anyone holding the <code className="font-mono text-xs">cva-web</code> account and
            a valid session could record a decision as that user. Per-analyst signing keys
            would close this and need key distribution on an air gap; they are declared as a
            limitation until built.
          </p>
          <p className="max-w-[80ch] text-sm text-ink-muted">
            Login and lockout events go to the application log, not to the ledger, in this
            version.
          </p>
          <p className="max-w-[80ch] text-sm text-ink-muted">
            Displayed times come from the host clock and are not evidence of when anything
            happened. The <span className="font-mono text-xs">seq</span> column is.
          </p>
        </Panel>
      </Section>
    </Shell>
  );
}

function Row({ row }: { row: TimelineRow }) {
  return (
    <Panel className="p-3.5">
      <div className="flex flex-wrap items-center gap-2">
        <strong className="break-all font-mono text-sm">{row.actor}</strong>
        {row.role ? <Chip tone="neutral">{row.role}</Chip> : null}
        <Chip tone="neutral">{row.action}</Chip>
        {row.kind === "analyst_event" && DISPOSITIONS.has(row.detail) ? (
          <DispositionChip value={row.detail as Disposition} />
        ) : null}
        {row.reason_code ? (
          <span className="font-mono text-2xs text-ink-faint">{row.reason_code}</span>
        ) : null}
        {row.refs_seq !== null ? (
          <span className="text-2xs text-ink-faint">
            approves seq <span className="font-mono">{row.refs_seq}</span>
          </span>
        ) : null}
        {row.audit_flagged ? (
          <Chip tone="quarantine" icon={Flag}>
            flagged for audit
          </Chip>
        ) : null}
        <span className="flex-1" />
        {row.scan_id ? (
          <Link
            href={`/scan?id=${row.scan_id}`}
            className="font-mono text-2xs text-ink-muted transition-colors hover:text-accent"
          >
            {row.scan_id}
          </Link>
        ) : null}
        <span
          className="font-mono text-2xs text-ink-faint"
          title="host clock, untrusted — ordering is by ledger seq"
        >
          {hostTime(row.created_at_utc)}
        </span>
      </div>
      {row.target ? (
        <p className="mt-1.5 break-all font-mono text-2xs text-ink-faint">{row.target}</p>
      ) : null}
      {row.kind === "scan_record" && row.detail ? (
        <p className="mt-1.5 break-words text-sm text-ink-muted">{row.detail}</p>
      ) : null}
      {row.justification ? (
        <blockquote className="mt-2 border-l-2 border-line-strong pl-3 text-sm text-ink-muted">
          {row.justification}
        </blockquote>
      ) : null}
      {row.key_id ? (
        <p className="mt-2 text-2xs text-ink-faint">
          signed under key{" "}
          <span className="font-mono">{row.key_id.slice(0, 12)}…</span>
        </p>
      ) : null}
    </Panel>
  );
}

function filterCls(active: boolean) {
  return cn(
    // 44px on a phone, 36px from `sm` up (see the findings filters for the reasoning).
    "inline-flex h-11 items-center rounded-full border px-4 font-mono text-2xs transition-colors active:scale-[0.97]",
    "sm:h-9 sm:px-3.5",
    active
      ? "border-accent bg-accent/10 text-accent"
      : "border-line bg-surface-panel text-ink-muted hover:border-line-strong hover:text-ink",
  );
}
