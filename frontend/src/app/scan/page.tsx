"use client";

import { AlertOctagon, ArrowLeft, Loader2 } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import * as React from "react";
import {
  AvailabilityChip,
  CIBar,
  DispositionChip,
  PlanRows,
  ProvenanceSummary,
  SealBadge,
} from "@/components/domain";
import { ScanNav } from "@/components/scan-nav";
import { Shell, useApp } from "@/components/shell";
import {
  Banner,
  Button,
  Chip,
  Empty,
  Eyebrow,
  Hash,
  Panel,
  Section,
  SectionHead,
  Stat,
} from "@/components/ui/primitives";
import { api } from "@/lib/api";
import type { ScanDetail } from "@/lib/types";
import { fixed } from "@/lib/ui";

export default function ScanPage() {
  return (
    <React.Suspense fallback={<Loading />}>
      <ScanBody />
    </React.Suspense>
  );
}

function Loading() {
  return (
    <div className="flex items-center gap-3 py-16 text-ink-muted">
      <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
      <span className="font-mono text-sm">loading</span>
    </div>
  );
}

function ScanBody() {
  const params = useSearchParams();
  const scanId = params.get("id") ?? "";
  const { session } = useApp();
  const [d, setD] = React.useState<ScanDetail | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!session?.authenticated || !scanId) return;
    setD(null);
    setError(null);
    api.scan(scanId).then(setD, (e: Error) => setError(e.message));
  }, [scanId, session?.authenticated]);

  if (!scanId) {
    return (
      <Shell title="No scan selected">
        <Empty title="No scan id in the address">
          <p>
            <Link className="text-accent" href="/">
              Pick one from the queue
            </Link>
            .
          </p>
        </Empty>
      </Shell>
    );
  }

  if (error) {
    return (
      <Shell title={scanId} subtitle="unreadable">
        <Empty title={`Scan ${scanId} cannot be read`}>
          <p>{error}</p>
          <p>
            It is not rendered best-effort: a report that mostly renders is a report that
            can hide a field.
          </p>
        </Empty>
      </Shell>
    );
  }

  if (!d) {
    return (
      <Shell title={scanId}>
        <Loading />
      </Shell>
    );
  }

  const verdict = d.verdict.toLowerCase() as "accept" | "review" | "quarantine";
  const contributors = (d.contributor_risk ?? []).slice(0, 8);
  const ciScale = Math.max(
    ...((d.contributor_risk ?? []).map((r) => r.ci_high) || [1]),
    0.0001,
  );

  return (
    <Shell
      title={scanId}
      subtitle={`${d.produced_by.profile_name} · ${d.produced_by.budget_tier} tier · written ${d.created_at_utc.slice(0, 19).replace("T", " ")} (host clock, untrusted)`}
      actions={<SealBadge seal={d.seal} />}
      wide
    >
      <ScanNav scanId={scanId} n={d.n_findings} />

      {d.seal.is_alarm ? (
        <div className="mb-6">
          <Banner
            tone="alarm"
            alarm
            icon={AlertOctagon}
            title="This report differs from the digest the ledger sealed"
          >
            {d.seal.detail}
          </Banner>
        </div>
      ) : null}

      {/* 1 — VERDICT */}
      <Section>
        <Panel
          className={`relative overflow-hidden p-6 ${
            verdict === "quarantine"
              ? "border-quarantine/35 bg-quarantine/[0.07]"
              : verdict === "review"
                ? "border-review/35 bg-review/[0.07]"
                : "border-accept/35 bg-accept/[0.07]"
          }`}
        >
          <div className="flex flex-wrap items-center gap-6">
            <div>
              <Eyebrow>Verdict</Eyebrow>
              <div
                className={`font-mono text-5xl font-bold leading-none tracking-tighter ${
                  verdict === "quarantine"
                    ? "text-quarantine"
                    : verdict === "review"
                      ? "text-review"
                      : "text-accept"
                }`}
              >
                {d.verdict}
              </div>
            </div>
            <div className="min-w-0 space-y-2">
              <div className="flex flex-wrap gap-2">
                <DispositionChip
                  value="quarantine"
                  label={`${d.effective.quarantine ?? 0} quarantine`}
                />
                <DispositionChip value="review" label={`${d.effective.review ?? 0} review`} />
                <DispositionChip value="accept" label={`${d.effective.accept ?? 0} accept`} />
                {d.effective.pending ? (
                  <DispositionChip
                    value="pending"
                    label={`${d.effective.pending} pending approval`}
                  />
                ) : null}
              </div>
              <p className="max-w-[60ch] text-xs text-ink-muted">
                Counts are the <strong>effective</strong> state: the risk engine&rsquo;s
                dispositions with every recorded human decision folded over them. The
                tool&rsquo;s own values stay visible on each finding, beside the effective
                one.
              </p>
            </div>
            <div className="ml-auto">
              <Link href={`/findings?id=${scanId}`}>
                <Button tone="primary">Open findings</Button>
              </Link>
            </div>
          </div>
        </Panel>
      </Section>

      {/* 2 — ACCESS ASSUMPTIONS */}
      <Section delay={0.04}>
        <SectionHead
          eyebrow="02"
          title="Access assumptions"
          note="What this scan could see, and what it could not. Every absence carries a reason."
        />
        <div className="grid gap-4 lg:grid-cols-2">
          <Panel className="p-4">
            <h3 className="mb-3 text-sm font-semibold">Present</h3>
            <div className="flex flex-wrap gap-2">
              {d.access_assumptions.capabilities_present.length ? (
                d.access_assumptions.capabilities_present.map((c) => (
                  <Chip key={c} tone="accept">
                    {c}
                  </Chip>
                ))
              ) : (
                <p className="text-sm text-ink-muted">
                  No capability resolved. Nothing in this scan inspected anything.
                </p>
              )}
            </div>
          </Panel>
          <Panel hatched className="p-4">
            <h3 className="mb-3 text-sm font-semibold">Absent</h3>
            <div className="space-y-1">
              {d.access_assumptions.capabilities_absent.length ? (
                d.access_assumptions.capabilities_absent.map((c) => (
                  <div
                    key={c.capability}
                    className="flex flex-wrap items-baseline gap-3 rounded border border-line px-3 py-2 text-sm"
                  >
                    <span className="min-w-[11rem] font-mono text-xs">{c.capability}</span>
                    <span className="text-ink-muted">{c.reason}</span>
                  </div>
                ))
              ) : (
                <p className="text-sm text-ink-muted">
                  Every capability this scan asked for resolved.
                </p>
              )}
            </div>
          </Panel>
        </div>
        {d.access_assumptions.consequence ? (
          <div className="mt-4">
            <Banner tone="info" title="Consequence">
              {d.access_assumptions.consequence}
            </Banner>
          </div>
        ) : null}

        <details className="mt-4 rounded-lg border border-line bg-surface-panel p-4">
          <summary className="cursor-pointer text-sm font-semibold">
            Check plan — {d.plan.filter((p) => p.state === "OK").length} of {d.plan.length}{" "}
            ran
          </summary>
          <p className="my-3 max-w-[70ch] text-sm text-ink-muted">
            Resolved and printed before any work started, so coverage was known at minute
            zero. All four states appear at equal prominence: a shorter report must never
            look like a cleaner result.
          </p>
          <PlanRows rows={d.plan} />
        </details>
      </Section>

      {/* 3 — CONTRIBUTOR RISK */}
      <Section delay={0.08}>
        <SectionHead
          eyebrow="03"
          title="Contributor risk"
          note="Before individual findings, because it is what an acceptance analyst acts on."
          actions={
            <Link href={`/contributors?id=${scanId}`}>
              <Button size="sm">All {(d.contributor_risk ?? []).length}</Button>
            </Link>
          }
        />
        {contributors.length ? (
          <Panel className="overflow-hidden">
            <div className="overflow-x-auto scroll-slim">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-line-strong">
                    {[
                      "Group",
                      "Attribution",
                      "n",
                      "flagged",
                      "posterior",
                      "95% interval",
                      "Disposition",
                    ].map((h) => (
                      <th
                        key={h}
                        className="whitespace-nowrap px-3 py-2 text-left font-mono text-2xs font-semibold uppercase tracking-wider text-ink-faint"
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {contributors.map((r) => (
                    <tr
                      key={`${r.group_key}:${r.group_value}`}
                      className="border-b border-line transition-colors last:border-0 hover:bg-surface-raised/40"
                    >
                      <td className="px-3 py-3">
                        <span className="font-mono">{r.group_value}</span>{" "}
                        <Chip tone="neutral">{r.group_key}</Chip>
                      </td>
                      <td className="px-3 py-3">
                        {r.contributor_source === "exif_cluster" ? (
                          <Chip tone="absent">hypothesis · exif_cluster</Chip>
                        ) : r.contributor_source ? (
                          <span className="font-mono text-2xs text-ink-muted">
                            {r.contributor_source}
                          </span>
                        ) : (
                          <span className="text-ink-faint">—</span>
                        )}
                      </td>
                      <td className="px-3 py-3 text-right font-mono tnum">{r.n_samples}</td>
                      <td className="px-3 py-3 text-right font-mono tnum">{r.n_flagged}</td>
                      <td className="px-3 py-3 text-right font-mono tnum">
                        {fixed(r.posterior_mean)}
                      </td>
                      <td className="px-3 py-3">
                        <CIBar row={r} scale={ciScale} />
                        <span className="font-mono text-2xs text-ink-faint tnum">
                          {fixed(r.ci_low)}–{fixed(r.ci_high)}
                          {r.cohort_rate_used !== undefined
                            ? ` vs ${fixed(r.cohort_rate_used)}`
                            : ""}
                        </span>
                      </td>
                      <td className="px-3 py-3">
                        <DispositionChip value={r.disposition ?? "accept"} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        ) : (
          <Empty title="No contributor rows">
            <p>
              No grouping key resolved for this dataset, so no source-level aggregation was
              possible. That is a limit of what was supplied, not a finding that every
              contributor is sound.
            </p>
          </Empty>
        )}
      </Section>

      {/* 5 — PROVENANCE */}
      <Section delay={0.12}>
        <SectionHead
          eyebrow="05"
          title="Provenance"
          actions={
            <Link href={`/provenance?id=${scanId}`}>
              <Button size="sm">Detail</Button>
            </Link>
          }
        />
        <ProvenanceSummary summary={d.provenance_summary} />
      </Section>

      {/* 7 — COVERAGE */}
      <Section delay={0.16}>
        <SectionHead
          eyebrow="07"
          title="Coverage"
          actions={
            <Link href={`/coverage?id=${scanId}`}>
              <Button size="sm">Full statement</Button>
            </Link>
          }
        />
        <div className="grid gap-4 sm:grid-cols-3">
          <Panel className="p-4">
            <Stat
              label="Assessed"
              value={d.coverage.assessed.length}
              tone="accept"
              countUp
              note="attack classes a check actually ran for"
            />
          </Panel>
          <Panel hatched className="p-4">
            <Stat
              label="Not assessed"
              value={d.coverage.not_assessed.length}
              tone="absent"
              countUp
              note="covered by a check that was UNAVAILABLE or DEGRADED"
            />
          </Panel>
          <Panel hatched className="p-4">
            <Stat
              label="Never covered"
              value={d.coverage.never_covered.length}
              tone="absent"
              countUp
              note="in the taxonomy and covered by nothing we have"
            />
          </Panel>
        </div>
      </Section>

      {/* 8 — REPRODUCTION */}
      <Section delay={0.2}>
        <SectionHead eyebrow="08" title="Reproduction" />
        <Panel className="p-4">
          <dl className="grid gap-x-6 gap-y-3 sm:grid-cols-[10rem_1fr]">
            <dt className="font-mono text-2xs uppercase tracking-wider text-ink-faint">
              Command
            </dt>
            <dd className="min-w-0">
              <code className="block break-all rounded-sm border border-line bg-surface-deep px-2 py-1 font-mono text-xs">
                {d.reproduction.command}
              </code>
            </dd>
            <dt className="font-mono text-2xs uppercase tracking-wider text-ink-faint">
              Profile hash
            </dt>
            <dd>
              <Hash value={d.produced_by.profile_hash} />
            </dd>
            <dt className="font-mono text-2xs uppercase tracking-wider text-ink-faint">
              Code commit
            </dt>
            <dd>
              <Hash value={d.produced_by.code_commit} />
            </dd>
            <dt className="font-mono text-2xs uppercase tracking-wider text-ink-faint">
              report.json sha256
            </dt>
            <dd>
              <Hash value={d.report_sha256} />
            </dd>
          </dl>
        </Panel>
      </Section>
    </Shell>
  );
}
