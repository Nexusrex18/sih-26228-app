"use client";

import { Loader2 } from "lucide-react";
import { useSearchParams } from "next/navigation";
import * as React from "react";
import { toast } from "sonner";
import { CIBar, DispositionChip, SealBadge } from "@/components/domain";
import { ScanNav } from "@/components/scan-nav";
import { Shell, useApp } from "@/components/shell";
import {
  Banner,
  Chip,
  Empty,
  Panel,
  Section,
  SectionHead,
  Stat,
} from "@/components/ui/primitives";
import { api } from "@/lib/api";
import type { ContributorRow, ScanDetail, TargetState } from "@/lib/types";
import { cn, fixed } from "@/lib/ui";

export default function ContributorsPage() {
  return (
    <React.Suspense fallback={null}>
      <Body />
    </React.Suspense>
  );
}

function Body() {
  const params = useSearchParams();
  const scanId = params.get("id") ?? "";
  const { session } = useApp();
  const [d, setD] = React.useState<ScanDetail | null>(null);
  const [targets, setTargets] = React.useState<TargetState[]>([]);
  const [group, setGroup] = React.useState("contributor");

  React.useEffect(() => {
    if (!session?.authenticated || !scanId) return;
    api.scan(scanId).then(setD, (e: Error) => toast.error(e.message));
    api.targets(scanId).then((r) => setTargets(r.targets), () => undefined);
  }, [scanId, session?.authenticated]);

  if (!d) {
    return (
      <Shell title="Contributor risk" subtitle={scanId}>
        <div className="flex items-center gap-3 py-16 text-ink-muted">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
          <span className="font-mono text-sm">loading</span>
        </div>
      </Shell>
    );
  }

  const all = d.contributor_risk ?? [];
  const groups = Array.from(new Set(all.map((r) => r.group_key)));
  const rows = all.filter((r) => r.group_key === group);
  const scale = Math.max(...rows.map((r) => r.ci_high), 0.0001);
  const baseline = d.contributor_baseline as Record<string, number | boolean> | null;

  return (
    <Shell
      title="Contributor risk"
      subtitle={`${scanId} · grouped by ${group}`}
      actions={<SealBadge seal={d.seal} />}
      wide
    >
      <ScanNav scanId={scanId} n={d.n_findings} />

      <Section>
        <SectionHead
          title="Source-level aggregation"
          note="A beta-binomial posterior, never a raw flag rate: 3 of 5 is 60% and must not outrank 800 of 10,000 at 8%."
        />
        <div className="flex flex-wrap gap-2">
          {groups.map((g) => (
            <button
              key={g}
              type="button"
              onClick={() => setGroup(g)}
              aria-pressed={group === g}
              className={cn(
                "inline-flex h-9 items-center rounded-full border px-4 font-mono text-xs transition-colors active:scale-[0.98]",
                group === g
                  ? "border-accent bg-accent/10 text-accent"
                  : "border-line bg-surface-panel text-ink-muted hover:border-line-strong hover:text-ink",
              )}
            >
              {g}
            </button>
          ))}
        </div>
      </Section>

      <Section delay={0.04}>
        {rows.length ? (
          <>
            {/* Cards on a phone, a table from `md` up. A seven-column table is unreadable
                at 390px and horizontal scrolling hides the column that matters. */}
            <div className="space-y-2 md:hidden">
              {rows.map((r) => (
                <ContributorCard
                  key={r.group_value}
                  row={r}
                  scale={scale}
                  target={findTarget(targets, r)}
                />
              ))}
            </div>
            <Panel className="hidden overflow-hidden md:block">
              <div className="overflow-x-auto scroll-slim">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-line-strong">
                      {[
                        group,
                        "Attribution",
                        "n",
                        "flagged",
                        "posterior",
                        "95% interval vs its cohort rate",
                        "Disposition",
                        "Status",
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
                    {rows.map((r) => {
                      const t = findTarget(targets, r);
                      return (
                        <tr
                          key={r.group_value}
                          className="border-b border-line transition-colors last:border-0 hover:bg-surface-raised/40"
                        >
                          <td className="px-3 py-3 font-mono">{r.group_value}</td>
                          <td className="px-3 py-3">
                            <Attribution row={r} />
                          </td>
                          <td className="px-3 py-3 text-right font-mono tnum">
                            {r.n_samples}
                          </td>
                          <td className="px-3 py-3 text-right font-mono tnum">
                            {r.n_flagged}
                          </td>
                          <td className="px-3 py-3 text-right font-mono tnum">
                            {fixed(r.posterior_mean, 4)}
                          </td>
                          <td className="px-3 py-3">
                            <CIBar row={r} scale={scale} />
                            <span className="font-mono text-2xs text-ink-faint tnum">
                              {fixed(r.ci_low, 4)}–{fixed(r.ci_high, 4)}
                              {r.cohort_rate_used !== undefined
                                ? ` vs ${fixed(r.cohort_rate_used, 4)} (leave-one-out)`
                                : ""}
                            </span>
                            {r.excludes_cohort_rate ? (
                              <div className="mt-1">
                                <Chip tone="quarantine">
                                  interval excludes the cohort rate
                                </Chip>
                              </div>
                            ) : null}
                          </td>
                          <td className="px-3 py-3">
                            <DispositionChip value={r.disposition ?? "accept"} />
                          </td>
                          <td className="px-3 py-3">
                            <Status target={t} />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Panel>
          </>
        ) : (
          <Empty title="No rows for this grouping">
            <p>
              No <span className="font-mono">{group}</span> key resolved for this dataset.
              Contributor attribution runs a precedence — signed sidecar, directory
              convention, format field, then camera-serial clustering — and tier 4 is not
              implemented, so a dataset identifiable only that way resolves nothing rather
              than a weak guess.
            </p>
          </Empty>
        )}
      </Section>

      <Section delay={0.08}>
        <div className="grid gap-4 lg:grid-cols-2">
          <Panel hatched={!baseline} className="p-4">
            <h3 className="mb-3 text-sm font-semibold">Absolute baseline</h3>
            {baseline ? (
              <>
                <dl className="grid gap-x-6 gap-y-2 sm:grid-cols-[10rem_1fr] text-sm">
                  <Dt>Reference set</Dt>
                  <dd className="font-mono tnum">
                    {String(baseline.reference_flagged)}/{String(baseline.reference_n)} ={" "}
                    {fixed(Number(baseline.reference_rate), 4)}
                  </dd>
                  <Dt>Upper 95% bound</Dt>
                  <dd className="font-mono tnum">
                    {fixed(Number(baseline.reference_rate_ci_high), 4)}
                  </dd>
                  <Dt>This cohort</Dt>
                  <dd className="font-mono tnum">
                    {fixed(Number(baseline.cohort_rate), 4)} (all survivors)
                  </dd>
                </dl>
                {baseline.cohort_exceeds_reference ? (
                  <div className="mt-3">
                    <Banner tone="warn" title="The cohort prior is unreliable here">
                      The baseline every group is compared with is itself dirtier than clean
                      data, so &ldquo;nobody stands out&rdquo; would not mean
                      &ldquo;nothing is wrong&rdquo;.
                    </Banner>
                  </div>
                ) : null}
              </>
            ) : (
              <div className="space-y-2 text-sm text-ink-muted">
                <p>
                  {d.contributor_baseline_unavailable ??
                    "No reference dataset was supplied with --reference-dataset."}
                </p>
                <p>
                  The cohort prior has one blind spot and this is it: a cohort contaminated
                  as a whole, where no contributor stands out.
                </p>
              </div>
            )}
          </Panel>

          <Panel hatched={!d.permutation_test} className="p-4">
            <h3 className="mb-3 text-sm font-semibold">Permutation test</h3>
            {d.permutation_test ? (
              <>
                <Stat
                  label="p"
                  value={d.permutation_test.p_value.toPrecision(3)}
                  note={`statistic ${fixed(d.permutation_test.statistic, 4)} over ${d.permutation_test.n_permutations} permutations`}
                />
                <p className="mt-3 text-sm text-ink-muted">
                  {d.permutation_test.conclusion}
                </p>
                <p className="mt-2 text-sm text-ink-muted">
                  This is the Sybil case the cohort prior structurally cannot see: are flags
                  non-randomly distributed across groups at all?
                </p>
              </>
            ) : (
              <p className="text-sm text-ink-muted">Not computed for this scan.</p>
            )}
          </Panel>
        </div>
      </Section>
    </Shell>
  );
}

function findTarget(targets: TargetState[], row: ContributorRow) {
  return targets.find(
    (t) => t.target_type === row.group_key && t.target_ref === row.group_value,
  );
}

function Attribution({ row }: { row: ContributorRow }) {
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

function Status({ target }: { target?: TargetState }) {
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

function ContributorCard({
  row,
  scale,
  target,
}: {
  row: ContributorRow;
  scale: number;
  target?: TargetState;
}) {
  return (
    <Panel className="p-4">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="break-all font-mono text-sm">{row.group_value}</span>
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
          <span className="font-mono text-ink-muted">
            {fixed(row.posterior_mean, 4)}
          </span>
        </span>
      </div>
      <CIBar row={row} scale={scale} />
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
        <Status target={target} />
      </div>
    </Panel>
  );
}

function Dt({ children }: { children: React.ReactNode }) {
  return (
    <dt className="font-mono text-2xs uppercase tracking-wider text-ink-faint">
      {children}
    </dt>
  );
}
