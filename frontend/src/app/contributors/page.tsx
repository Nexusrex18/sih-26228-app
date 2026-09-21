"use client";

import { Loader2 } from "lucide-react";
import { useSearchParams } from "next/navigation";
import * as React from "react";
import { toast } from "sonner";
import {
  Attribution,
  ContributorCard,
  DispositionChip,
  SealBadge,
  TargetStatus,
} from "@/components/domain";
import { ChartCard, ContributorScatter, ForestPlot } from "@/components/charts";
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
                "inline-flex h-11 items-center rounded-full border px-4 font-mono text-xs transition-colors active:scale-[0.98] sm:h-9",
                group === g
                  ? "border-accent bg-accent/10 text-accent"
                  : "border-line bg-surface-panel text-ink-muted hover:border-line-strong hover:text-ink",
              )}
            >
              {g}
            </button>
          ))}
        </div>
        {rows.length ? (
          <div className="mt-4">
            <ChartCard
              title="Sample size against posterior"
              note="bubble area = flagged samples · red = interval excludes its cohort rate"
            >
              <ContributorScatter
                rows={rows.map((r) => ({
                  label: r.group_value,
                  n: Math.max(1, r.n_samples),
                  posterior: r.posterior_mean,
                  flagged: r.n_flagged,
                  excludes: Boolean(r.excludes_cohort_rate),
                }))}
                cohortRate={
                  baseline && typeof baseline.cohort_rate === "number"
                    ? baseline.cohort_rate
                    : undefined
                }
              />
              <p className="mt-2 text-2xs text-ink-faint">
                A small contributor sits far up the axis on little evidence; a large one
                close to the line is a lot of evidence of little. The interval, not the
                height, is what decides.
              </p>
            </ChartCard>
          </div>
        ) : null}
      </Section>

      <Section delay={0.04}>
        {rows.length ? (
          <>
            <ChartCard
              title="95% intervals"
              note="point = posterior · bar = 95% interval · red = excludes its cohort rate"
              className="mb-4"
            >
              <ForestPlot
                rows={rows.map((r) => ({
                  label: r.group_value,
                  mean: r.posterior_mean,
                  lo: r.ci_low,
                  hi: r.ci_high,
                  cohort: r.cohort_rate_used,
                  excludes: Boolean(r.excludes_cohort_rate),
                  n: r.n_samples,
                }))}
                cohortRate={
                  baseline && typeof baseline.cohort_rate === "number"
                    ? baseline.cohort_rate
                    : undefined
                }
              />
            </ChartCard>
            {/* Cards on a phone, a table from `md` up. A seven-column table is unreadable
                at 390px and horizontal scrolling hides the column that matters. */}
            <div className="space-y-2 md:hidden">
              {rows.map((r) => (
                <ContributorCard
                  key={r.group_value}
                  row={r}
                  target={findTarget(targets, r)}
                  showStatus
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
                            <TargetStatus target={t} />
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

function Dt({ children }: { children: React.ReactNode }) {
  return (
    <dt className="font-mono text-2xs uppercase tracking-wider text-ink-faint">
      {children}
    </dt>
  );
}
