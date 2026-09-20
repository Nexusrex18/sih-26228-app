"use client";

import { Check, CircleDashed, Circle, Loader2 } from "lucide-react";
import { useSearchParams } from "next/navigation";
import * as React from "react";
import { toast } from "sonner";
import { SealBadge } from "@/components/domain";
import { ReliabilityDiagram } from "@/components/reliability";
import { ScanNav } from "@/components/scan-nav";
import { Shell, useApp } from "@/components/shell";
import {
  Banner,
  Button,
  Chip,
  Empty,
  Eyebrow,
  Panel,
  Section,
  SectionHead,
  Stat,
} from "@/components/ui/primitives";
import { api } from "@/lib/api";
import type { Calibration, Coverage, CoverageRow, ScanSummary, SealState } from "@/lib/types";
import { cn, pct } from "@/lib/ui";

export default function CoveragePage() {
  return (
    <React.Suspense fallback={null}>
      <Body />
    </React.Suspense>
  );
}

interface Payload {
  scan_id: string;
  profile_name: string;
  budget_tier: string;
  coverage: Coverage;
  calibration: Calibration | null;
  other?: {
    scan_id: string;
    profile_name: string;
    budget_tier: string;
    coverage: Coverage;
  };
  diff?: {
    only_left: string[];
    only_right: string[];
    both: string[];
    shrank: boolean;
    lost_reasons: Record<string, string[]>;
  };
}

function Body() {
  const params = useSearchParams();
  const scanId = params.get("id") ?? "";
  const { session } = useApp();
  const [data, setData] = React.useState<Payload | null>(null);
  const [seal, setSeal] = React.useState<SealState | null>(null);
  const [candidates, setCandidates] = React.useState<ScanSummary[]>([]);
  const [compare, setCompare] = React.useState("");
  const [busy, setBusy] = React.useState(false);

  React.useEffect(() => {
    if (!session?.authenticated || !scanId) return;
    api.coverage(scanId).then(setData, (e: Error) => toast.error(e.message));
    api.scan(scanId).then((d) => setSeal(d.seal), () => undefined);
    api.scans().then((r) => setCandidates(r.scans.filter((s) => s.readable)), () => undefined);
  }, [scanId, session?.authenticated]);

  const runCompare = async (other: string) => {
    setBusy(true);
    try {
      setData(await api.coverage(scanId, other || undefined));
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  if (!data) {
    return (
      <Shell title="Coverage statement" subtitle={scanId}>
        <div className="flex items-center gap-3 py-16 text-ink-muted">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
          <span className="font-mono text-sm">loading</span>
        </div>
      </Shell>
    );
  }

  const c = data.coverage;

  return (
    <Shell
      title="Coverage statement"
      subtitle={`${scanId} · ${data.profile_name} profile, ${data.budget_tier} tier`}
      actions={<SealBadge seal={seal} />}
      wide
    >
      <ScanNav scanId={scanId} />

      <Section>
        <p className="max-w-[75ch] text-sm text-ink-muted">
          This statement is <strong className="text-ink">generated</strong> by walking the
          detector registry against this scan&rsquo;s capabilities — never hand-written. It
          is also <strong className="text-ink">per scan</strong>: the same asset under a
          narrower profile produces a smaller statement, and that is the capability model
          working, not a defect.
        </p>
        <p className="mt-2 max-w-[75ch] text-sm text-ink-muted">
          Only attack classes are counted. Operational reports — a declared gap, a clock
          regression — are true statements about the system&rsquo;s own state, not things an
          attacker does, and counting them would inflate the statement.
        </p>
      </Section>

      <Section delay={0.04}>
        <div className="grid gap-4 sm:grid-cols-3">
          <Panel className="p-4">
            <Stat
              label="Assessed"
              value={c.assessed.length}
              countUp
              tone="accept"
              note="a check ran for this class"
            />
          </Panel>
          <Panel hatched className="p-4">
            <Stat
              label="Not assessed"
              value={c.not_assessed.length}
              countUp
              tone="absent"
              note="a check exists and did not run"
            />
          </Panel>
          <Panel hatched className="p-4">
            <Stat
              label="Never covered"
              value={c.never_covered.length}
              countUp
              tone="absent"
              note="in the taxonomy, covered by nothing we have"
            />
          </Panel>
        </div>
        <p className="mt-3 font-mono text-2xs text-ink-faint tnum">
          {pct(c.assessed_fraction)} of {c.total_attack_classes} attack classes in the
          taxonomy, as this report states it.
        </p>
      </Section>

      <Section delay={0.06}>
        <SectionHead title="Covered by a check that ran" />
        {c.assessed.length ? (
          <ClassTable rows={c.assessed} tone="accept" head="Checks" />
        ) : (
          <Empty title="Nothing was assessed in this scan" hatched>
            <p>
              No attack class had a check that ran. A report this short is a small statement
              about what was examined, not a clean result.
            </p>
          </Empty>
        )}
      </Section>

      <Section delay={0.08}>
        <SectionHead
          title="Covered by a check that did not run"
          note="The check exists. It was UNAVAILABLE or DEGRADED, with a reason."
        />
        {c.not_assessed.length ? (
          <ClassTable
            rows={c.not_assessed}
            tone="absent"
            head="Checks that would have covered it"
            hatched
          />
        ) : (
          <p className="text-sm text-ink-muted">
            Every registered check that covers a class ran.
          </p>
        )}
      </Section>

      <Section delay={0.1}>
        <SectionHead
          title="Covered by nothing"
          note="Declared, never hidden. This is the explicit non-coverage statement PS §2.2.5 asks for."
        />
        {c.never_covered.length ? (
          <Panel hatched className="flex flex-wrap gap-2 p-4">
            {c.never_covered.map((k) => (
              <Chip key={k} tone="absent" icon={Circle}>
                {k}
              </Chip>
            ))}
          </Panel>
        ) : (
          <p className="text-sm text-ink-muted">
            Every attack class in the taxonomy has a check.
          </p>
        )}
      </Section>

      {c.operational_reports.length ? (
        <Section delay={0.12}>
          <SectionHead title="Operational reports" />
          <p className="mb-3 max-w-[70ch] text-sm text-ink-muted">
            True statements about the system&rsquo;s own state, not attack classes. They are
            listed and deliberately not counted.
          </p>
          <div className="flex flex-wrap gap-2">
            {c.operational_reports.map((k) => (
              <Chip key={k} tone="neutral">
                {k}
              </Chip>
            ))}
          </div>
        </Section>
      ) : null}

      {/* The reliability diagram lives INSIDE the coverage statement, not in an appendix:
          the system's own confidence is evidence, and evidence gets shown. */}
      <Section delay={0.14}>
        <SectionHead title="Calibration" />
        {data.calibration ? (
          <div className="grid gap-4 lg:grid-cols-2">
            <Panel className="p-4">
              <Stat
                label="Brier score"
                value={
                  data.calibration.brier === null
                    ? "—"
                    : data.calibration.brier.toPrecision(4)
                }
                note={
                  data.calibration.scored_on === "out_of_fold" ? (
                    <>
                      {data.calibration.method}, scored out of fold
                    </>
                  ) : (
                    <>
                      {data.calibration.method}, <strong>not scored</strong> — there was no
                      second group to hold out, so no number is reported rather than an
                      in-sample one presented as held out
                    </>
                  )
                }
              />
              <div className="mt-4">
                <ReliabilityDiagram calibration={data.calibration} />
              </div>
            </Panel>

            <Panel className="p-4">
              <h3 className="mb-3 text-sm font-semibold">
                Which confidences are on the calibrated scale
              </h3>
              <div className="flex flex-wrap gap-2">
                {data.calibration.calibrated_detectors.length ? (
                  data.calibration.calibrated_detectors.map((d) => (
                    <Chip key={d} tone="accept" icon={Check}>
                      {d}
                    </Chip>
                  ))
                ) : (
                  <span className="text-sm text-ink-muted">None.</span>
                )}
              </div>
              <p className="mt-3 max-w-[60ch] text-sm text-ink-muted">
                Every other finding carries its detector&rsquo;s own raw score, and the
                disposition table&rsquo;s confidence thresholds are not applied to those:
                they route on severity alone and are capped at review.
              </p>
              {data.calibration.excluded_detectors.length ? (
                <>
                  <h3 className="mb-3 mt-5 text-sm font-semibold">
                    Excluded from calibration
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {data.calibration.excluded_detectors.map((d) => (
                      <Chip key={d} tone="absent" icon={Circle}>
                        {d}
                      </Chip>
                    ))}
                  </div>
                  <p className="mt-3 max-w-[60ch] text-sm text-ink-muted">
                    A hash mismatch is arithmetic, not a belief. Fitting a curve to a
                    deterministic check would produce a calibrated probability on a
                    certainty.
                  </p>
                </>
              ) : null}
            </Panel>
          </div>
        ) : (
          <Empty title="No calibration for this scan" hatched>
            <p>
              No calibration set was supplied, so every confidence here is its
              detector&rsquo;s own raw score and no reliability claim is made about any of
              them.
            </p>
          </Empty>
        )}
      </Section>

      <Section delay={0.16}>
        <SectionHead
          title="Standing limitations"
          note="Code declares what it does; humans declare what the code depends on."
        />
        <Panel className="space-y-2 p-4">
          {c.standing_limitations.length ? (
            c.standing_limitations.map((s, i) => (
              <p key={i} className="max-w-[80ch] text-sm text-ink-muted">
                {s}
              </p>
            ))
          ) : (
            <p className="max-w-[80ch] text-sm text-ink-muted">
              This report carries no standing limitations, which is itself worth checking:{" "}
              <code className="font-mono text-xs">docs/coverage-standing.yaml</code> is the
              reviewed file they come from.
            </p>
          )}
        </Panel>
      </Section>

      {/* Two scans side by side: the point that coverage is per scan. */}
      <Section delay={0.18}>
        <SectionHead title="Compare with another scan" />
        <div className="flex flex-wrap items-center gap-2">
          <label htmlFor="cmp" className="sr-only">
            Compare with
          </label>
          <select
            id="cmp"
            value={compare}
            onChange={(e) => setCompare(e.target.value)}
            className="h-11 min-w-[16rem] max-w-full rounded border border-line-strong bg-surface-panel px-3 font-mono text-xs text-ink"
          >
            <option value="">choose a scan</option>
            {candidates
              .filter((s) => s.scan_id !== scanId)
              .map((s) => (
                <option key={s.scan_id} value={s.scan_id}>
                  {s.scan_id} · {s.profile_name} / {s.budget_tier}
                </option>
              ))}
          </select>
          <Button
            tone="primary"
            className="h-11"
            disabled={busy || !compare}
            icon={busy ? Loader2 : undefined}
            onClick={() => void runCompare(compare)}
          >
            Compare
          </Button>
          {data.other ? (
            <Button
              className="h-11"
              onClick={() => {
                setCompare("");
                void runCompare("");
              }}
            >
              Clear
            </Button>
          ) : null}
        </div>

        {data.other && data.diff ? (
          <div className="mt-5 space-y-4">
            <div className="grid gap-4 lg:grid-cols-2">
              <StatementSize
                scanId={data.scan_id}
                label={`${data.profile_name} / ${data.budget_tier}`}
                coverage={data.coverage}
                here
              />
              <StatementSize
                scanId={data.other.scan_id}
                label={`${data.other.profile_name} / ${data.other.budget_tier}`}
                coverage={data.other.coverage}
              />
            </div>

            {data.diff.shrank ? (
              <Banner tone="absent" title="The other scan makes a smaller statement">
                {data.diff.only_left.length} attack class(es) assessed here are not assessed
                there. A smaller statement under a narrower profile is the capability model
                working — and it is also less that anyone may conclude from that scan.
              </Banner>
            ) : null}

            {data.diff.only_left.length ? (
              <Panel hatched className="overflow-hidden">
                <div className="border-b border-line px-4 py-3">
                  <h3 className="text-sm font-semibold">Assessed here and not there</h3>
                  <p className="text-xs text-ink-muted">
                    What the other scan&rsquo;s profile gave up, and why its statement is
                    smaller.
                  </p>
                </div>
                <div className="overflow-x-auto scroll-slim">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-line-strong">
                        <th className="px-3 py-2 text-left font-mono text-2xs uppercase tracking-wider text-ink-faint">
                          Attack class
                        </th>
                        <th className="px-3 py-2 text-left font-mono text-2xs uppercase tracking-wider text-ink-faint">
                          In {data.other.scan_id} it would have needed
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.diff.only_left.map((k) => (
                        <tr key={k} className="border-b border-line last:border-0">
                          <td className="px-3 py-2 font-mono text-xs">{k}</td>
                          <td className="px-3 py-2">
                            {(data.diff?.lost_reasons[k] ?? []).length ? (
                              <span className="flex flex-wrap gap-1.5">
                                {(data.diff?.lost_reasons[k] ?? []).map((chk) => (
                                  <Chip key={chk} tone="absent" icon={CircleDashed}>
                                    {chk}
                                  </Chip>
                                ))}
                              </span>
                            ) : (
                              <span className="text-2xs text-ink-faint">
                                covered by nothing there
                              </span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>
            ) : null}

            {data.diff.only_right.length ? (
              <Panel className="p-4">
                <h3 className="mb-3 text-sm font-semibold">Assessed there and not here</h3>
                <div className="flex flex-wrap gap-2">
                  {data.diff.only_right.map((k) => (
                    <Chip key={k} tone="accept" icon={Check}>
                      {k}
                    </Chip>
                  ))}
                </div>
              </Panel>
            ) : null}
          </div>
        ) : null}
      </Section>
    </Shell>
  );
}

/** The size of the statement, drawn to scale. The bar is `assessed_fraction` exactly as the
 *  report states it — the dashboard computes no coverage row of its own (D-E2). */
function StatementSize({
  scanId,
  label,
  coverage,
  here,
}: {
  scanId: string;
  label: string;
  coverage: Coverage;
  here?: boolean;
}) {
  const frac = Math.max(0, Math.min(1, coverage.assessed_fraction));
  return (
    <Panel className={cn("p-4", here && "border-accent/40")}>
      <div className="flex flex-wrap items-baseline gap-2">
        <span className="font-mono text-sm">{scanId}</span>
        {here ? <Chip tone="accent">this scan</Chip> : null}
      </div>
      <p className="mb-3 font-mono text-2xs text-ink-faint">{label}</p>
      <Eyebrow>Assessed</Eyebrow>
      <div className="mt-1 flex items-center gap-3">
        <span className="font-mono text-3xl font-semibold leading-none tnum">
          {coverage.assessed.length}
        </span>
        <span className="font-mono text-2xs text-ink-faint tnum">
          of {coverage.total_attack_classes} · {pct(coverage.assessed_fraction)}
        </span>
      </div>
      <div
        className="mt-3 h-3 w-full overflow-hidden rounded-sm border border-line bg-surface-deep"
        role="img"
        aria-label={`${pct(coverage.assessed_fraction)} of attack classes assessed`}
      >
        <div
          className="h-full bg-accent/60"
          style={{ width: `${frac * 100}%` }}
        />
      </div>
      <p className="mt-2 text-2xs text-ink-faint">
        The bar is the statement&rsquo;s size, to scale. Two bars of different lengths are
        two different claims about what was examined.
      </p>
    </Panel>
  );
}

function ClassTable({
  rows,
  tone,
  head,
  hatched,
}: {
  rows: CoverageRow[];
  tone: "accept" | "absent";
  head: string;
  hatched?: boolean;
}) {
  return (
    <Panel hatched={hatched} className="overflow-hidden">
      <div className="overflow-x-auto scroll-slim">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-line-strong">
              <th className="px-3 py-2 text-left font-mono text-2xs font-semibold uppercase tracking-wider text-ink-faint">
                Attack class
              </th>
              <th className="px-3 py-2 text-left font-mono text-2xs font-semibold uppercase tracking-wider text-ink-faint">
                {head}
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.attack_class} className="border-b border-line last:border-0">
                <td className="px-3 py-3 align-top font-mono text-xs">{r.attack_class}</td>
                <td className="px-3 py-3">
                  <span className="flex flex-wrap gap-1.5">
                    {r.checks.map((chk) => (
                      <Chip
                        key={chk}
                        tone={tone}
                        icon={tone === "accept" ? Check : CircleDashed}
                      >
                        {chk}
                      </Chip>
                    ))}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
