"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown, Filter, Loader2 } from "lucide-react";
import { useSearchParams } from "next/navigation";
import * as React from "react";
import { toast } from "sonner";
import { DecideSheet } from "@/components/decide-sheet";
import {
  AvailabilityChip,
  DispositionChip,
  EffectiveState,
  PlanRows,
  SealBadge,
  SeverityText,
} from "@/components/domain";
import { ScanNav } from "@/components/scan-nav";
import { Shell, useApp } from "@/components/shell";
import {
  Button,
  Chip,
  Empty,
  Panel,
  Section,
  SectionHead,
} from "@/components/ui/primitives";
import { api } from "@/lib/api";
import type { Finding, PlanRow } from "@/lib/types";
import { cn, fixed, isAbsent } from "@/lib/ui";

export default function FindingsPage() {
  return (
    <React.Suspense fallback={null}>
      <Body />
    </React.Suspense>
  );
}

type Filters = {
  disposition?: string;
  nature?: string;
  availability?: string;
  module?: string;
};

function Body() {
  const params = useSearchParams();
  const scanId = params.get("id") ?? "";
  const { session, health, refreshHealth } = useApp();

  const [filters, setFilters] = React.useState<Filters>({});
  const [page, setPage] = React.useState(1);
  const [data, setData] = React.useState<{
    findings: Finding[];
    total: number;
    unfiltered_total: number;
    pages: number;
    modules: string[];
    plan: PlanRow[];
  } | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [seal, setSeal] = React.useState<import("@/lib/types").SealState | null>(null);
  const [open, setOpen] = React.useState<Finding | null>(null);

  const load = React.useCallback(async () => {
    if (!session?.authenticated || !scanId) return;
    setLoading(true);
    try {
      const r = await api.findings(scanId, { ...filters, page });
      setData(r);
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, [scanId, filters, page, session?.authenticated]);

  React.useEffect(() => {
    void load();
  }, [load]);

  React.useEffect(() => {
    if (!session?.authenticated || !scanId) return;
    api.scan(scanId).then((d) => setSeal(d.seal), () => undefined);
  }, [scanId, session?.authenticated]);

  const setFilter = (key: keyof Filters, value?: string) => {
    setPage(1);
    setFilters((f) => {
      const next = { ...f };
      if (!value || next[key] === value) delete next[key];
      else next[key] = value;
      return next;
    });
  };

  /* Keyboard triage. An analyst working a 10,000-finding page should not need a mouse. */
  const [cursor, setCursor] = React.useState(-1);
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName) || e.metaKey || e.ctrlKey)
        return;
      const n = data?.findings.length ?? 0;
      if (!n) return;
      if (e.key === "j") {
        e.preventDefault();
        setCursor((c) => Math.min(c + 1, n - 1));
      } else if (e.key === "k") {
        e.preventDefault();
        setCursor((c) => Math.max(c - 1, 0));
      } else if (e.key === "Enter" && cursor >= 0) {
        e.preventDefault();
        setOpen(data!.findings[cursor]);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [data, cursor]);

  return (
    <Shell
      title="Findings"
      subtitle={
        data ? `${scanId} · ${data.total} of ${data.unfiltered_total} shown` : scanId
      }
      actions={<SealBadge seal={seal} />}
      wide
    >
      <ScanNav scanId={scanId} n={data?.unfiltered_total} />

      <Section>
        <SectionHead
          title="Filter"
          note={
            <>
              Grouped by effective disposition. <strong>Nature</strong> is the
              &ldquo;deliberately or inadvertently&rdquo; distinction: duplicates and OOD
              are quality, trigger injection is adversarial, and they are not triaged the
              same way.
            </>
          }
        />
        <div className="space-y-2">
          <FilterRow
            label="Disposition"
            current={filters.disposition}
            options={["quarantine", "review", "accept"]}
            onPick={(v) => setFilter("disposition", v)}
          />
          <FilterRow
            label="Nature"
            current={filters.nature}
            options={["adversarial", "quality", "indeterminate"]}
            onPick={(v) => setFilter("nature", v)}
          />
          <FilterRow
            label="State"
            current={filters.availability}
            options={["UNAVAILABLE", "DEGRADED"]}
            onPick={(v) => setFilter("availability", v)}
          />
          {data?.modules.length ? (
            <FilterRow
              label="Module"
              current={filters.module}
              options={data.modules}
              onPick={(v) => setFilter("module", v)}
            />
          ) : null}
        </div>
        <p className="mt-3 text-2xs text-ink-faint">
          Keyboard: <Kbd>j</Kbd> <Kbd>k</Kbd> move, <Kbd>Enter</Kbd> decide.
        </p>
      </Section>

      <Section delay={0.04}>
        {loading && !data ? (
          <div className="flex items-center gap-3 py-16 text-ink-muted">
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
            <span className="font-mono text-sm">loading findings</span>
          </div>
        ) : null}

        {data && data.findings.length === 0 ? (
          <Empty
            title={
              data.unfiltered_total
                ? "No findings match this filter"
                : "This scan produced no findings"
            }
          >
            {data.unfiltered_total ? (
              <p>{data.unfiltered_total} findings exist under other filters.</p>
            ) : (
              <>
                <p>
                  Zero findings is a statement about <em>what was checked</em>. Here is
                  what ran and what did not — read that before reading the zero.
                </p>
                <div className="mt-4">
                  <PlanRows rows={data.plan} />
                </div>
              </>
            )}
          </Empty>
        ) : null}

        <div className="space-y-2">
          <AnimatePresence initial={false}>
            {data?.findings.map((f, i) => (
              <FindingCard
                key={f.finding_id}
                f={f}
                index={i}
                focused={i === cursor}
                onDecide={() => setOpen(f)}
              />
            ))}
          </AnimatePresence>
        </div>

        {data && data.pages > 1 ? (
          <div className="mt-6 flex items-center justify-center gap-3">
            <Button size="sm" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
              Previous
            </Button>
            <span className="font-mono text-xs text-ink-muted tnum">
              page {page} of {data.pages}
            </span>
            <Button
              size="sm"
              disabled={page >= data.pages}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
            </Button>
          </div>
        ) : null}
      </Section>

      <DecideSheet
        scanId={scanId}
        finding={open}
        onClose={() => setOpen(null)}
        writable={Boolean(health?.writable)}
        healthText={health?.text ?? ""}
        onRecorded={async () => {
          await refreshHealth();
          await load();
        }}
      />
    </Shell>
  );
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-block rounded-sm border border-line-strong border-b-2 bg-surface-raised px-1.5 font-mono text-2xs text-ink-muted">
      {children}
    </span>
  );
}

function FilterRow({
  label,
  current,
  options,
  onPick,
}: {
  label: string;
  current?: string;
  options: string[];
  onPick: (v?: string) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="flex w-full shrink-0 items-center gap-1.5 font-mono text-2xs uppercase tracking-wider text-ink-faint sm:w-[5.5rem]">
        <Filter className="h-3 w-3" aria-hidden />
        {label}
      </span>
      <button
        type="button"
        onClick={() => onPick(undefined)}
        aria-pressed={!current}
        className={chipCls(!current)}
      >
        any
      </button>
      {options.map((o) => (
        <button
          key={o}
          type="button"
          onClick={() => onPick(o)}
          aria-pressed={current === o}
          className={chipCls(current === o)}
        >
          {o}
        </button>
      ))}
    </div>
  );
}

function chipCls(active: boolean) {
  return cn(
    // 44px on a phone — Apple's minimum target, and these are the controls an analyst
    // taps most — dropping to 32px from `sm` up, where a pointer is precise and a row of
    // chunky pills would read as a toolbar. The row wraps rather than scrolling sideways,
    // so no filter is ever hidden off the edge.
    "inline-flex h-11 items-center rounded-full border px-4 font-mono text-2xs transition-colors",
    "sm:h-8 sm:px-3",
    "active:scale-[0.97]",
    active
      ? "border-accent bg-accent/10 text-accent"
      : "border-line bg-surface-panel text-ink-muted hover:border-line-strong hover:text-ink",
  );
}

function FindingCard({
  f,
  index,
  focused,
  onDecide,
}: {
  f: Finding;
  index: number;
  focused: boolean;
  onDecide: () => void;
}) {
  const [expanded, setExpanded] = React.useState(false);
  const absent = isAbsent(f.availability);
  const st = f.state;

  React.useEffect(() => {
    if (focused) {
      document
        .getElementById(`f-${f.finding_id}`)
        ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [focused, f.finding_id]);

  return (
    <motion.article
      id={`f-${f.finding_id}`}
      layout
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      transition={{ type: "spring", bounce: 0, duration: 0.35, delay: Math.min(index * 0.02, 0.2) }}
      className={cn(
        "overflow-hidden rounded-lg border bg-surface-panel transition-colors",
        focused ? "border-accent shadow-glow" : "border-line hover:border-line-strong",
      )}
    >
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        className="flex w-full items-start gap-3 px-4 py-3 text-left transition-colors hover:bg-surface-raised/40 active:bg-surface-raised/60"
      >
        <span className="hidden pt-0.5 font-mono text-2xs text-ink-faint sm:block">
          {f.finding_id.slice(0, 8)}
        </span>
        <span className="min-w-0 flex-1">
          <span className="mb-1.5 block break-words text-sm leading-relaxed text-ink">
            {f.reason}
          </span>
          <span className="flex flex-wrap items-center gap-2">
            <DispositionChip value={st?.effective ?? f.disposition} />
            {st?.pending ? <DispositionChip value="pending" label="pending" /> : null}
            <SeverityText value={f.severity} />
            <span className="font-mono text-2xs text-ink-faint tnum">
              conf {fixed(f.confidence, 2)}
            </span>
            <Chip tone="neutral">{f.nature}</Chip>
            <span className="hidden font-mono text-2xs text-ink-faint sm:inline">
              {f.target_type} {f.target_ref}
            </span>
            <AvailabilityChip
              state={f.availability}
              exclusionReason={f.exclusion_reason}
            />
          </span>
        </span>
        <ChevronDown
          className={cn(
            "mt-1 h-4 w-4 shrink-0 text-ink-faint transition-transform",
            expanded && "rotate-180",
          )}
          aria-hidden
        />
      </button>

      <AnimatePresence initial={false}>
        {expanded ? (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ type: "spring", bounce: 0, duration: 0.3 }}
            className={cn("border-t border-line", absent && "hatched")}
          >
            <div className="space-y-3 px-4 py-4 text-sm">
              <dl className="grid gap-x-6 gap-y-2 sm:grid-cols-[9rem_1fr]">
                <Dt>Rule that fired</Dt>
                <dd className="font-mono text-xs">{f.disposition_rule}</dd>
                <Dt>Attack class</Dt>
                <dd className="font-mono text-xs">{f.attack_class}</dd>
                <Dt>Produced by</Dt>
                <dd className="font-mono text-xs">
                  {f.produced_by} {f.detector_version}
                </dd>
                <Dt>Target</Dt>
                <dd className="break-all font-mono text-xs">
                  {f.target_type} {f.target_ref}
                </dd>
              </dl>

              {f.limitations.length ? (
                <div>
                  <Dt>Limitations</Dt>
                  {f.limitations.map((l, i) => (
                    <p key={i} className="mt-1 text-xs text-ink-muted">
                      {l}
                    </p>
                  ))}
                </div>
              ) : null}

              <div className="flex flex-wrap items-center gap-3 pt-1">
                <Button tone="primary" size="sm" onClick={onDecide}>
                  Open and decide
                </Button>
                {st ? (
                  <EffectiveState
                    state={st}
                    original={f.disposition}
                    originalRule={f.disposition_rule}
                  />
                ) : null}
              </div>
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </motion.article>
  );
}

function Dt({ children }: { children: React.ReactNode }) {
  return (
    <dt className="font-mono text-2xs uppercase tracking-wider text-ink-faint">
      {children}
    </dt>
  );
}
