"use client";

import { motion } from "framer-motion";
import {
  AlertOctagon,
  Check,
  CircleDashed,
  Clock4,
  Flag,
  type LucideIcon,
} from "lucide-react";
import { PatternLines } from "@visx/pattern";
import * as React from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  Pie,
  PieChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { CountUp, Eyebrow, usePrefersReducedMotion } from "@/components/ui/primitives";
import { cn, fixed } from "@/lib/ui";

/**
 * The chart kit. One place for every chart, so they read as one system.
 *
 * Colour follows the job, never the chart:
 *  - disposition charts use the RESERVED status hues (quarantine / review / accept /
 *    pending), always with an icon and a word in the legend beside them;
 *  - magnitude (severity, counts, confidence) uses ONE hue — the structural accent —
 *    stepped in lightness, never a rainbow;
 *  - "did not run" (UNAVAILABLE / DEGRADED) is the HATCH, never a faded fill. A faded
 *    slice reads as "small"; a hatched one reads as "absent", which is the truth.
 *
 * Nothing here computes a disposition, a confidence or a coverage number. Every chart
 * counts or groups what the API returned.
 */

/* ------------------------------------------------------------------ tones */

export type Tone = "quarantine" | "review" | "accept" | "pending" | "absent" | "accent";

export const TONE_FILL: Record<Tone, string> = {
  quarantine: "hsl(var(--quarantine))",
  review: "hsl(var(--review))",
  accept: "hsl(var(--accept))",
  pending: "hsl(var(--pending))",
  absent: "url(#cva-hatch)",
  accent: "hsl(var(--accent))",
};

const TONE_ICON: Record<Tone, LucideIcon | null> = {
  quarantine: AlertOctagon,
  review: Flag,
  accept: Check,
  pending: Clock4,
  absent: CircleDashed,
  accent: null,
};

const TONE_TEXT: Record<Tone, string> = {
  quarantine: "text-quarantine",
  review: "text-review",
  accept: "text-accept",
  pending: "text-pending",
  absent: "text-absent",
  accent: "text-accent",
};

/** The hatch as an SVG pattern, for inside charts (visx's PatternLines). Place once per
 *  <svg> and reference it as `fill="url(#cva-hatch)"`. Panels keep the CSS `.hatched`. */
export function HatchDefs() {
  return (
    <defs>
      <PatternLines
        id="cva-hatch"
        width={6}
        height={6}
        stroke="hsl(var(--absent))"
        strokeWidth={1.5}
        background="hsl(var(--absent) / 0.14)"
        orientation={["diagonal"]}
      />
    </defs>
  );
}

/** Lightness steps of the accent for ordinal magnitude (severity): one hue, light→dark. */
const ACCENT_STEPS = [
  "hsl(var(--accent) / 0.28)",
  "hsl(var(--accent) / 0.42)",
  "hsl(var(--accent) / 0.58)",
  "hsl(var(--accent) / 0.76)",
  "hsl(var(--accent) / 0.95)",
];

/* ------------------------------------------------------------------ frame */

export function ChartCard({
  title,
  note,
  children,
  className,
  actions,
}: {
  title: string;
  note?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  actions?: React.ReactNode;
}) {
  return (
    <div className={cn("glass rounded-xl border border-line/80 p-4", className)}>
      <div className="mb-3 flex flex-wrap items-baseline gap-2">
        <h3 className="font-display text-sm font-semibold tracking-wide text-ink-strong">{title}</h3>
        {note ? <span className="text-2xs text-ink-faint">{note}</span> : null}
        {actions ? <div className="ml-auto">{actions}</div> : null}
      </div>
      {children}
    </div>
  );
}

function Tip({ children }: { children: React.ReactNode }) {
  return <div className="chart-tip">{children}</div>;
}

/* ------------------------------------------------------------------ legend */

export function Legend({
  items,
  active,
  onPick,
}: {
  items: { key: string; label: string; value: number; tone: Tone }[];
  active?: string | null;
  onPick?: (key: string) => void;
}) {
  return (
    <ul className="space-y-1.5">
      {items.map((it) => {
        const Icon = TONE_ICON[it.tone];
        const body = (
          <>
            <span
              aria-hidden
              className={cn(
                "h-2.5 w-2.5 shrink-0 rounded-sm",
                it.tone === "absent" ? "hatched border border-absent/50" : "",
              )}
              style={it.tone === "absent" ? undefined : { background: TONE_FILL[it.tone] }}
            />
            {Icon ? <Icon className={cn("h-3.5 w-3.5 shrink-0", TONE_TEXT[it.tone])} aria-hidden /> : null}
            <span className="min-w-0 flex-1 truncate text-xs text-ink-muted">{it.label}</span>
            <span className="font-mono text-xs text-ink tnum">{it.value}</span>
          </>
        );
        return (
          <li key={it.key}>
            {onPick ? (
              <button
                type="button"
                onClick={() => onPick(it.key)}
                aria-pressed={active === it.key}
                className={cn(
                  "flex w-full items-center gap-2 rounded px-1.5 py-1 text-left transition-colors",
                  active === it.key ? "bg-surface-raised" : "hover:bg-surface-raised/60",
                )}
              >
                {body}
              </button>
            ) : (
              <div className="flex items-center gap-2 px-1.5 py-1">{body}</div>
            )}
          </li>
        );
      })}
    </ul>
  );
}

/* ------------------------------------------------------------------ donut */

export function Donut({
  data,
  centerValue,
  centerLabel,
  active,
  onPick,
  stack,
}: {
  data: { key: string; label: string; value: number; tone: Tone }[];
  centerValue: number | string;
  centerLabel: string;
  active?: string | null;
  onPick?: (key: string) => void;
  /** Legend below the ring rather than beside it — for narrow cards. */
  stack?: boolean;
}) {
  const reduced = usePrefersReducedMotion();
  const [hover, setHover] = React.useState<string | null>(null);
  const shown = data.filter((d) => d.value > 0);
  const total = shown.reduce((a, d) => a + d.value, 0);
  const focus = hover ?? active ?? null;

  return (
    <div
      className={cn(
        "grid items-center gap-4",
        !stack && "sm:grid-cols-[minmax(0,180px)_1fr]",
      )}
    >
      <div
        className="relative mx-auto aspect-square w-full max-w-[180px]"
        role="img"
        aria-label={`${centerLabel}: ${data.map((d) => `${d.label} ${d.value}`).join(", ")}`}
      >
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <HatchDefs />
            <Pie
              data={total ? shown : [{ key: "none", label: "none", value: 1, tone: "absent" as Tone }]}
              dataKey="value"
              nameKey="label"
              innerRadius="68%"
              outerRadius="96%"
              paddingAngle={shown.length > 1 ? 2 : 0}
              cornerRadius={4}
              stroke="hsl(var(--surface-panel))"
              strokeWidth={2}
              isAnimationActive={!reduced} animationDuration={400} animationEasing="ease-out"
              onMouseEnter={(_, i) => setHover(shown[i]?.key ?? null)}
              onMouseLeave={() => setHover(null)}
              onClick={(_, i) => shown[i] && onPick?.(shown[i].key)}
              style={{ cursor: onPick ? "pointer" : "default", outline: "none" }}
            >
              {(total ? shown : [{ key: "none", tone: "absent" as Tone }]).map((d) => (
                <Cell
                  key={d.key}
                  fill={TONE_FILL[d.tone]}
                  opacity={focus && focus !== d.key ? 0.35 : 1}
                  style={{ transition: "opacity 200ms ease" }}
                />
              ))}
            </Pie>
            <Tooltip
              content={({ active: a, payload }) =>
                a && payload?.length && total ? (
                  <Tip>
                    <p className="font-mono text-ink">{String(payload[0].name)}</p>
                    <p className="font-mono text-ink-muted tnum">
                      {Number(payload[0].value)} · {fixed((100 * Number(payload[0].value)) / total, 1)}%
                    </p>
                  </Tip>
                ) : null
              }
            />
          </PieChart>
        </ResponsiveContainer>
        <div className="pointer-events-none absolute inset-0 grid place-items-center text-center">
          <div>
            <div className="font-mono text-3xl font-semibold leading-none tnum">
              {typeof centerValue === "number" ? <CountUp value={centerValue} /> : centerValue}
            </div>
            <div className="mt-1 font-mono text-2xs uppercase tracking-wider text-ink-faint">
              {centerLabel}
            </div>
          </div>
        </div>
      </div>
      <Legend items={data} active={active} onPick={onPick} />
    </div>
  );
}

/* ------------------------------------------------------------------ radial gauge */

export function Gauge({
  fraction,
  label,
  sub,
  tone = "accent",
}: {
  fraction: number;
  label: string;
  sub?: string;
  tone?: Tone;
}) {
  const reduced = usePrefersReducedMotion();
  const f = Math.max(0, Math.min(1, fraction));
  const r = 52;
  const c = 2 * Math.PI * r;
  const arc = 0.75; // a 270° instrument dial
  return (
    <div className="flex flex-col items-center" role="img" aria-label={`${label}: ${(f * 100).toFixed(1)}%`}>
      <div className="relative h-[150px] w-[150px]">
        <svg viewBox="0 0 120 120" className="h-full w-full -rotate-[225deg]">
          <circle
            cx="60"
            cy="60"
            r={r}
            fill="none"
            stroke="hsl(var(--line-strong))"
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={`${c * arc} ${c}`}
          />
          <motion.circle
            cx="60"
            cy="60"
            r={r}
            fill="none"
            stroke={TONE_FILL[tone]}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={`${c * arc} ${c}`}
            initial={{ strokeDashoffset: reduced ? c * arc * (1 - f) : c * arc }}
            animate={{ strokeDashoffset: c * arc * (1 - f) }}
            transition={{ duration: reduced ? 0 : 0.5, ease: [0.23, 1, 0.32, 1] }}
          />
          {/* Tick marks at every 10%, so the dial reads as an instrument, not a ring. */}
          {Array.from({ length: 11 }, (_, i) => {
            const a = ((i / 10) * arc) * 2 * Math.PI;
            const x1 = 60 + Math.cos(a) * 42;
            const y1 = 60 + Math.sin(a) * 42;
            const x2 = 60 + Math.cos(a) * (i % 5 === 0 ? 37 : 39.5);
            const y2 = 60 + Math.sin(a) * (i % 5 === 0 ? 37 : 39.5);
            return (
              <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="hsl(var(--ink-faint))" strokeWidth="1" />
            );
          })}
        </svg>
        <div className="absolute inset-0 grid place-items-center text-center">
          <div>
            <div className="font-mono text-3xl font-semibold leading-none tnum">
              <CountUp value={Math.round(f * 100)} />
              <span className="text-lg text-ink-faint">%</span>
            </div>
            {sub ? <div className="mt-1 font-mono text-2xs text-ink-faint">{sub}</div> : null}
          </div>
        </div>
      </div>
      <Eyebrow className="mt-1">{label}</Eyebrow>
    </div>
  );
}

/* ------------------------------------------------------------------ segmented bar */

/** One bar, whole = 100%, segments to scale. Absent segments are hatched. */
export function ProportionBar({
  segments,
}: {
  segments: { key: string; label: string; value: number; tone: Tone }[];
}) {
  const reduced = usePrefersReducedMotion();
  const total = segments.reduce((a, s) => a + s.value, 0);
  return (
    <div className="space-y-3">
      <div
        className="flex h-3.5 w-full gap-[2px] overflow-hidden rounded-full bg-surface-deep"
        role="img"
        aria-label={segments.map((s) => `${s.label} ${s.value}`).join(", ")}
      >
        {total === 0 ? (
          <div className="hatched h-full w-full" />
        ) : (
          segments
            .filter((s) => s.value > 0)
            .map((s, i) => (
              <motion.div
                key={s.key}
                title={`${s.label}: ${s.value}`}
                className={cn("h-full first:rounded-l-full last:rounded-r-full", s.tone === "absent" && "hatched")}
                style={{
                  width: `${(100 * s.value) / total}%`,
                  background: s.tone === "absent" ? undefined : TONE_FILL[s.tone],
                  transformOrigin: "left",
                }}
                initial={{ transform: reduced ? "scaleX(1)" : "scaleX(0)" }}
                animate={{ transform: "scaleX(1)" }}
                transition={{ duration: 0.35, delay: reduced ? 0 : i * 0.05, ease: [0.23, 1, 0.32, 1] }}
              />
            ))
        )}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1.5">
        {segments.map((s) => {
          const Icon = TONE_ICON[s.tone];
          return (
            <span key={s.key} className="inline-flex items-center gap-1.5 text-xs text-ink-muted">
              <span
                aria-hidden
                className={cn("h-2.5 w-2.5 rounded-sm", s.tone === "absent" && "hatched border border-absent/50")}
                style={s.tone === "absent" ? undefined : { background: TONE_FILL[s.tone] }}
              />
              {Icon ? <Icon className={cn("h-3 w-3", TONE_TEXT[s.tone])} aria-hidden /> : null}
              {s.label}
              <span className="font-mono text-ink tnum">{s.value}</span>
            </span>
          );
        })}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ bar list */

/** Ranked horizontal bars (severity). One hue stepped in lightness: an ordinal scale. */
export function BarList({
  data,
}: {
  data: { key: string; label: string; value: number }[];
}) {
  const reduced = usePrefersReducedMotion();
  const max = Math.max(1, ...data.map((d) => d.value));
  return (
    <div className="space-y-2">
      {data.map((d, i) => (
        <div key={d.key} className="grid grid-cols-[4.5rem_1fr_2.5rem] items-center gap-3">
          <span className="font-mono text-2xs uppercase tracking-wider text-ink-muted">{d.label}</span>
          <div className="h-2.5 overflow-hidden rounded-full bg-surface-deep" title={`${d.label}: ${d.value}`}>
            <motion.div
              className="h-full rounded-full"
              style={{
                width: `${(100 * d.value) / max}%`,
                background: ACCENT_STEPS[Math.max(0, ACCENT_STEPS.length - 1 - i)],
                transformOrigin: "left",
              }}
              initial={{ transform: reduced ? "scaleX(1)" : "scaleX(0)" }}
              animate={{ transform: "scaleX(1)" }}
              transition={{ duration: 0.35, delay: reduced ? 0 : i * 0.04, ease: [0.23, 1, 0.32, 1] }}
            />
          </div>
          <span className="text-right font-mono text-xs text-ink tnum">{d.value}</span>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ stacked bars */

export function StackedDispositionBars({
  rows,
  onPick,
}: {
  rows: { key: string; label: string; quarantine: number; review: number; accept: number }[];
  onPick?: (key: string) => void;
}) {
  const reduced = usePrefersReducedMotion();
  const height = Math.max(120, rows.length * 38 + 30);
  return (
    <div style={{ height }} role="img" aria-label="Findings by disposition, per scan">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 8, bottom: 4, left: 0 }} barCategoryGap={10}>
          <CartesianGrid horizontal={false} stroke="hsl(var(--line))" strokeDasharray="2 4" />
          <XAxis type="number" tick={{ fill: "hsl(var(--ink-faint))", fontSize: 11 }} stroke="hsl(var(--line-strong))" allowDecimals={false} />
          <YAxis
            type="category"
            dataKey="label"
            width={72}
            tick={{ fill: "hsl(var(--ink-muted))", fontSize: 11, fontFamily: "var(--font-mono)" }}
            stroke="hsl(var(--line-strong))"
          />
          <Tooltip
            cursor={{ fill: "hsl(var(--accent) / 0.06)" }}
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <Tip>
                  <p className="mb-1 font-mono text-ink">{String(label)}</p>
                  {payload.map((p) => (
                    <p key={String(p.dataKey)} className="font-mono text-ink-muted tnum">
                      {String(p.dataKey)} {Number(p.value)}
                    </p>
                  ))}
                </Tip>
              ) : null
            }
          />
          {(["quarantine", "review", "accept"] as const).map((k, i, arr) => (
            <Bar
              key={k}
              dataKey={k}
              stackId="d"
              fill={TONE_FILL[k]}
              stroke="hsl(var(--surface-panel))"
              strokeWidth={2}
              radius={i === arr.length - 1 ? [0, 4, 4, 0] : 0}
              isAnimationActive={!reduced} animationDuration={400} animationEasing="ease-out"
              onClick={(d: { key?: string }) => d?.key && onPick?.(d.key)}
              style={{ cursor: onPick ? "pointer" : "default" }}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ------------------------------------------------------------------ heat grid */

/** Counts by two categorical axes. Intensity = count, one hue. Click a column to filter. */
export function HeatGrid({
  rows,
  cols,
  count,
  activeCol,
  onCol,
}: {
  rows: string[];
  cols: string[];
  count: (row: string, col: string) => number;
  activeCol?: string | null;
  onCol?: (col: string) => void;
}) {
  const max = Math.max(1, ...rows.flatMap((r) => cols.map((c) => count(r, c))));
  return (
    <div className="overflow-hidden">
      <div
        className="grid gap-1.5"
        style={{ gridTemplateColumns: `4.5rem repeat(${cols.length}, minmax(0, 1fr))` }}
      >
        <span />
        {cols.map((c) => (
          <button
            key={c}
            type="button"
            onClick={() => onCol?.(c)}
            aria-pressed={activeCol === c}
            className={cn(
              "min-h-11 truncate rounded px-1 py-1 text-center font-mono text-2xs transition-colors sm:min-h-0",
              activeCol === c ? "sel-on" : "text-ink-faint hover:text-ink",
            )}
            title={`Filter findings to ${c}`}
          >
            {c}
          </button>
        ))}
        {rows.map((r) => (
          <React.Fragment key={r}>
            <span className="self-center font-mono text-2xs uppercase tracking-wider text-ink-muted">{r}</span>
            {cols.map((c) => {
              const n = count(r, c);
              const t = n / max;
              return (
                <button
                  key={c}
                  type="button"
                  onClick={() => onCol?.(c)}
                  title={`${r} × ${c}: ${n}`}
                  className={cn(
                    "grid h-11 place-items-center rounded-md border font-mono text-xs tnum sm:h-9",
                    "transition-transform duration-press ease-out-strong active:scale-[0.97]",
                    activeCol && activeCol !== c ? "opacity-40" : "",
                    n ? "border-accent/25 text-ink" : "border-line text-ink-faint",
                  )}
                  style={{ background: n ? `hsl(var(--accent) / ${0.08 + t * 0.55})` : "transparent" }}
                >
                  {n || "·"}
                </button>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ histogram */

export function Histogram({
  values,
  bins = 10,
  label,
}: {
  values: number[];
  bins?: number;
  label: string;
}) {
  const reduced = usePrefersReducedMotion();
  const data = Array.from({ length: bins }, (_, i) => ({
    bin: `${(i / bins).toFixed(1)}`,
    lo: i / bins,
    hi: (i + 1) / bins,
    n: 0,
  }));
  for (const v of values) {
    const i = Math.min(bins - 1, Math.max(0, Math.floor(v * bins)));
    data[i].n += 1;
  }
  return (
    <div className="h-[170px]" role="img" aria-label={`${label} histogram`}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 6, right: 6, bottom: 0, left: -18 }} barCategoryGap={3}>
          <CartesianGrid vertical={false} stroke="hsl(var(--line))" strokeDasharray="2 4" />
          <XAxis dataKey="bin" tick={{ fill: "hsl(var(--ink-faint))", fontSize: 10 }} stroke="hsl(var(--line-strong))" />
          <YAxis allowDecimals={false} tick={{ fill: "hsl(var(--ink-faint))", fontSize: 10 }} stroke="hsl(var(--line-strong))" />
          <Tooltip
            cursor={{ fill: "hsl(var(--accent) / 0.06)" }}
            content={({ active, payload }) => {
              const p = payload?.[0]?.payload as { lo: number; hi: number; n: number } | undefined;
              return active && p ? (
                <Tip>
                  <p className="font-mono text-ink tnum">
                    {label} {p.lo.toFixed(1)}–{p.hi.toFixed(1)}
                  </p>
                  <p className="font-mono text-ink-muted tnum">{p.n} finding{p.n === 1 ? "" : "s"}</p>
                </Tip>
              ) : null;
            }}
          />
          <Bar dataKey="n" fill="hsl(var(--accent) / 0.7)" radius={[4, 4, 0, 0]} isAnimationActive={!reduced} animationDuration={400} animationEasing="ease-out" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ------------------------------------------------------------------ contributor scatter */

export function ContributorScatter({
  rows,
  cohortRate,
}: {
  rows: {
    label: string;
    n: number;
    posterior: number;
    flagged: number;
    excludes: boolean;
  }[];
  cohortRate?: number;
}) {
  const reduced = usePrefersReducedMotion();
  const normal = rows.filter((r) => !r.excludes);
  const outliers = rows.filter((r) => r.excludes);
  return (
    <div className="h-[260px]" role="img" aria-label="Contributors: sample count against posterior flag rate">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 10, right: 16, bottom: 18, left: -8 }}>
          <CartesianGrid stroke="hsl(var(--line))" strokeDasharray="2 4" />
          <XAxis
            type="number"
            dataKey="n"
            name="samples"
            scale="log"
            domain={["auto", "auto"]}
            tick={{ fill: "hsl(var(--ink-faint))", fontSize: 10 }}
            stroke="hsl(var(--line-strong))"
            label={{ value: "samples (log)", position: "insideBottom", offset: -8, fill: "hsl(var(--ink-faint))", fontSize: 10 }}
          />
          <YAxis
            type="number"
            dataKey="posterior"
            name="posterior"
            tick={{ fill: "hsl(var(--ink-faint))", fontSize: 10 }}
            stroke="hsl(var(--line-strong))"
            tickFormatter={(v: number) => v.toFixed(2)}
          />
          <ZAxis type="number" dataKey="flagged" range={[50, 520]} />
          {cohortRate !== undefined ? (
            <ReferenceLine
              y={cohortRate}
              stroke="hsl(var(--review))"
              strokeDasharray="5 4"
              label={{ value: "cohort rate", position: "insideTopRight", fill: "hsl(var(--review))", fontSize: 10 }}
            />
          ) : null}
          <Tooltip
            cursor={{ stroke: "hsl(var(--line-strong))" }}
            content={({ active, payload }) => {
              const p = payload?.[0]?.payload as (typeof rows)[number] | undefined;
              return active && p ? (
                <Tip>
                  <p className="font-mono text-ink">{p.label}</p>
                  <p className="font-mono text-ink-muted tnum">
                    n {p.n} · flagged {p.flagged}
                  </p>
                  <p className="font-mono text-ink-muted tnum">posterior {fixed(p.posterior, 4)}</p>
                  {p.excludes ? <p className="mt-1 text-quarantine">interval excludes the cohort rate</p> : null}
                </Tip>
              ) : null;
            }}
          />
          <Scatter data={normal} fill="hsl(var(--accent) / 0.55)" stroke="hsl(var(--accent))" strokeWidth={1.5} isAnimationActive={!reduced} animationDuration={400} animationEasing="ease-out" />
          <Scatter data={outliers} fill="hsl(var(--quarantine) / 0.55)" stroke="hsl(var(--quarantine))" strokeWidth={1.5} isAnimationActive={!reduced} animationDuration={400} animationEasing="ease-out" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ------------------------------------------------------------------ forest plot */

/**
 * Each contributor's posterior with its 95% interval: Recharts' ErrorBar, asymmetric, drawn
 * horizontally. A row whose interval excludes its cohort rate is in the quarantine hue —
 * the report decided that (`excludes_cohort_rate`); the chart only draws it. The numbers
 * are printed beside the plot in the table below it, so the picture is never the only
 * carrier.
 */
export function ForestPlot({
  rows,
  cohortRate,
}: {
  rows: {
    label: string;
    mean: number;
    lo: number;
    hi: number;
    cohort?: number;
    excludes: boolean;
    n: number;
  }[];
  cohortRate?: number;
}) {
  const reduced = usePrefersReducedMotion();
  const data = rows.map((r) => ({
    ...r,
    err: [Math.max(0, r.mean - r.lo), Math.max(0, r.hi - r.mean)] as [number, number],
  }));
  const normal = data.filter((d) => !d.excludes);
  const outliers = data.filter((d) => d.excludes);
  const height = Math.max(140, rows.length * 30 + 44);
  const tip = ({ active, payload }: { active?: boolean; payload?: readonly { payload?: unknown }[] }) => {
    const p = payload?.[0]?.payload as (typeof data)[number] | undefined;
    return active && p ? (
      <Tip>
        <p className="font-mono text-ink">{p.label}</p>
        <p className="font-mono text-ink-muted tnum">posterior {fixed(p.mean, 4)}</p>
        <p className="font-mono text-ink-muted tnum">
          95% {fixed(p.lo, 4)}–{fixed(p.hi, 4)}
        </p>
        {p.cohort !== undefined ? (
          <p className="font-mono text-ink-muted tnum">vs cohort {fixed(p.cohort, 4)} (leave-one-out)</p>
        ) : null}
        <p className="font-mono text-ink-faint tnum">n {p.n}</p>
        {p.excludes ? <p className="mt-1 text-quarantine">interval excludes the cohort rate</p> : null}
      </Tip>
    ) : null;
  };
  return (
    <div style={{ height }} role="img" aria-label="Forest plot: each group's posterior and 95% interval">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
          <CartesianGrid horizontal={false} stroke="hsl(var(--line))" strokeDasharray="2 4" />
          <XAxis
            type="number"
            dataKey="mean"
            domain={[0, "auto"]}
            tick={{ fill: "hsl(var(--ink-faint))", fontSize: 10 }}
            stroke="hsl(var(--line-strong))"
            tickFormatter={(v: number) => v.toFixed(2)}
            label={{ value: "posterior flag rate", position: "insideBottom", offset: -8, fill: "hsl(var(--ink-faint))", fontSize: 10 }}
          />
          <YAxis
            type="category"
            dataKey="label"
            allowDuplicatedCategory={false}
            width={92}
            tick={{ fill: "hsl(var(--ink-muted))", fontSize: 11, fontFamily: "var(--font-mono)" }}
            stroke="hsl(var(--line-strong))"
          />
          {cohortRate !== undefined ? (
            <ReferenceLine
              x={cohortRate}
              stroke="hsl(var(--review))"
              strokeDasharray="5 4"
              label={{ value: "cohort", position: "top", fill: "hsl(var(--review))", fontSize: 10 }}
            />
          ) : null}
          <Tooltip cursor={{ stroke: "hsl(var(--line-strong))" }} content={tip} />
          <Scatter data={normal} fill="hsl(var(--accent))" isAnimationActive={!reduced} animationDuration={400} animationEasing="ease-out">
            <ErrorBar dataKey="err" direction="x" width={6} strokeWidth={2} stroke="hsl(var(--accent) / 0.7)" />
          </Scatter>
          <Scatter data={outliers} fill="hsl(var(--quarantine))" isAnimationActive={!reduced} animationDuration={400} animationEasing="ease-out">
            <ErrorBar dataKey="err" direction="x" width={6} strokeWidth={2} stroke="hsl(var(--quarantine) / 0.8)" />
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ------------------------------------------------------------------ audit charts */

export function ActorBars({ data }: { data: { actor: string; n: number }[] }) {
  const reduced = usePrefersReducedMotion();
  const height = Math.max(110, data.length * 34 + 24);
  return (
    <div style={{ height }} role="img" aria-label="Decisions recorded per person">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 12, bottom: 4, left: 0 }} barCategoryGap={8}>
          <XAxis type="number" allowDecimals={false} tick={{ fill: "hsl(var(--ink-faint))", fontSize: 10 }} stroke="hsl(var(--line-strong))" />
          <YAxis
            type="category"
            dataKey="actor"
            width={84}
            tick={{ fill: "hsl(var(--ink-muted))", fontSize: 11, fontFamily: "var(--font-mono)" }}
            stroke="hsl(var(--line-strong))"
          />
          <Tooltip
            cursor={{ fill: "hsl(var(--accent) / 0.06)" }}
            content={({ active, payload }) => {
              const p = payload?.[0]?.payload as { actor: string; n: number } | undefined;
              return active && p ? (
                <Tip>
                  <p className="font-mono text-ink">{p.actor}</p>
                  <p className="font-mono text-ink-muted tnum">{p.n} decision{p.n === 1 ? "" : "s"}</p>
                </Tip>
              ) : null;
            }}
          />
          <Bar dataKey="n" fill="hsl(var(--accent) / 0.75)" radius={[0, 4, 4, 0]} isAnimationActive={!reduced} animationDuration={400} animationEasing="ease-out" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Cumulative records over ledger seq. The x axis is SEQ, never time: the clock is untrusted. */
export function SeqActivity({
  rows,
}: {
  rows: { seq: number; kind: string }[];
}) {
  const reduced = usePrefersReducedMotion();
  let decisions = 0;
  let scans = 0;
  const data = [...rows]
    .sort((a, b) => a.seq - b.seq)
    .map((r) => {
      if (r.kind === "analyst_event") decisions += 1;
      else if (r.kind === "scan_record") scans += 1;
      return { seq: r.seq, decisions, scans };
    });
  return (
    <div className="h-[200px]" role="img" aria-label="Records over ledger sequence">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 10, bottom: 14, left: -18 }}>
          <defs>
            <linearGradient id="cva-area-a" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="hsl(var(--accent))" stopOpacity={0.45} />
              <stop offset="100%" stopColor="hsl(var(--accent))" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid vertical={false} stroke="hsl(var(--line))" strokeDasharray="2 4" />
          <XAxis
            dataKey="seq"
            type="number"
            domain={["dataMin", "dataMax"]}
            allowDecimals={false}
            tick={{ fill: "hsl(var(--ink-faint))", fontSize: 10 }}
            stroke="hsl(var(--line-strong))"
            label={{ value: "ledger seq", position: "insideBottom", offset: -8, fill: "hsl(var(--ink-faint))", fontSize: 10 }}
          />
          <YAxis allowDecimals={false} tick={{ fill: "hsl(var(--ink-faint))", fontSize: 10 }} stroke="hsl(var(--line-strong))" />
          <Tooltip
            content={({ active, payload }) => {
              const p = payload?.[0]?.payload as { seq: number; decisions: number; scans: number } | undefined;
              return active && p ? (
                <Tip>
                  <p className="font-mono text-ink tnum">seq {p.seq}</p>
                  <p className="font-mono text-ink-muted tnum">{p.decisions} analyst decisions so far</p>
                  <p className="font-mono text-ink-muted tnum">{p.scans} sealed scans so far</p>
                </Tip>
              ) : null;
            }}
          />
          <Area
            type="stepAfter"
            dataKey="decisions"
            stroke="hsl(var(--accent))"
            strokeWidth={2}
            fill="url(#cva-area-a)"
            isAnimationActive={!reduced} animationDuration={400} animationEasing="ease-out"
          />
          <Area
            type="stepAfter"
            dataKey="scans"
            stroke="hsl(var(--ink-muted))"
            strokeWidth={1.5}
            strokeDasharray="4 3"
            fill="transparent"
            isAnimationActive={!reduced} animationDuration={400} animationEasing="ease-out"
          />
        </AreaChart>
      </ResponsiveContainer>
      <div className="mt-1 flex flex-wrap gap-4 text-2xs text-ink-muted">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-0.5 w-4 bg-accent" aria-hidden /> analyst decisions
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-0.5 w-4 border-t border-dashed border-ink-muted" aria-hidden /> sealed scans
        </span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ KPI tile */

export function Kpi({
  label,
  value,
  note,
  icon: Icon,
  tone,
  index = 0,
}: {
  label: string;
  value: number;
  note?: string;
  icon: LucideIcon;
  tone?: Tone;
  index?: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, transform: "translateY(8px)" }}
      animate={{ opacity: 1, transform: "translateY(0px)" }}
      transition={{ duration: 0.25, delay: index * 0.05, ease: [0.23, 1, 0.32, 1] }}
      className="glass relative overflow-hidden rounded-xl border border-line/80 p-4"
    >
      <div className="relative flex items-start justify-between gap-3">
        <Eyebrow>{label}</Eyebrow>
        {/* A tile with no status keeps a neutral icon: hue here would claim a meaning. */}
        <Icon className={cn("h-4 w-4", tone ? TONE_TEXT[tone] : "text-ink-faint")} aria-hidden />
      </div>
      <div
        className={cn(
          "relative mt-2 font-mono text-3xl font-medium leading-none tracking-tight tnum",
          tone ? TONE_TEXT[tone] : "text-ink-strong",
        )}
      >
        <CountUp value={value} />
      </div>
      {note ? <p className="relative mt-1.5 text-xs text-ink-faint">{note}</p> : null}
    </motion.div>
  );
}
