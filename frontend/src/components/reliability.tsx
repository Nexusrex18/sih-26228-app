"use client";

import * as React from "react";
import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { Panel, usePrefersReducedMotion } from "@/components/ui/primitives";
import type { Calibration } from "@/lib/types";
import { fixed } from "@/lib/ui";

/**
 * The reliability diagram.
 *
 * Predicted against empirical, one point per bin, with the identity diagonal as the
 * reference: a point below the line is a detector claiming more confidence than it earns.
 * Point area carries `n`, because a bin holding four samples and a bin holding four hundred
 * are not the same evidence and a diagram that draws them the same size says they are.
 *
 * One series, so no legend box and no categorical palette: the accent is structural here,
 * the diagonal is recessive ink, and the exact numbers are printed beneath the plot. The
 * picture is an aid, never the only carrier — the table under it is the primary artefact.
 */
export function ReliabilityDiagram({ calibration }: { calibration: Calibration }) {
  const reduced = usePrefersReducedMotion();
  const bins = calibration.reliability_bins ?? [];
  if (!bins.length) return null;

  const data = bins.map((b) => ({ x: b.p_mean, y: b.empirical, n: b.n }));

  return (
    <div className="space-y-3">
      <div
        className="h-[280px] w-full"
        role="img"
        aria-label={`Reliability diagram. ${bins
          .map(
            (b) =>
              `predicted ${fixed(b.p_mean, 2)}, empirical ${fixed(b.empirical, 2)}, n ${b.n}`,
          )
          .join("; ")}. The diagonal is perfect calibration.`}
      >
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 16, bottom: 28, left: 8 }}>
            <CartesianGrid stroke="hsl(var(--line))" strokeDasharray="2 4" />
            <XAxis
              type="number"
              dataKey="x"
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              tick={{ fill: "hsl(var(--ink-faint))", fontSize: 11 }}
              stroke="hsl(var(--line-strong))"
              label={{
                value: "predicted",
                position: "insideBottom",
                offset: -14,
                fill: "hsl(var(--ink-faint))",
                fontSize: 11,
              }}
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              tick={{ fill: "hsl(var(--ink-faint))", fontSize: 11 }}
              stroke="hsl(var(--line-strong))"
              width={44}
              label={{
                value: "empirical",
                angle: -90,
                position: "insideLeft",
                fill: "hsl(var(--ink-faint))",
                fontSize: 11,
              }}
            />
            <ZAxis type="number" dataKey="n" range={[60, 420]} name="samples" />
            {/* Perfect calibration. Labelled, because an unlabelled diagonal is decoration. */}
            <ReferenceLine
              segment={[
                { x: 0, y: 0 },
                { x: 1, y: 1 },
              ]}
              stroke="hsl(var(--line-strong))"
              strokeDasharray="4 4"
              ifOverflow="extendDomain"
            />
            <Tooltip
              cursor={{ stroke: "hsl(var(--line-strong))" }}
              content={<BinTooltip />}
            />
            <Scatter
              data={data}
              fill="hsl(var(--accent) / 0.55)"
              stroke="hsl(var(--accent))"
              strokeWidth={2}
              isAnimationActive={!reduced}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <p className="text-2xs text-ink-faint">
        The dashed diagonal is perfect calibration. A point below it is a confidence the
        detector did not earn; point area is the number of samples in the bin.
      </p>

      {/* The table view. Every value in the plot is readable without the plot. */}
      <Panel className="overflow-hidden">
        <div className="overflow-x-auto scroll-slim">
          <table className="w-full text-sm">
            <caption className="sr-only">Reliability bins, exact values</caption>
            <thead>
              <tr className="border-b border-line-strong">
                {["bin", "predicted", "empirical", "n"].map((h) => (
                  <th
                    key={h}
                    className="px-3 py-2 text-left font-mono text-2xs font-semibold uppercase tracking-wider text-ink-faint"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {bins.map((b, i) => (
                <tr key={i} className="border-b border-line last:border-0">
                  <td className="px-3 py-2 font-mono text-2xs text-ink-faint tnum">
                    {i + 1}
                  </td>
                  <td className="px-3 py-2 font-mono tnum">{fixed(b.p_mean, 2)}</td>
                  <td className="px-3 py-2 font-mono tnum">{fixed(b.empirical, 2)}</td>
                  <td className="px-3 py-2 font-mono tnum">{b.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}

function BinTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload: { x: number; y: number; n: number } }[];
}) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="rounded border border-line-strong bg-surface-panel px-3 py-2 float-shadow">
      <p className="font-mono text-2xs text-ink-faint">n {p.n}</p>
      <p className="font-mono text-xs tnum">predicted {fixed(p.x, 2)}</p>
      <p className="font-mono text-xs tnum">empirical {fixed(p.y, 2)}</p>
    </div>
  );
}
