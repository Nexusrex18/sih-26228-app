"use client";

import * as React from "react";
import type { Health } from "@/lib/types";
import { cn } from "@/lib/ui";

/**
 * THE LEDGER PULSE — the one bold element in the design.
 *
 * A live signal trace, the way an instrument shows a channel. Its state is the ledger's:
 *
 *   verified     a steady heartbeat in the accept hue
 *   unverified   the same beat in amber: alive, not yet checked
 *   failed       an irregular, spiking trace in the quarantine hue
 *   unavailable  a FLAT LINE — no signal is the honest picture of an unreachable ledger
 *
 * The label beside it is the ledger's own clock: the record count, not a time. Motion is a
 * CSS animation (it keeps running smoothly while the page is busy loading data) and stops
 * under prefers-reduced-motion via the global rule.
 */
export function LedgerPulse({ health }: { health: Health | null }) {
  const kind = health?.kind ?? "unavailable";
  const records = health?.verify?.records_checked ?? 0;
  const tone = {
    verified: "text-accept",
    unverified: "text-review",
    failed: "text-quarantine",
    unavailable: "text-absent",
  }[kind];

  // One period of the trace, 200 units wide. Drawn twice so the scroll loops seamlessly.
  const beat = (x: number) => {
    if (kind === "unavailable") return `L${x + 200} 20`;
    if (kind === "failed") {
      return `L${x + 40} 20 L${x + 48} 6 L${x + 54} 34 L${x + 62} 12 L${x + 70} 30 L${x + 78} 20 L${x + 120} 20 L${x + 126} 2 L${x + 134} 36 L${x + 142} 20 L${x + 200} 20`;
    }
    return `L${x + 70} 20 L${x + 78} 16 L${x + 86} 20 L${x + 94} 20 L${x + 100} 4 L${x + 108} 34 L${x + 114} 20 L${x + 124} 20 L${x + 132} 15 L${x + 142} 20 L${x + 200} 20`;
  };
  const d = `M0 20 ${beat(0)} ${beat(200)}`;

  return (
    <div
      className={cn("flex items-center gap-3", tone)}
      role="img"
      aria-label={
        kind === "unavailable"
          ? "Ledger pulse: no signal, the ledger is unreachable"
          : `Ledger pulse: ${kind}, ${records} records`
      }
    >
      <div className="relative h-9 w-40 overflow-hidden sm:w-56">
        <svg
          viewBox="0 0 400 40"
          preserveAspectRatio="none"
          className={cn("absolute inset-y-0 left-0 h-full w-[200%]", kind !== "unavailable" && "ledger-trace")}
          aria-hidden
        >
          <path
            d={d}
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinejoin="round"
            className="ledger-trace-glow"
            vectorEffect="non-scaling-stroke"
          />
        </svg>
        {/* Fade both ends so the trace appears out of and into the panel, like a scope. */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              "linear-gradient(90deg, hsl(var(--surface-panel)) 0%, transparent 18%, transparent 82%, hsl(var(--surface-panel)) 100%)",
          }}
        />
      </div>
      <div className="hidden font-mono text-2xs leading-tight sm:block">
        <div className="text-ink-faint">ledger</div>
        <div className="tnum">
          {kind === "unavailable" ? "no signal" : `${records} rec`}
        </div>
      </div>
    </div>
  );
}
