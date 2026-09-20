"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/ui";

/**
 * The section tabs, in the single-file report's fixed read order (plan §7.2): verdict,
 * access assumptions, contributor risk, findings, provenance, shift, coverage,
 * reproduction. Both artefacts tell the same story in the same sequence, so an analyst who
 * has read one can navigate the other.
 */
const TABS = [
  { href: "/scan", label: "Overview" },
  { href: "/contributors", label: "Contributor risk" },
  { href: "/findings", label: "Findings", counted: true },
  { href: "/provenance", label: "Provenance" },
  { href: "/coverage", label: "Coverage" },
];

export function ScanNav({ scanId, n }: { scanId: string; n?: number }) {
  const pathname = usePathname();
  return (
    <nav aria-label="Report sections" className="mb-6 flex flex-wrap gap-2">
      {TABS.map((t) => {
        const active = pathname.startsWith(t.href);
        return (
          <Link
            key={t.href}
            href={`${t.href}?id=${scanId}`}
            aria-current={active ? "page" : undefined}
            className={cn(
              "inline-flex h-8 items-center gap-2 rounded-full border px-3.5 font-mono text-xs transition-colors",
              active
                ? "border-accent bg-accent/10 text-accent"
                : "border-line bg-surface-panel text-ink-muted hover:border-line-strong hover:text-ink",
            )}
          >
            {t.label}
            {t.counted && n !== undefined ? (
              <span className="text-ink-faint tnum">{n}</span>
            ) : null}
          </Link>
        );
      })}
      <span className="flex-1" />
      <Link
        href={`/audit?id=${scanId}`}
        className="inline-flex h-8 items-center rounded-full border border-line bg-surface-panel px-3.5 font-mono text-xs text-ink-muted transition-colors hover:border-line-strong hover:text-ink"
      >
        Audit trail
      </Link>
    </nav>
  );
}
