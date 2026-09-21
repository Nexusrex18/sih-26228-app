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
              "inline-flex h-11 items-center gap-2 rounded-full border px-4 font-mono text-xs transition-colors sm:h-8 sm:px-3.5",
              active
                ? "sel-on"
                : "sel-off",
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
        className="inline-flex h-11 items-center rounded-full border border-line bg-surface-panel px-4 font-mono text-xs text-ink-muted transition-colors hover:border-line-strong hover:text-ink sm:h-8 sm:px-3.5"
      >
        Audit trail
      </Link>
    </nav>
  );
}
