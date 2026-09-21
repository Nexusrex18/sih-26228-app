"use client";

import { Command } from "cmdk";
import {
  Activity,
  FileSearch,
  Gauge,
  Layers,
  Microscope,
  ScrollText,
  ShieldCheck,
  Users,
} from "lucide-react";
import { useRouter } from "next/navigation";
import * as React from "react";
import { api } from "@/lib/api";
import type { ScanSummary } from "@/lib/types";

/**
 * ⌘K / Ctrl+K: jump to any scan or page.
 *
 * No open/close animation, deliberately: a palette summoned by a keyboard shortcut is used
 * dozens of times a day, and motion on it would only put time between the analyst and the
 * page they asked for.
 */
export function CommandPalette({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const router = useRouter();
  const [scans, setScans] = React.useState<ScanSummary[]>([]);

  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        onOpenChange(!open);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onOpenChange]);

  React.useEffect(() => {
    if (!open) return;
    api.scans().then((r) => setScans(r.scans.filter((s) => s.readable)), () => undefined);
  }, [open]);

  const go = (href: string) => {
    onOpenChange(false);
    router.push(href);
  };

  const scanPages = [
    { path: "/scan", label: "Overview", icon: Gauge },
    { path: "/findings", label: "Findings", icon: Microscope },
    { path: "/contributors", label: "Contributor risk", icon: Users },
    { path: "/provenance", label: "Provenance", icon: ShieldCheck },
    { path: "/coverage", label: "Coverage", icon: Layers },
  ];

  return (
    <Command.Dialog
      open={open}
      onOpenChange={onOpenChange}
      label="Jump to a scan or page"
      overlayClassName="fixed inset-0 z-[60] bg-surface-deep/70 backdrop-blur-sm"
      contentClassName="fixed left-1/2 top-[14vh] z-[61] w-[min(640px,calc(100vw-2rem))] -translate-x-1/2 overflow-hidden rounded-xl border border-line-strong glass float-shadow"
    >
      <Command.Input
        placeholder="Jump to a scan, a page, or a scan's findings…"
        className="h-12 w-full border-b border-line bg-transparent px-4 font-sans text-sm text-ink outline-none placeholder:text-ink-faint"
      />
      <Command.List className="max-h-[60vh] overflow-y-auto p-2 scroll-slim">
        <Command.Empty className="px-3 py-6 text-center text-sm text-ink-muted">
          Nothing matches.
        </Command.Empty>

        <Command.Group heading="Pages" className="cmdk-group">
          <Item onSelect={() => go("/")} icon={FileSearch} label="Scans" hint="triage queue" />
          <Item onSelect={() => go("/audit")} icon={ScrollText} label="Audit trail" hint="every decision, by seq" />
          <Item onSelect={() => go("/verification")} icon={Activity} label="Verification" hint="the ledger check" />
        </Command.Group>

        {scans.map((s) => (
          <Command.Group key={s.scan_id} heading={`${s.scan_id} · ${s.profile_name}`} className="cmdk-group">
            {scanPages.map((p) => (
              <Item
                key={p.path}
                value={`${s.scan_id} ${p.label}`}
                onSelect={() => go(`${p.path}?id=${s.scan_id}`)}
                icon={p.icon}
                label={p.label}
                hint={s.scan_id}
              />
            ))}
          </Command.Group>
        ))}
      </Command.List>
      <div className="flex items-center gap-3 border-t border-line px-4 py-2 font-mono text-2xs text-ink-faint">
        <span>↑↓ move</span>
        <span>↵ open</span>
        <span>esc close</span>
      </div>
    </Command.Dialog>
  );
}

function Item({
  onSelect,
  icon: Icon,
  label,
  hint,
  value,
}: {
  onSelect: () => void;
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  hint?: string;
  value?: string;
}) {
  return (
    <Command.Item
      value={value ?? label}
      onSelect={onSelect}
      className="flex cursor-pointer items-center gap-3 rounded-md px-3 py-2.5 text-sm text-ink-muted data-[selected=true]:bg-surface-raised data-[selected=true]:text-ink-strong"
    >
      <Icon className="h-4 w-4 shrink-0 text-ink-faint" />
      <span className="flex-1">{label}</span>
      {hint ? <span className="font-mono text-2xs text-ink-faint">{hint}</span> : null}
    </Command.Item>
  );
}
