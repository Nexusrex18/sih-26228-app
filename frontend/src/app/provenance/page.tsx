"use client";

import { Loader2 } from "lucide-react";
import { useSearchParams } from "next/navigation";
import * as React from "react";
import { toast } from "sonner";
import { DecideSheet } from "@/components/decide-sheet";
import {
  AvailabilityChip,
  DispositionChip,
  ProvenanceSummary,
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
import type { Finding, ScanDetail } from "@/lib/types";

export default function ProvenancePage() {
  return (
    <React.Suspense fallback={null}>
      <Body />
    </React.Suspense>
  );
}

function Body() {
  const params = useSearchParams();
  const scanId = params.get("id") ?? "";
  const { session, health, refreshHealth } = useApp();
  const [d, setD] = React.useState<ScanDetail | null>(null);
  const [rows, setRows] = React.useState<Finding[] | null>(null);
  const [open, setOpen] = React.useState<Finding | null>(null);

  const load = React.useCallback(async () => {
    if (!session?.authenticated || !scanId) return;
    try {
      const [detail, found] = await Promise.all([
        api.scan(scanId),
        // `prov.*` is the whole module, and the server filters it — a 10k-finding scan
        // never ships its other 9,900 rows here.
        api.findings(scanId, { module: "prov", per_page: 200 }),
      ]);
      setD(detail);
      setRows(found.findings);
    } catch (e) {
      toast.error((e as Error).message);
    }
  }, [scanId, session?.authenticated]);

  React.useEffect(() => {
    void load();
  }, [load]);

  if (!d) {
    return (
      <Shell title="Provenance" subtitle={scanId}>
        <div className="flex items-center gap-3 py-16 text-ink-muted">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
          <span className="font-mono text-sm">loading</span>
        </div>
      </Shell>
    );
  }

  return (
    <Shell
      title="Provenance"
      subtitle={scanId}
      actions={<SealBadge seal={d.seal} />}
      wide
    >
      <ScanNav scanId={scanId} n={d.n_findings} />

      <Section>
        <p className="max-w-[75ch] text-sm text-ink-muted">
          Two ledgers, never conflated. The{" "}
          <strong className="text-ink">inference ledger</strong> below is the field artefact
          under audit. The <strong className="text-ink">audit ledger</strong>, in the banner
          above and in the audit trail, records what this scanner and its analysts did.
        </p>
      </Section>

      <Section delay={0.04}>
        <SectionHead title="Inference ledger — the artefact under audit" />
        <ProvenanceSummary summary={d.provenance_summary} />
      </Section>

      <Section delay={0.08}>
        <SectionHead
          title="Provenance findings"
          note="Deterministic checks. Confidence is 1.0 because a hash either matched or it did not."
        />
        {rows === null ? (
          <div className="flex items-center gap-3 py-8 text-ink-muted">
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
            <span className="font-mono text-sm">loading findings</span>
          </div>
        ) : rows.length ? (
          <div className="space-y-2">
            {rows.map((f) => (
              <ProvRow key={f.finding_id} f={f} onDecide={() => setOpen(f)} />
            ))}
          </div>
        ) : (
          <Empty title="No provenance finding">
            <p>
              Every <code className="font-mono text-xs">prov.*</code> check that ran found
              what it expected. Read that together with the summary above: how much was
              checked, how much of the chain was witnessed by an anchor, and how long the
              unwitnessed window is.
            </p>
          </Empty>
        )}
      </Section>

      <Section delay={0.12}>
        <SectionHead title="What this does not establish" />
        <Panel hatched className="space-y-3 p-4">
          <p className="max-w-[80ch] text-sm text-ink-muted">
            The seal proves a record was produced by the pipeline and has not been altered
            since. It cannot attest that the input image was genuine before it entered the
            pipeline.
          </p>
          <p className="max-w-[80ch] text-sm text-ink-muted">
            Tail truncation and split-view within an anchoring interval are not visible in
            band. Verification against an external anchor is what bounds them, and the
            unwitnessed-window count above is how much is currently outside that bound.
          </p>
          <p className="max-w-[80ch] text-sm text-ink-muted">
            A history rewritten by the holder of the signing key verifies. That residual is
            what an HSM, the anchoring ceremony and two-person custody address — not this
            dashboard.
          </p>
        </Panel>
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

function ProvRow({ f, onDecide }: { f: Finding; onDecide: () => void }) {
  const st = f.state;
  return (
    <Panel className="p-4">
      <p className="mb-2 break-words text-sm leading-relaxed">{f.reason}</p>
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-2xs text-ink-faint">
          {f.finding_id.slice(0, 8)}
        </span>
        <DispositionChip value={st?.effective ?? f.disposition} />
        {st?.pending ? <DispositionChip value="pending" label="pending" /> : null}
        <SeverityText value={f.severity} />
        <Chip tone="neutral">{f.attack_class}</Chip>
        <AvailabilityChip state={f.availability} exclusionReason={f.exclusion_reason} />
        <span className="break-all font-mono text-2xs text-ink-faint">
          {f.target_type} {f.target_ref}
        </span>
        <span className="flex-1" />
        <Button size="sm" onClick={onDecide}>
          Open and decide
        </Button>
      </div>
    </Panel>
  );
}
