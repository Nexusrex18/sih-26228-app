"use client";

import { Check, Download, Loader2, RefreshCw, ShieldAlert } from "lucide-react";
import * as React from "react";
import { toast } from "sonner";
import { Shell, useApp } from "@/components/shell";
import {
  Banner,
  Button,
  Chip,
  Empty,
  Panel,
  Section,
  SectionHead,
  Stat,
} from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { hostTime } from "@/lib/ui";

export default function VerificationPage() {
  const { session, health, refreshHealth } = useApp();
  const [busy, setBusy] = React.useState(false);
  const [exporting, setExporting] = React.useState(false);
  const [exported, setExported] = React.useState<string | null>(null);

  const state = health?.verify ?? null;
  const canExport = session?.role === "approver" || session?.role === "admin";

  return (
    <Shell
      title="Ledger verification"
      subtitle="Run by shelling out to cva-seal verify"
      actions={
        <Button
          size="sm"
          icon={busy ? Loader2 : RefreshCw}
          disabled={busy}
          className={busy ? "[&_svg]:animate-spin" : undefined}
          onClick={async () => {
            setBusy(true);
            try {
              await api.reverify();
              await refreshHealth();
            } catch (e) {
              toast.error((e as Error).message);
            } finally {
              setBusy(false);
            }
          }}
        >
          Verify now
        </Button>
      }
    >
      <Section>
        {!state || state.state === "UNAVAILABLE" ? (
          <Empty title="Verification has not run" hatched>
            <p>{state?.detail || "No verification has been attempted."}</p>
            <p>
              The workflow stays read-only until it does. A fold we cannot verify is not a
              record anyone may act on.
            </p>
          </Empty>
        ) : (
          <Panel className="p-4">
            <div className="mb-4 flex flex-wrap items-center gap-3">
              {state.state === "OK" ? (
                <Chip tone="accept" icon={Check}>
                  verified
                </Chip>
              ) : (
                <Chip tone="quarantine" icon={ShieldAlert}>
                  FAILED
                </Chip>
              )}
              <span className="text-2xs text-ink-faint">
                checked {hostTime(state.checked_at)} (host clock, untrusted)
              </span>
            </div>

            <p className="mb-5 max-w-[75ch] text-sm text-ink-muted">{state.summary}</p>

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <Stat label="Records" value={state.records_checked} countUp />
              <Stat
                label="Anchors"
                value={state.anchors_verified}
                countUp
                note={`of ${state.anchors_in_chain} in chain`}
              />
              {/* The unwitnessed window is the number this page exists to show: those
                  records are still trusting the key holder alone. It is never a zero by
                  omission. */}
              <Stat
                label="Unwitnessed"
                value={state.unwitnessed_records}
                countUp
                tone="absent"
                note="records after the last anchor"
              />
              <Stat
                label="Declared gaps"
                value={state.declared_gaps}
                countUp
                tone={state.declared_gaps > 0 ? "absent" : "neutral"}
                note="intervals the field unit admitted it could not seal"
              />
            </div>

            <dl className="mt-6 grid gap-x-6 gap-y-2 text-sm sm:grid-cols-[8rem_1fr]">
              <Dt>Durability</Dt>
              <dd className="font-mono text-xs">
                {state.durability || "—"}
                {state.loss_window ? ` · loss window ${state.loss_window}` : ""}
              </dd>
              <Dt>Command</Dt>
              <dd>
                <code className="block break-all rounded-sm border border-line bg-surface-deep px-2 py-1 font-mono text-xs">
                  {state.command}
                </code>
                <span className="mt-1 block text-2xs text-ink-faint">
                  Run it yourself; this page parses exactly that output.
                </span>
              </dd>
            </dl>
          </Panel>
        )}
      </Section>

      {state && state.findings.length ? (
        <Section delay={0.04}>
          <SectionHead title="Findings" />
          <div className="space-y-2">
            {state.findings.map((f, i) => (
              <Panel key={i} className="p-4">
                <div className="mb-1.5 flex flex-wrap items-center gap-2">
                  <Chip tone="quarantine">{String(f.severity ?? "—")}</Chip>
                  <Chip tone="neutral">{String(f.attack_class ?? "—")}</Chip>
                  <span className="break-all font-mono text-2xs text-ink-faint">
                    {String(f.target_ref ?? "")}
                  </span>
                  {f.primary_check ? (
                    <span className="text-2xs text-ink-faint">
                      primary check:{" "}
                      <span className="font-mono">{String(f.primary_check)}</span>
                    </span>
                  ) : null}
                </div>
                <p className="break-words text-sm text-ink-muted">
                  {String(f.reason ?? "")}
                </p>
              </Panel>
            ))}
          </div>
        </Section>
      ) : null}

      {state && state.limitations.length ? (
        <Section delay={0.08}>
          <SectionHead title="What the verifier does not claim" />
          <Panel hatched className="space-y-3 p-4">
            {state.limitations.map((l, i) => (
              <p key={i} className="max-w-[80ch] text-sm text-ink-muted">
                {l}
              </p>
            ))}
          </Panel>
        </Section>
      ) : null}

      <Section delay={0.12}>
        <SectionHead title="Independent verification" />
        <Panel className="p-4">
          <p className="max-w-[80ch] text-sm text-ink-muted">
            Export the ledger and verify it on another machine with the trust root alone —
            without this dashboard, and without this database. That is the check a third
            party can run, and the procedure is in{" "}
            <code className="font-mono text-xs">docs/VERIFICATION-PROCEDURE.md</code>.
          </p>
          <div className="mt-4">
            {canExport ? (
              <Button
                icon={exporting ? Loader2 : Download}
                disabled={exporting}
                className={exporting ? "[&_svg]:animate-spin" : undefined}
                onClick={async () => {
                  setExporting(true);
                  try {
                    const r = await api.exportLedger();
                    setExported(r.path);
                    toast.success("Ledger exported");
                  } catch (e) {
                    toast.error((e as Error).message);
                  } finally {
                    setExporting(false);
                  }
                }}
              >
                Export the ledger
              </Button>
            ) : (
              <p className="text-2xs text-ink-faint">
                Exporting needs the approver or admin role.
              </p>
            )}
          </div>
          {exported ? (
            <div className="mt-4">
              <Banner tone="ok" title="Exported">
                <p className="break-all">
                  Written to <span className="font-mono text-xs">{exported}</span>. Verify it
                  on another machine with the trust root alone.
                </p>
              </Banner>
            </div>
          ) : null}
        </Panel>
      </Section>
    </Shell>
  );
}

function Dt({ children }: { children: React.ReactNode }) {
  return (
    <dt className="font-mono text-2xs uppercase tracking-wider text-ink-faint">
      {children}
    </dt>
  );
}
