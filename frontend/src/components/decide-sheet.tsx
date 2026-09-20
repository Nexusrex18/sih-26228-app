"use client";

import { AlertTriangle, Check, Loader2, ShieldAlert } from "lucide-react";
import * as React from "react";
import { toast } from "sonner";
import {
  DispositionChip,
  EffectiveState,
  HostTime,
  SeqItem,
  SeqRail,
  SeverityText,
} from "@/components/domain";
import { Sheet } from "@/components/ui/sheet";
import { Button, Chip, Empty, Panel } from "@/components/ui/primitives";
import { ApiError, api } from "@/lib/api";
import type { Finding, LedgerEventRow } from "@/lib/types";
import { useApp } from "@/components/shell";
import { cn, fixed, justificationCheck, raises } from "@/lib/ui";

/**
 * The decide flow.
 *
 * Nothing here mutates anything locally. The six steps are the server's (plan §5.3) and
 * this component only asks: on a refusal it shows the reason and re-renders the state the
 * analyst already had, so the screen never shows something the ledger does not contain.
 */
export function DecideSheet({
  scanId,
  finding,
  onClose,
  writable,
  healthText,
  onRecorded,
}: {
  scanId: string;
  finding: Finding | null;
  onClose: () => void;
  writable: boolean;
  healthText: string;
  onRecorded: () => Promise<void> | void;
}) {
  const { session } = useApp();
  const policy = session?.policy;
  const can = session?.can ?? [];

  const [detail, setDetail] = React.useState<{
    finding: Finding;
    is_prov: boolean;
    reason_codes: string[];
    request_id: string;
    events: LedgerEventRow[];
    events_available: boolean;
  } | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [refusal, setRefusal] = React.useState<{ title: string; detail: string } | null>(
    null,
  );

  const [newDisposition, setNewDisposition] = React.useState("review");
  const [reasonCode, setReasonCode] = React.useState("");
  const [justification, setJustification] = React.useState("");

  React.useEffect(() => {
    if (!finding) {
      setDetail(null);
      return;
    }
    setRefusal(null);
    setJustification("");
    setReasonCode("");
    api.finding(scanId, finding.finding_id).then(setDetail, (e: Error) =>
      toast.error(e.message),
    );
  }, [finding, scanId]);

  if (!finding) return null;

  const f = detail?.finding ?? finding;
  const st = f.state;
  const effective = st?.effective ?? f.disposition;
  const jc = justificationCheck(
    justification,
    reasonCode === "other_with_justification"
      ? (policy?.min_justification_chars_other ?? 80)
      : (policy?.min_justification_chars ?? 30),
    policy?.min_justification_words ?? 5,
  );
  const lowering = !raises(effective, newDisposition as never) && newDisposition !== effective;
  const canOverride = can.includes("override");

  async function send(body: Record<string, unknown>) {
    setBusy(true);
    setRefusal(null);
    try {
      const r = await api.act(scanId, f.finding_id, {
        request_id: detail?.request_id,
        ...body,
      });
      toast.success(r.message);
      setJustification("");
      setDetail((d) => (d ? { ...d, finding: r.finding, events: r.events } : d));
      await onRecorded();
    } catch (e) {
      if (e instanceof ApiError) {
        setRefusal({ title: e.title, detail: e.payload.detail ?? e.message });
      } else {
        setRefusal({ title: "That action failed", detail: (e as Error).message });
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <Sheet
      open={Boolean(finding)}
      onClose={onClose}
      title={`Finding ${f.finding_id.slice(0, 8)}`}
      description={`${f.detector_id} · ${f.target_type} ${f.target_ref}`}
      footer={
        writable && canOverride ? (
          <Button
            tone="primary"
            className="w-full"
            disabled={busy || !jc.ok || !reasonCode || newDisposition === effective}
            icon={busy ? Loader2 : Check}
            onClick={() =>
              void send({
                action: "override",
                new_disposition: newDisposition,
                reason_code: reasonCode,
                justification,
              })
            }
          >
            {lowering ? "Request this change" : "Record this decision"}
          </Button>
        ) : null
      }
    >
      <div className="space-y-5 pb-2">
        {/* ---- what the tool said ---- */}
        <Panel className={cn("p-4", f.availability !== "OK" && "hatched")}>
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <DispositionChip value={effective} />
            {st?.pending ? (
              <DispositionChip value="pending" label="pending a second approver" />
            ) : null}
            <Chip tone="neutral">{f.nature}</Chip>
            <Chip tone="neutral">{f.attack_class}</Chip>
          </div>
          <p className="text-sm leading-relaxed">{f.reason}</p>

          {/* Severity and confidence are SEPARATE fields and never merged into one number.
              Severity is impact; confidence is belief. */}
          <div className="mt-4 flex flex-wrap gap-6">
            <Field label="Severity" note="impact if true">
              <SeverityText value={f.severity} />
            </Field>
            <Field label="Confidence" note="calibrated belief">
              <span className="font-mono text-sm tnum">{fixed(f.confidence, 2)}</span>
            </Field>
            <Field label="Score / threshold" note={f.detector_id}>
              <span className="font-mono text-sm tnum">
                {fixed(f.score_raw, 4)} / {fixed(f.threshold, 4)}
              </span>
            </Field>
          </div>

          <div className="mt-4 border-t border-line pt-3">
            <EffectiveState
              state={st}
              original={f.disposition}
              originalRule={f.disposition_rule}
            />
          </div>
        </Panel>

        {refusal ? (
          <div className="rounded border border-quarantine/40 border-l-[3px] border-l-quarantine bg-quarantine/[0.08] px-4 py-3">
            <strong className="block text-sm">{refusal.title}</strong>
            <p className="mt-1 text-sm text-ink-muted">{refusal.detail}</p>
            <p className="mt-2 text-xs text-ink-faint">
              Nothing was written to the ledger and nothing changed here.
            </p>
          </div>
        ) : null}

        {/* ---- pending approval ---- */}
        {st?.pending ? (
          <Panel className="p-4">
            <h3 className="mb-2 text-sm font-semibold">Awaiting a second person</h3>
            <p className="text-sm text-ink-muted">
              <span className="font-mono">{st.pending.requested_by}</span> requested{" "}
              <span className="font-mono">{st.pending.new_disposition}</span> at ledger seq{" "}
              <span className="font-mono">{st.pending.seq}</span>
              {st.pending.reason_code ? (
                <>
                  {" "}
                  under <span className="font-mono">{st.pending.reason_code}</span>
                </>
              ) : null}
              .
            </p>
            <blockquote className="mt-2 border-l-2 border-line-strong pl-3 text-sm text-ink-muted">
              {st.pending.justification}
            </blockquote>
            {can.includes("approve") &&
            session?.actor_id !== st.pending.requested_by ? (
              <Button
                tone="primary"
                size="sm"
                className="mt-3"
                disabled={busy}
                onClick={() =>
                  void send({ action: "approve", refs_seq: st.pending!.seq })
                }
              >
                Approve this change
              </Button>
            ) : session?.actor_id === st.pending.requested_by ? (
              <p className="mt-3 text-xs text-ink-faint">
                You requested this change, so you cannot approve it. Lowering a disposition
                is the dangerous direction in an acceptance workflow, and it takes a second
                person.
              </p>
            ) : (
              <p className="mt-3 text-xs text-ink-faint">
                This needs an account with the <span className="font-mono">approver</span>{" "}
                role.
              </p>
            )}
          </Panel>
        ) : null}

        {/* ---- the decide form ---- */}
        {!writable ? (
          <Empty title="The workflow is unavailable" hatched>
            <p>{healthText}</p>
            <p>
              No action is offered, because an action that cannot be recorded is an action
              that did not happen.
            </p>
          </Empty>
        ) : !canOverride ? (
          <Empty title="Your role is read-only here">
            <p>
              An account with the <span className="font-mono">{session?.role}</span> role
              can read everything and record nothing.
              {session?.role === "admin"
                ? " Administrators manage accounts and hold no workflow rights by default, so one person never controls both identity and decisions."
                : ""}
            </p>
          </Empty>
        ) : (
          <div className="space-y-4">
            <div>
              <label className="mb-1.5 block text-xs text-ink-muted" htmlFor="disp">
                New disposition
              </label>
              <div className="flex flex-wrap gap-2" id="disp">
                {(["quarantine", "review", "accept"] as const).map((d) => (
                  <button
                    key={d}
                    type="button"
                    disabled={d === effective}
                    onClick={() => setNewDisposition(d)}
                    aria-pressed={newDisposition === d}
                    className={cn(
                      "inline-flex h-11 items-center gap-2 rounded border px-4 font-mono text-xs transition-all",
                      "active:scale-[0.98] disabled:opacity-35",
                      newDisposition === d
                        ? "border-accent bg-accent/10 text-accent"
                        : "border-line bg-surface-panel text-ink-muted hover:border-line-strong",
                    )}
                  >
                    {d}
                    {d === effective ? " (current)" : ""}
                  </button>
                ))}
              </div>
              <p className="mt-1.5 text-xs text-ink-faint">
                {lowering
                  ? "Lowering waits for a second, different approver."
                  : "Raising takes effect immediately."}
              </p>
            </div>

            <div>
              <label className="mb-1.5 block text-xs text-ink-muted" htmlFor="rc">
                Reason
              </label>
              <select
                id="rc"
                value={reasonCode}
                onChange={(e) => setReasonCode(e.target.value)}
                className="h-11 w-full rounded border border-line-strong bg-surface-deep px-3 text-sm focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/25"
              >
                <option value="">choose a reason</option>
                {(detail?.reason_codes ?? []).map((c) => (
                  <option key={c} value={c}>
                    {c.replace(/_/g, " ")}
                  </option>
                ))}
              </select>
              <p className="mt-1.5 text-xs text-ink-faint">
                A closed set, so overrides can be counted and audited by cause instead of
                read one by one.
              </p>
            </div>

            {detail?.is_prov ? (
              <div className="rounded border border-quarantine/40 border-l-[3px] border-l-quarantine bg-quarantine/[0.08] px-4 py-3">
                <h3 className="flex items-center gap-2 text-sm font-semibold text-quarantine">
                  <ShieldAlert className="h-4 w-4" aria-hidden />
                  This is a deterministic failure
                </h3>
                <p className="mt-1 text-sm text-ink-muted">
                  The arithmetic did not hold. An override records that a human judged it
                  acceptable. <strong>It does not make the arithmetic hold.</strong>
                </p>
              </div>
            ) : null}

            {detail?.is_prov && reasonCode === "false_positive_confirmed" ? (
              <div className="rounded border border-review/40 border-l-[3px] border-l-review bg-review/[0.08] px-4 py-3">
                <h3 className="flex items-center gap-2 text-sm font-semibold text-review">
                  <AlertTriangle className="h-4 w-4" aria-hidden />
                  This will be flagged for audit
                </h3>
                <p className="mt-1 text-sm text-ink-muted">
                  Allowed, because some genuinely are. A deterministic check reporting a
                  mismatch it did not have is rare; satisfy yourself that this is one of
                  those cases.
                </p>
              </div>
            ) : null}

            <div>
              <label className="mb-1.5 block text-xs text-ink-muted" htmlFor="just">
                Justification
              </label>
              <textarea
                id="just"
                value={justification}
                onChange={(e) => setJustification(e.target.value)}
                rows={4}
                placeholder="What did you check, and what did you conclude? A later auditor reads this, not the score."
                className="w-full resize-y rounded border border-line-strong bg-surface-deep px-3 py-2 text-sm leading-relaxed focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/25"
              />
              <p className="mt-1.5 flex flex-wrap items-center gap-2 text-xs">
                <span
                  className={cn(
                    "font-mono tnum",
                    jc.ok ? "text-accept" : "text-review",
                  )}
                >
                  {jc.chars}/
                  {reasonCode === "other_with_justification"
                    ? policy?.min_justification_chars_other
                    : policy?.min_justification_chars}{" "}
                  characters · {jc.words}/{policy?.min_justification_words} words
                </span>
                <span className="text-ink-faint">
                  — an override without a recorded justification is how assurance systems
                  fail in practice.
                </span>
              </p>
            </div>

            <div className="flex flex-wrap gap-2 border-t border-line pt-4">
              <Button
                size="sm"
                disabled={busy}
                onClick={() => void send({ action: "acknowledge" })}
              >
                Acknowledge
              </Button>
            </div>
          </div>
        )}

        {/* ---- the record ---- */}
        <div>
          <h3 className="mb-3 text-sm font-semibold">Decision record</h3>
          {detail?.events.length ? (
            <SeqRail>
              {detail.events.map((e, i) => (
                <SeqItem
                  key={e.seq}
                  seq={e.seq}
                  index={i}
                  tone={
                    e.action === "override" || e.action === "approve"
                      ? "decision"
                      : "neutral"
                  }
                >
                  <Panel className="p-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <strong className="font-mono text-xs">{e.actor_id}</strong>
                      <Chip tone="neutral">{e.role}</Chip>
                      <Chip tone="accent">{e.action}</Chip>
                      {e.new_disposition ? (
                        <DispositionChip
                          value={e.new_disposition as "accept" | "review" | "quarantine"}
                        />
                      ) : null}
                      {e.refs_seq !== null ? (
                        <span className="text-2xs text-ink-faint">
                          approves seq{" "}
                          <span className="font-mono">{e.refs_seq}</span>
                        </span>
                      ) : null}
                      <span className="ml-auto">
                        <HostTime value={e.created_at_utc} />
                      </span>
                    </div>
                    {e.justification ? (
                      <p className="mt-2 text-xs text-ink-muted">{e.justification}</p>
                    ) : null}
                  </Panel>
                </SeqItem>
              ))}
            </SeqRail>
          ) : detail && !detail.events_available ? (
            <Empty title="The decision record could not be read" hatched>
              <p>
                The ledger daemon is not answering, so this finding&rsquo;s history is
                unknown — which is not the same as empty.
              </p>
            </Empty>
          ) : (
            <Empty title="No human has acted on this finding">
              <p>
                Its disposition is the one the risk engine wrote ({f.disposition}, by rule{" "}
                {f.disposition_rule}).
              </p>
            </Empty>
          )}
        </div>
      </div>
    </Sheet>
  );
}

function Field({
  label,
  note,
  children,
}: {
  label: string;
  note: string;
  children: React.ReactNode;
}) {
  return (
    <div className="min-w-[6rem]">
      <div className="font-mono text-2xs uppercase tracking-wider text-ink-faint">
        {label}
      </div>
      <div className="mt-0.5">{children}</div>
      <div className="text-2xs text-ink-faint">{note}</div>
    </div>
  );
}
