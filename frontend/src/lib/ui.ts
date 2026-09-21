import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { Availability, Disposition, Severity } from "./types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** `accept` is the weakest claim about an asset and `quarantine` the strongest, so an
 *  override *towards* quarantine is a raise and takes effect immediately; an override away
 *  from it needs a second person (plan D-E7). */
export const DISPOSITION_RANK: Record<Disposition, number> = {
  accept: 0,
  review: 1,
  quarantine: 2,
};

export function raises(from: Disposition, to: Disposition) {
  return DISPOSITION_RANK[to] > DISPOSITION_RANK[from];
}

export const DISPOSITION_TONE: Record<string, string> = {
  quarantine: "text-quarantine border-quarantine/35 bg-quarantine/10",
  review: "text-review border-review/35 bg-review/10",
  accept: "text-accept border-accept/35 bg-accept/10",
  pending: "text-pending border-pending/35 bg-pending/10",
  absent: "text-absent border-absent/30 bg-absent/[0.07]",
};

export const SEVERITY_TONE: Record<Severity, string> = {
  critical: "text-quarantine",
  high: "text-quarantine",
  medium: "text-review",
  low: "text-ink-muted",
  info: "text-ink-faint",
};

export const SEVERITY_ORDER: Severity[] = [
  "critical",
  "high",
  "medium",
  "low",
  "info",
];

export function isAbsent(state: Availability) {
  return state === "UNAVAILABLE" || state === "DEGRADED" || state === "ERROR";
}

export function short(value: string | null | undefined, n = 12) {
  if (!value) return "—";
  return value.length <= n ? value : `${value.slice(0, n)}…`;
}

/** Times come from the host clock, which this system does not trust. Everything that
 *  renders one says so; ordering is always by ledger `seq`. */
export function hostTime(iso: string | null | undefined) {
  if (!iso) return "—";
  return iso.slice(0, 19).replace("T", " ");
}

export function pct(value: number, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

export function fixed(value: number | null | undefined, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

/** Mirrors the server's justification rule. The rule is WORDS, not characters: a string of
 *  punctuation clears a length floor and records nothing. Shown live so an analyst is never
 *  surprised by a refusal after typing three paragraphs. */
export function justificationCheck(
  text: string,
  minChars: number,
  minWords: number,
) {
  const trimmed = text.trim();
  const words = trimmed.split(/\s+/).filter((w) => /[^\W_]/.test(w));
  return {
    chars: trimmed.length,
    words: words.length,
    ok: trimmed.length >= minChars && words.length >= minWords,
  };
}
