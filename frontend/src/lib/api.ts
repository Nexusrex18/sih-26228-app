"use client";

import type {
  Coverage,
  Finding,
  Health,
  PlanRow,
  Refusal,
  ScanDetail,
  ScanSummary,
  Session,
  TargetState,
  TimelineRow,
} from "./types";

/**
 * The Flask API client.
 *
 * Same-origin, session-cookie auth, CSRF token on every write. There is no bearer token and
 * no second credential path: adding one would mean a second way to authenticate to the most
 * exposed process on the host.
 */
const BASE = "/api";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly payload: Partial<Refusal> & { detail?: string },
  ) {
    super(payload.detail || payload.error || `request failed (${status})`);
    this.name = "ApiError";
  }

  /** True when the server refused deliberately and nothing was written. */
  get refused(): boolean {
    return this.payload.nothing_changed === true;
  }

  get code(): string {
    return this.payload.error ?? "unknown";
  }

  get title(): string {
    return this.payload.title ?? "That action was refused";
  }
}

let csrfToken = "";

export function setCsrfToken(token: string) {
  csrfToken = token;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set("Accept", "application/json");
  if (init?.method && init.method !== "GET") {
    headers.set("Content-Type", "application/json");
    headers.set("X-CSRF-Token", csrfToken);
  }
  const resp = await fetch(`${BASE}${path}`, {
    ...init,
    headers,
    credentials: "same-origin",
  });
  const text = await resp.text();
  let payload: unknown = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = { detail: text.slice(0, 400) };
  }
  if (!resp.ok) {
    throw new ApiError(resp.status, (payload ?? {}) as Partial<Refusal>);
  }
  return payload as T;
}

export const api = {
  session: () => request<Session>("/session"),
  health: () => request<Health>("/health"),
  reverify: () => request<Health>("/audit/verify", { method: "POST", body: "{}" }),

  scans: () =>
    request<{ ledger_readable: boolean; scans: ScanSummary[] }>("/scans"),

  scan: (scanId: string) => request<ScanDetail>(`/scans/${scanId}`),

  findings: (scanId: string, params: Record<string, string | number | undefined>) => {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== "" && v !== null) q.set(k, String(v));
    }
    const qs = q.toString();
    return request<{
      scan_id: string;
      total: number;
      unfiltered_total: number;
      page: number;
      per_page: number;
      pages: number;
      modules: string[];
      filters: Record<string, string | null>;
      plan: PlanRow[];
      findings: Finding[];
    }>(`/scans/${scanId}/findings${qs ? `?${qs}` : ""}`);
  },

  finding: (scanId: string, findingId: string) =>
    request<{
      scan_id: string;
      finding: Finding;
      is_prov: boolean;
      reason_codes: string[];
      request_id: string;
      events: import("./types").LedgerEventRow[];
      events_available: boolean;
    }>(`/scans/${scanId}/findings/${findingId}`),

  targets: (scanId: string) =>
    request<{ targets: TargetState[] }>(`/scans/${scanId}/targets`),

  coverage: (scanId: string, compare?: string) =>
    request<{
      scan_id: string;
      profile_name: string;
      budget_tier: string;
      coverage: Coverage;
      calibration: import("./types").Calibration | null;
      other?: {
        scan_id: string;
        profile_name: string;
        budget_tier: string;
        coverage: Coverage;
      };
      diff?: {
        only_left: string[];
        only_right: string[];
        both: string[];
        shrank: boolean;
        lost_reasons: Record<string, string[]>;
      };
    }>(`/scans/${scanId}/coverage${compare ? `?compare=${compare}` : ""}`),

  audit: (scanId?: string) =>
    request<{ readable: boolean; decisions: number; rows: TimelineRow[] }>(
      `/audit${scanId ? `?scan_id=${scanId}` : ""}`,
    ),

  exportLedger: () =>
    request<{ ok: true; path: string; detail: string }>("/audit/export", {
      method: "POST",
      body: "{}",
    }),

  act: (
    scanId: string,
    findingId: string,
    body: Record<string, unknown>,
  ) =>
    request<{
      ok: true;
      seq: number;
      deduped: boolean;
      action: string;
      message: string;
      finding: Finding;
      events: import("./types").LedgerEventRow[];
    }>(`/scans/${scanId}/findings/${findingId}/act`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  targetAct: (scanId: string, body: Record<string, unknown>) =>
    request<{
      ok: true;
      seq: number;
      deduped: boolean;
      message: string;
      targets: TargetState[];
    }>(`/scans/${scanId}/targets/act`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
};
