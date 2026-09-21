/** Shapes the Flask API returns. Mirrors `schemas/report.schema.json` where they overlap. */

export type Disposition = "accept" | "review" | "quarantine";
export type Severity = "info" | "low" | "medium" | "high" | "critical";
export type Nature = "adversarial" | "quality" | "indeterminate";
export type Availability = "OK" | "DEGRADED" | "UNAVAILABLE" | "ERROR";
export type TargetType =
  | "sample"
  | "contributor"
  | "batch"
  | "model"
  | "record"
  | "dataset";
export type Role = "viewer" | "analyst" | "approver" | "admin";

export interface Evidence {
  kind:
    | "image_crop"
    | "contact_sheet"
    | "heatmap"
    | "plot"
    | "table"
    | "hash"
    | "json";
  caption: string;
  path: string | null;
}

export interface PendingChange {
  seq: number;
  requested_by: string;
  action: string;
  new_disposition: Disposition | null;
  reason_code: string | null;
  justification: string;
}

export interface FindingState {
  original: Disposition;
  original_rule: string;
  effective: Disposition;
  cited_seq: number | null;
  changed_by_human: boolean;
  assignee: string | null;
  acknowledged_by: string[];
  last_seq: number;
  pending: PendingChange | null;
  superseded: { seq: number; requested_by: string; new_disposition: string | null }[];
}

export interface Finding {
  finding_id: string;
  scan_id: string;
  detector_id: string;
  detector_version: string;
  target_type: TargetType;
  target_ref: string;
  severity: Severity;
  confidence: number;
  score_raw: number;
  threshold: number;
  reason: string;
  attack_class: string;
  evidence: Evidence[];
  access_assumptions: string[];
  limitations: string[];
  disposition: Disposition;
  disposition_rule: string;
  produced_by: string;
  nature: Nature;
  exclusion_reason?: "capability" | "budget" | null;
  availability: Availability;
  state: FindingState | null;
}

export interface PlanRow {
  check_id: string;
  state: Availability;
  reason: string;
  missing?: string[];
  mode?: string | null;
  exclusion_reason?: "capability" | "budget" | null;
  estimated_cost?: string | null;
  attack_classes: string[];
  elapsed_s?: number | null;
}

export interface ContributorRow {
  group_key: "contributor" | "batch" | "source";
  group_value: string;
  contributor_source?:
    | "sidecar"
    | "directory"
    | "format_field"
    | "exif_cluster"
    | "none"
    | null;
  n_samples: number;
  n_flagged: number;
  posterior_mean: number;
  ci_low: number;
  ci_high: number;
  excludes_cohort_rate?: boolean;
  cohort_rate_used?: number;
  excludes_reference_rate?: boolean;
  disposition?: Disposition;
}

export interface SealState {
  state: "sealed" | "differs" | "not_sealed" | "unknown";
  label: string;
  detail: string;
  is_alarm: boolean;
  sealed_digest: string | null;
  on_disk_digest: string | null;
  ledger_seq: number | null;
}

export interface CoverageRow {
  attack_class: string;
  checks: string[];
}

export interface Coverage {
  assessed: CoverageRow[];
  not_assessed: CoverageRow[];
  never_covered: string[];
  operational_reports: string[];
  standing_limitations: string[];
  total_attack_classes: number;
  assessed_fraction: number;
}

export interface ProvenanceSummary {
  records_verified: number;
  anchors_checked: number;
  declared_degraded_intervals: number;
  records_after_last_anchor: number;
  custody_type: string;
  durability_window: string;
  ledger_state?: "sealed" | "differs" | "not_sealed";
  not_assessed_reason?: string;
}

export interface Calibration {
  method: string;
  brier: number | null;
  scored_on: "out_of_fold" | "none";
  calibrated_detectors: string[];
  reliability_bins: { p_mean: number; empirical: number; n: number }[];
  excluded_detectors: string[];
}

export interface ScanSummary {
  scan_id: string;
  created_at_utc: string;
  verdict: string;
  profile_name: string;
  budget_tier: string;
  readable: boolean;
  unreadable_reason: string | null;
  n_findings: number;
  counts: Record<Disposition, number>;
  seal: SealState | null;
}

export interface ScanDetail {
  scan_id: string;
  created_at_utc: string;
  verdict: string;
  produced_by: Record<string, string>;
  target: Record<string, unknown> | null;
  access_assumptions: {
    capabilities_present: string[];
    capabilities_absent: { capability: string; reason: string }[];
    consequence?: string;
  };
  plan: PlanRow[];
  contributor_risk: ContributorRow[] | null;
  contributor_baseline: Record<string, number | boolean> | null;
  contributor_baseline_unavailable?: string;
  permutation_test: {
    statistic: number;
    p_value: number;
    n_permutations: number;
    conclusion: string;
  } | null;
  provenance_summary: ProvenanceSummary | null;
  drift_summary: Record<string, unknown> | null;
  calibration: Calibration | null;
  reproduction: {
    command: string;
    volatile_paths: string[];
    seeds: Record<string, number>;
    env?: Record<string, string>;
    version_pins_hash?: string | null;
    determinism_notes?: string[];
  };
  report_sha256: string;
  seal: SealState;
  events_available: boolean;
  effective: Record<string, number>;
  tool_counts: Record<Disposition, number>;
  coverage: Coverage;
  n_findings: number;
}

export interface LedgerEventRow {
  seq: number;
  actor_id: string;
  role: string;
  action: string;
  new_disposition: string | null;
  reason_code: string | null;
  assignee: string | null;
  refs_seq: number | null;
  justification: string;
  target_ref: string;
  target_type: string;
  created_at_utc: string;
}

export interface TimelineRow {
  seq: number;
  kind: "analyst_event" | "scan_record" | "other";
  created_at_utc: string;
  actor: string;
  role: string;
  action: string;
  target: string;
  detail: string;
  justification: string;
  reason_code: string;
  refs_seq: number | null;
  scan_id: string;
  audit_flagged: boolean;
  key_id: string;
}

export interface VerifyState {
  state: "OK" | "FAILED" | "UNAVAILABLE";
  summary: string;
  detail: string;
  checked_at: string;
  records_checked: number;
  anchors_verified: number;
  /** null when the verifier does not report it (Module C's `verify --json` does not). */
  anchors_in_chain: number | null;
  unwitnessed_records: number;
  declared_gaps: number;
  durability: string;
  loss_window: string;
  command: string;
  findings: Record<string, unknown>[];
  limitations: string[];
}

export interface Health {
  reachable: boolean;
  writable: boolean;
  verified: boolean;
  kind: "verified" | "unverified" | "failed" | "unavailable";
  text: string;
  detail: string;
  verify: VerifyState | null;
}

export interface Session {
  authenticated: boolean;
  actor_id?: string;
  role?: Role;
  can?: string[];
  csrf_token: string;
  policy?: {
    four_eyes: boolean;
    min_justification_chars: number;
    min_justification_words: number;
    min_justification_chars_other: number;
  };
}

export interface TargetState {
  target_type: string;
  target_ref: string;
  status: "active" | "quarantined";
  cited_seq: number | null;
  last_seq: number;
  pending_release: PendingChange | null;
}

/** Every refusal carries this, and `nothing_changed` is the property the write path exists
 *  to guarantee — the UI repeats it verbatim rather than paraphrasing. */
export interface Refusal {
  ok: false;
  error: string;
  title: string;
  detail: string;
  nothing_changed: true;
}
