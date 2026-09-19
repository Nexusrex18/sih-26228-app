"""Module C — inference provenance & output integrity (PS 2.2.3, audit-trail half of 2.2.5).

Two sub-packages with different import rules (plan D13):

  seal/    STANDALONE Mode-C SDK. May import only the stdlib, `cryptography`, `rfc8785` and itself
           (CI invariant 4, tests/provenance/test_import_allowlist.py).
  checks/  scan-side plug-ins (`prov.ledger_verify`, `prov.recompute`). May import `core/`.

Neither may import `detectors/`, `features/` or `risk/` (CI invariant 1).
Build source: sih26228-notes/Plan/Module-C-Provenance-Seal-Plan.md
"""
