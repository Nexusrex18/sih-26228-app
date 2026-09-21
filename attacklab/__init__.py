"""Attack lab (repo root, shared across modules — NOT under ``cva/``, per backend_plan.md §6.4).

Module A's data-poisoning scripts live here. Every script takes an explicit ``seed`` and threads a
``numpy.random.Generator`` through all randomness (never the global RNG), and returns
``(new_dataset, manifest)`` where the manifest is plain JSON-able ground truth that tests and the
benchmark score against. Same seed + same inputs -> identical manifest and identical file bytes.
"""
