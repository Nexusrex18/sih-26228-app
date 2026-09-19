"""V14 — the reproducibility harness (cva/core/determinism.py, backend_plan.md §2.9).

`arm()` mutates process-global state (three RNGs, torch's deterministic-algorithms switch and
the CVA_DETERMINISTIC env var), so an autouse fixture snapshots all of it and puts it back.
Nothing here re-executes the interpreter: `os.execv` is replaced before `ensure_hashseed` can
reach it.
"""
from __future__ import annotations

import os
import random
import sys
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cva.core import determinism  # noqa: E402
from cva.core.determinism import (  # noqa: E402
    ENV_FLAG,
    arm,
    armed,
    ensure_hashseed,
    hashseed,
    ort_session_options,
    recorded_seeds,
)


@pytest.fixture(autouse=True)
def isolated_global_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Start every test unarmed and undo whatever `arm()` does to the process."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    torch_det = torch.are_deterministic_algorithms_enabled()
    torch_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    # setenv-then-delenv makes monkeypatch remember the ORIGINAL value (or absence), so a later
    # direct `os.environ[...] = ...` inside arm()/ensure_hashseed() is rolled back on teardown.
    for name in (ENV_FLAG, "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED"):
        monkeypatch.setenv(name, "sentinel")
        monkeypatch.delenv(name)
    torch.use_deterministic_algorithms(False)                 # every test starts unarmed
    yield
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
    torch.use_deterministic_algorithms(torch_det, warn_only=torch_warn_only)


def draws() -> dict[str, float]:
    return {"python": random.random(),
            "numpy": float(np.random.rand()),
            "torch": float(torch.rand(1).item())}


# --------------------------------------------------------------------------------------
# arm(): seeds
# --------------------------------------------------------------------------------------
def test_arm_same_seed_gives_the_same_draw_in_all_three_generators() -> None:
    arm(1234)
    first = draws()
    arm(1234)
    second = draws()
    assert first == second


def test_arm_different_seed_gives_a_different_draw_in_all_three_generators() -> None:
    arm(1234)
    first = draws()
    arm(1235)
    second = draws()
    for name in ("python", "numpy", "torch"):
        assert first[name] != second[name], name


def test_arm_pins_python_random() -> None:
    arm(7)
    a = [random.random() for _ in range(5)]
    arm(7)
    assert [random.random() for _ in range(5)] == a


def test_arm_pins_numpy() -> None:
    arm(7)
    a = np.random.rand(5).tolist()
    arm(7)
    assert np.random.rand(5).tolist() == a


def test_arm_pins_torch() -> None:
    arm(7)
    a = torch.rand(5).tolist()
    arm(7)
    assert torch.rand(5).tolist() == a


def test_arm_reseeding_rewinds_the_stream_rather_than_continuing_it() -> None:
    arm(5)
    a = random.random()
    _ = random.random()
    arm(5)
    assert random.random() == a


def test_arm_numpy_seed_is_reduced_modulo_2_pow_32() -> None:
    arm(3)
    a = np.random.rand(3).tolist()
    arm(2**32 + 3)
    assert np.random.rand(3).tolist() == a


def test_arm_accepts_a_seed_wider_than_32_bits() -> None:
    arm(2**40 + 1)                                            # numpy alone would reject this raw
    assert armed()


def test_arm_enables_torch_deterministic_algorithms() -> None:
    assert not torch.are_deterministic_algorithms_enabled()
    arm(0)
    assert torch.are_deterministic_algorithms_enabled()


def test_arm_sets_the_deterministic_flag_and_armed_reads_it() -> None:
    assert not armed()
    arm(0)
    assert os.environ[ENV_FLAG] == "1"
    assert armed()


def test_armed_is_true_only_for_the_literal_one(monkeypatch: pytest.MonkeyPatch) -> None:
    for value, expect in (("1", True), ("0", False), ("true", False), ("", False)):
        monkeypatch.setenv(ENV_FLAG, value)
        assert armed() is expect, value
    monkeypatch.delenv(ENV_FLAG)
    assert armed() is False


def test_arm_without_torch_still_seeds_python_and_numpy_and_does_not_raise(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # scoped, so torch is importable again before the fixture teardown touches it
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "torch", None)                 # `import torch` -> ImportError
        arm(11)
        a = (random.random(), float(np.random.rand()))
        arm(11)
        assert (random.random(), float(np.random.rand())) == a
        assert armed()


def test_arm_never_sets_cublas_workspace_config() -> None:
    """That variable belongs to the GPU overlay; the CPU profile must not carry it."""
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    arm(42)
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_arm_does_not_touch_pythonhashseed() -> None:
    """Assigning it after start-up would change nothing for this process yet make a report lie."""
    arm(42)
    assert "PYTHONHASHSEED" not in os.environ


# --------------------------------------------------------------------------------------
# PYTHONHASHSEED is read from the environment, never assigned by arm()
# --------------------------------------------------------------------------------------
def test_hashseed_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    assert hashseed() is None
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    assert hashseed() == "42"
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    assert hashseed() == "random"


def forbid_execv(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"os.execv was reached: {args!r}")
    monkeypatch.setattr(os, "execv", boom)


def test_ensure_hashseed_is_a_no_op_when_the_variable_is_already_set(
        monkeypatch: pytest.MonkeyPatch) -> None:
    forbid_execv(monkeypatch)
    monkeypatch.setenv("PYTHONHASHSEED", "7")
    ensure_hashseed(1)                                        # would raise if it re-executed
    assert os.environ["PYTHONHASHSEED"] == "7"                # and it must not overwrite it


def test_ensure_hashseed_treats_an_empty_but_set_variable_as_set(
        monkeypatch: pytest.MonkeyPatch) -> None:
    forbid_execv(monkeypatch)
    monkeypatch.setenv("PYTHONHASHSEED", "")
    ensure_hashseed(1)
    assert os.environ["PYTHONHASHSEED"] == ""


def test_ensure_hashseed_sets_the_variable_and_reexecs_once_when_absent(
        monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(os, "execv", lambda path, argv: calls.append((path, list(argv))))
    monkeypatch.setattr(sys, "orig_argv", [sys.executable, "-m", "cva", "scan"], raising=False)
    assert "PYTHONHASHSEED" not in os.environ
    ensure_hashseed(99)
    assert os.environ["PYTHONHASHSEED"] == "99"
    assert calls == [(sys.executable, [sys.executable, "-m", "cva", "scan"])]


def test_ensure_hashseed_does_not_reexec_a_second_time(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(os, "execv", lambda path, argv: calls.append((path, list(argv))))
    monkeypatch.setattr(sys, "orig_argv", [sys.executable, "-m", "cva"], raising=False)
    ensure_hashseed(5)
    ensure_hashseed(5)                                        # the re-exec'd child sees it set
    assert len(calls) == 1


# --------------------------------------------------------------------------------------
# ONNX Runtime session options
# --------------------------------------------------------------------------------------
def test_ort_session_options_is_none_when_not_armed() -> None:
    assert not armed()
    assert ort_session_options() is None


def test_ort_session_options_is_pinned_to_one_sequential_thread_when_armed() -> None:
    ort = pytest.importorskip("onnxruntime")
    arm(0)
    so = ort_session_options()
    assert so is not None
    assert so.intra_op_num_threads == 1
    assert so.inter_op_num_threads == 1
    assert so.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL


def test_ort_session_options_follow_the_flag_not_a_prior_arm_call(
        monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("onnxruntime")
    monkeypatch.setenv(ENV_FLAG, "1")
    assert ort_session_options() is not None
    monkeypatch.setenv(ENV_FLAG, "0")
    assert ort_session_options() is None


# --------------------------------------------------------------------------------------
# recorded_seeds: a report may only claim what was actually seeded
# --------------------------------------------------------------------------------------
def test_recorded_seeds_reports_only_the_scan_seed_when_unarmed() -> None:
    assert recorded_seeds(17) == {"scan": 17}


def test_recorded_seeds_adds_python_numpy_and_torch_when_armed() -> None:
    arm(17)
    assert recorded_seeds(17) == {"scan": 17, "python": 17, "numpy": 17, "torch": 17}


def test_recorded_seeds_numpy_entry_is_the_reduced_seed() -> None:
    arm(2**32 + 9)
    seeds = recorded_seeds(2**32 + 9)
    assert seeds["numpy"] == 9
    assert seeds["python"] == 2**32 + 9 and seeds["torch"] == 2**32 + 9


def test_recorded_seeds_claims_match_what_arm_did() -> None:
    """The recorded numpy seed, replayed by hand, reproduces the stream arm() left behind."""
    arm(2**32 + 9)
    from_arm = np.random.rand(4).tolist()
    np.random.seed(recorded_seeds(2**32 + 9)["numpy"])
    assert np.random.rand(4).tolist() == from_arm


def test_module_constants_match_the_documented_contract() -> None:
    assert determinism.ENV_FLAG == "CVA_DETERMINISTIC"
