"""SubprocessModel — a model behind an executable, for anything that will not import here.

The protocol is the smallest one that carries a tensor: the command reads a float32 `.npy` on
stdin and writes a `.npy` of outputs (logits, or probabilities with `returns_logits=False`) on
stdout. NumPy's `.npy` is read with `allow_pickle=False`: the process on the other end is the
thing under audit, so its output is untrusted bytes and never a pickle. (A command must go through a
BytesIO in both directions, `np.load(io.BytesIO(sys.stdin.buffer.read()))` and `np.save` to a
BytesIO then `sys.stdout.buffer.write`: a pipe is not seekable and NumPy raises on it.)

What this adapter does NOT do is contain the command. The process-level egress guard sees
Python's `socket` module and nothing else, so it cannot see a subprocess (plan §5.8, R7); only
`unshare -rn` proves the command had no route. The command is chosen by the operator running
the scan, not by the artefact, but a wrapped model is still a program that runs.
"""
from __future__ import annotations

import io
import subprocess
from collections.abc import Sequence

import numpy as np

from .callable import CallableHandle

MAX_OUTPUT_BYTES = 64 << 20


class SubprocessModel(CallableHandle):
    fmt = "subprocess"

    def __init__(self, command: Sequence[str], model_id: str, input_shape: tuple[int, ...],
                 num_classes: int, returns_logits: bool = True, timeout_s: float = 60.0):
        if not command:
            raise ValueError("SubprocessModel needs a non-empty command")
        self._command = [str(c) for c in command]      # a list, never a shell string
        self._timeout_s = timeout_s
        super().__init__(self._call, model_id, input_shape, num_classes, returns_logits)

    def _call(self, x: np.ndarray) -> np.ndarray:
        buf = io.BytesIO()
        np.save(buf, np.asarray(x, dtype=np.float32), allow_pickle=False)
        proc = subprocess.run(self._command, input=buf.getvalue(), capture_output=True,
                              timeout=self._timeout_s, check=False)
        if proc.returncode != 0:
            tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["no stderr"]
            raise RuntimeError(f"command exited {proc.returncode}: {tail[0][:200]}")
        if len(proc.stdout) > MAX_OUTPUT_BYTES:
            raise RuntimeError(f"command wrote {len(proc.stdout)} bytes; the cap is "
                               f"{MAX_OUTPUT_BYTES}")
        return np.asarray(np.load(io.BytesIO(proc.stdout), allow_pickle=False),
                          dtype=np.float32)
