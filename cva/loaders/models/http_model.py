"""HTTPModel — a model behind an HTTP endpoint, LOCALHOST ONLY.

An air-gapped assurance tool that will POST the audited data to an arbitrary URL is an
exfiltration tool with a model-shaped API. So the host is checked at construction and only
loopback is accepted: `localhost`, `127.0.0.0/8`, `::1`. Redirects are refused too (a
loopback server answering `302 -> http://elsewhere` would otherwise walk the request off the
machine) and proxies are ignored (an `HTTP_PROXY` in the environment would carry a loopback
request to a remote proxy). This is what the loopback allow-list in plan §5.8 is for.

Protocol: `POST {"inputs": [[...]]}` -> `{"outputs": [[...]]}`. Query-only by construction:
no weights, activations, gradients or graph, and the probe records why for each.
"""
from __future__ import annotations

import ipaddress
import json
import urllib.error
import urllib.request
from urllib.parse import urlsplit

import numpy as np

from .callable import CallableHandle

MAX_RESPONSE_BYTES = 64 << 20
_LOOPBACK_NAMES = {"localhost", "localhost.localdomain"}


def require_loopback(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"HTTPModel accepts http(s) URLs only, not {parts.scheme!r}")
    host = (parts.hostname or "").lower()
    try:
        ok = ipaddress.ip_address(host).is_loopback
    except ValueError:
        ok = host in _LOOPBACK_NAMES
    if not ok:
        raise ValueError(
            f"HTTPModel is localhost-only and {host!r} is not a loopback host: the audited "
            "data would leave this machine. Run the model locally, or wrap it with "
            "SubprocessModel.")
    return url


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):   # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError(req.full_url, code,
                                     f"redirect to {newurl} refused (localhost-only)",
                                     headers, fp)


class HTTPModel(CallableHandle):
    fmt = "http"

    def __init__(self, url: str, model_id: str, input_shape: tuple[int, ...],
                 num_classes: int, returns_logits: bool = True, timeout_s: float = 30.0):
        self._url = require_loopback(url)
        self._timeout_s = timeout_s
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)
        super().__init__(self._call, model_id, input_shape, num_classes, returns_logits)

    def _call(self, x: np.ndarray) -> np.ndarray:
        body = json.dumps({"inputs": np.asarray(x, dtype=np.float32).tolist()}).encode()
        req = urllib.request.Request(self._url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        with self._opener.open(req, timeout=self._timeout_s) as resp:
            raw = resp.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise RuntimeError(f"response exceeds {MAX_RESPONSE_BYTES} bytes")
        return np.asarray(json.loads(raw)["outputs"], dtype=np.float32)
