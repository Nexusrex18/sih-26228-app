"""Drive a real, installed Chrome over the DevTools protocol — standard library only.

The HTTP-level e2e tests prove every route answers. They cannot prove the page RENDERS:
a React component that throws on load serves a perfectly good HTML shell and a blank
screen. Only a browser executing the bundle can see that, so this runs one.

No Playwright, no Selenium, no websocket package: the air-gapped build cannot fetch them,
and a 100-line RFC 6455 client over `socket` is enough for the DevTools protocol. Chrome
is launched headless with its own throwaway profile and killed afterwards.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

CHROME_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")


def find_chrome() -> str | None:
    for name in CHROME_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


class _WebSocket:
    """Just enough RFC 6455 for a client: text frames out (masked), frames in."""

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        assert url.startswith("ws://"), url
        hostport, _, path = url[len("ws://"):].partition("/")
        host, _, port = hostport.partition(":")
        self.sock = socket.create_connection((host, int(port)), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(
            (f"GET /{path} HTTP/1.1\r\nHost: {hostport}\r\nUpgrade: websocket\r\n"
             f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
             "Sec-WebSocket-Version: 13\r\n\r\n").encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("websocket handshake: connection closed")
            head += chunk
        status = head.split(b"\r\n", 1)[0]
        if b" 101 " not in status:
            raise ConnectionError(f"websocket handshake refused: {status!r}")
        self._buf = head.split(b"\r\n\r\n", 1)[1]

    def _read(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("websocket closed")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def send(self, text: str) -> None:
        payload = text.encode()
        header = bytearray([0x81])                       # FIN + text
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header += struct.pack(">H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", n)
        mask = os.urandom(4)
        header += mask
        self.sock.sendall(bytes(header) + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

    def recv(self) -> str:
        parts: list[bytes] = []
        while True:
            b0, b1 = self._read(2)
            opcode, fin = b0 & 0x0F, b0 & 0x80
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            data = self._read(n)
            if opcode == 0x9:                             # ping -> pong
                self.sock.sendall(bytes([0x8A, 0x80 | len(data)]) + b"\0\0\0\0" + data)
                continue
            if opcode == 0x8:
                raise ConnectionError("websocket closed by the browser")
            parts.append(data)
            if fin:
                return b"".join(parts).decode("utf-8", "replace")

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class Chrome:
    """A headless Chrome with one page, and the errors that page produced."""

    def __init__(self, chrome: str, *, width: int = 1280, height: int = 900) -> None:
        self.profile = Path(tempfile.mkdtemp(prefix="cva-e2e-chrome-"))
        self.proc = subprocess.Popen(
            [chrome, "--headless=new", "--remote-debugging-port=0",
             f"--user-data-dir={self.profile}", "--no-first-run",
             "--no-default-browser-check", "--disable-extensions", "--disable-gpu",
             "--disable-dev-shm-usage", "--disable-background-networking",
             "--disable-component-update", "--disable-sync", "--metrics-recording-only",
             f"--window-size={width},{height}", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        port_file = self.profile / "DevToolsActivePort"
        deadline = time.monotonic() + 30
        while not port_file.exists():
            if self.proc.poll() is not None or time.monotonic() > deadline:
                self.quit()
                raise RuntimeError("Chrome did not start its DevTools endpoint")
            time.sleep(0.1)
        port = int(port_file.read_text().split()[0])
        targets = json.loads(urllib.request.urlopen(
            f"http://127.0.0.1:{port}/json/list", timeout=10).read())
        page = next(t for t in targets if t.get("type") == "page")
        self.ws = _WebSocket(page["webSocketDebuggerUrl"])
        self._id = 0
        self.events: list[dict[str, Any]] = []
        for domain in ("Page", "Runtime", "Log", "Network"):
            self.call(f"{domain}.enable")

    # -- protocol ---------------------------------------------------------------------------
    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._id += 1
        my_id = self._id
        self.ws.send(json.dumps({"id": my_id, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == my_id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            if "method" in msg:
                self.events.append(msg)

    def evaluate(self, expression: str) -> Any:
        r = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True,
                                           "awaitPromise": True})
        if "exceptionDetails" in r:
            raise RuntimeError(f"evaluate failed: {r['exceptionDetails'].get('text')}: "
                               f"{expression[:120]}")
        return r.get("result", {}).get("value")

    # -- what a test needs --------------------------------------------------------------------
    def goto(self, url: str) -> None:
        self.call("Page.navigate", {"url": url})
        self.wait_for("document.readyState === 'complete'", what=f"load {url}")

    def wait_for(self, js_condition: str, *, what: str, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        last: Any = None
        while time.monotonic() < deadline:
            try:
                last = self.evaluate(f"Boolean({js_condition})")
            except RuntimeError:
                last = None
            if last:
                return
            time.sleep(0.15)
        try:
            dialog = str(self.evaluate(
                "(document.querySelector('[role=dialog]') || {}).innerText || ''") or "")
        except RuntimeError:
            dialog = ""
        if dialog:
            raise AssertionError(f"timed out waiting for {what}. Open dialog:\n{dialog[:1500]}")
        text = self.text()[:600]
        raise AssertionError(f"timed out waiting for {what}. Page text:\n{text}")

    def wait_text(self, needle: str, *, timeout: float = 20.0) -> None:
        self.wait_for(f"document.body && document.body.innerText.includes({json.dumps(needle)})",
                      what=f"text {needle!r}", timeout=timeout)

    def text(self) -> str:
        try:
            return str(self.evaluate("document.body ? document.body.innerText : ''") or "")
        except RuntimeError:
            return ""

    def url(self) -> str:
        return str(self.evaluate("location.href"))

    def click_text(self, text: str, selector: str = "button, a") -> None:
        """Click the first visible element matching `selector` whose text contains `text`."""
        found = self.evaluate(f"""(() => {{
            const els = [...document.querySelectorAll({json.dumps(selector)})]
              .filter(e => e.offsetParent !== null && !e.disabled
                           && e.innerText.includes({json.dumps(text)}));
            if (!els.length) return false;
            els[0].scrollIntoView({{block: 'center'}});
            els[0].click();
            return true;
        }})()""")
        assert found, f"no clickable {selector} containing {text!r}. Page:\n{self.text()[:600]}"

    def type_into(self, selector: str, text: str) -> None:
        """Focus and TYPE — real input events, so React's controlled inputs see them."""
        ok = self.evaluate(f"""(() => {{
            const el = document.querySelector({json.dumps(selector)});
            if (!el) return false; el.focus(); return true; }})()""")
        assert ok, f"no element {selector}"
        self.call("Input.insertText", {"text": text})

    def select(self, selector: str, value: str) -> None:
        ok = self.evaluate(f"""(() => {{
            const el = document.querySelector({json.dumps(selector)});
            if (!el) return false;
            const setter = Object.getOwnPropertyDescriptor(
                HTMLSelectElement.prototype, 'value').set;
            setter.call(el, {json.dumps(value)});
            el.dispatchEvent(new Event('change', {{bubbles: true}}));
            return el.value === {json.dumps(value)};
        }})()""")
        assert ok, f"could not select {value!r} in {selector}"

    def set_viewport(self, width: int, height: int, *, mobile: bool) -> None:
        self.call("Emulation.setDeviceMetricsOverride",
                  {"width": width, "height": height, "deviceScaleFactor": 2 if mobile else 1,
                   "mobile": mobile})

    def screenshot(self, path: Path) -> None:
        data = self.call("Page.captureScreenshot", {"format": "png"})["data"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(data))

    def errors(self) -> list[str]:
        """Uncaught exceptions, console.error calls, and error-level log entries."""
        out = []
        for e in self.events:
            m, p = e.get("method"), e.get("params", {})
            if m == "Runtime.exceptionThrown":
                d = p.get("exceptionDetails", {})
                out.append("exception: " + str(
                    d.get("exception", {}).get("description") or d.get("text")))
            elif m == "Runtime.consoleAPICalled" and p.get("type") == "error":
                out.append("console.error: " + " ".join(
                    str(a.get("value", a.get("description", ""))) for a in p.get("args", [])))
            elif m == "Log.entryAdded" and p.get("entry", {}).get("level") == "error":
                entry = p["entry"]
                out.append(f"log[{entry.get('source')}]: {entry.get('text')} {entry.get('url', '')}")
        return out

    def clear_events(self) -> None:
        self.events.clear()

    def quit(self) -> None:
        try:
            if hasattr(self, "ws"):
                self.ws.close()
        finally:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            shutil.rmtree(self.profile, ignore_errors=True)
