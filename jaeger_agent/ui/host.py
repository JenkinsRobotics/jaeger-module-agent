"""Jaeger Agent's own UI host: the voice loop in a window of its own, no app around it.

`python -m jaeger_agent.ui` starts the loop worker (jaeger_agent.body.loop, this same Python),
serves the built React page beside it (ui/ builds into static/), and bridges the two:
every loop event goes out as server-sent events, a typed line goes in as a text turn.
Sound is the agent's own — it hears and speaks on this machine's microphone and speaker
(its default half duplex), so the page only reads and writes words.

Loopback only, like the loop itself. The first start loads the model and takes a while;
/health says ready when the loop has reported in.
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).parent / "static"
EVENT_WAIT = 15.0


class Loop:
    """The worker and everyone listening to it."""

    def __init__(self, start_line):
        self.start_line = start_line
        self.process = None
        self.ready = False
        self.model = ""
        self.report = None
        self.lock = threading.Lock()
        self.listeners: set[queue.Queue] = set()

    def start(self):
        self.process = subprocess.Popen([sys.executable, "-u", "-m", "jaeger_agent.body.loop"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, bufsize=1)
        self.process.stdin.write(json.dumps(self.start_line) + "\n")
        self.process.stdin.flush()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("ready"):
                self.ready, self.model, self.report = True, event.get("model", ""), event.get("report")
            with self.lock:
                for q in self.listeners:
                    q.put(event)
        self.ready = False

    def send(self, data):
        self.process.stdin.write(json.dumps(data) + "\n")
        self.process.stdin.flush()

    def listen(self):
        q = queue.Queue()
        with self.lock:
            self.listeners.add(q)
        return q

    def unlisten(self, q):
        with self.lock:
            self.listeners.discard(q)


loop: Loop = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, body, kind="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self.reply(200, json.dumps({"ready": loop.ready, "model": loop.model, "report": loop.report}, default=str).encode())
        if self.path == "/events":   # everything the loop does, as it happens
            q = loop.listen()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                while True:
                    try:
                        line = b"data: " + json.dumps(q.get(timeout=EVENT_WAIT), default=str).encode() + b"\n\n"
                    except queue.Empty:
                        line = b": keep\n\n"
                    self.wfile.write(line)
                    self.wfile.flush()
            except OSError:   # the page closed
                pass
            finally:
                loop.unlisten(q)
            return
        path = STATIC / (self.path.lstrip("/") or "index.html")
        if path.is_file() and STATIC in path.resolve().parents or path == STATIC / "index.html":
            kind = {"html": "text/html", "js": "text/javascript", "css": "text/css", "svg": "image/svg+xml"}.get(path.suffix[1:], "application/octet-stream")
            return self.reply(200, path.read_bytes(), kind)
        self.reply(404, b'{"error":"not found"}')

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path == "/say":
            try:
                text = str(json.loads(body)["text"])[:2000].strip()
            except (ValueError, KeyError):
                return self.reply(422, b'{"error":"Send {text}"}')
            if not text:
                return self.reply(422, b'{"error":"empty"}')
            loop.send({"op": "text", "text": text})
            return self.reply(200, b'{"ok":true}')
        if self.path == "/interrupt":
            loop.send({"op": "interrupt"})
            return self.reply(200, b'{"ok":true}')
        self.reply(404, b'{"error":"not found"}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--model", default=None, help="a model name the agent knows (default: its own)")
    ap.add_argument("--hearing", default="large-v3-turbo", help="Whisper model for its ears")
    ap.add_argument("--no-open", action="store_true", help="don't open the page in a browser")
    args = ap.parse_args(argv)
    if not (STATIC / "index.html").is_file():
        print("No page built yet: run `npm install && npm run build` in ui/ once.", file=sys.stderr)
        return 1
    global loop
    loop = Loop({"op": "start", "session": "agent-ui", "hearing": args.hearing, "body": False,   # no stage here: words and voice only
                 **({"model": args.model} if args.model else {})})
    loop.start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Jaeger Agent UI: {url}  (the agent hears and speaks on this machine)")
    if not args.no_open:
        threading.Thread(target=lambda: (time.sleep(1), webbrowser.open(url)), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if loop.process:
            loop.process.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
