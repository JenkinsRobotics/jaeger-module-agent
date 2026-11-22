# ruff: noqa: E701, E702
"""A host's side of the agent: start the worker (loop.py), talk to it, close it.

The worker runs in the Python that has Jaeger Agent and its voice and hearing modules: this one, unless the host
names another. One JSON line each way; loop.py lists them. device.Device does all of this for a body; a host that
is not one (a browser page) uses it directly:

    host = AgentHost(log=..., cwd=...)
    host.on_control = body.control                          # the agent's avatar tool moves this body
    host.start({'op': 'start', 'persona': ..., 'session': ...})
    events = host.listen()                                 # words, voice sound with its lip marks and cues, scene notes
    host.send({'op': 'text', 'text': 'Hello'})

Moved from JaegerAvatar (jaeger_avatar/embedded/agent_host.py) on November 22, 2026. Until that copy is removed, a fix
made there is made here too.
"""
import atexit
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import uuid

PACKAGE = __package__                                                   # jaeger_agent.body
TOP = Path(__file__).resolve().parents[len(PACKAGE.split('.'))]         # the folder Python finds that package in


class AgentHost:
    def __init__(self, python=None, log=None, cwd=None):
        self.python, self.log, self.cwd = python or sys.executable, log, cwd
        self.process = None; self.ready = False; self.schema = None; self.model = None; self.pending = {}; self.lock = threading.RLock()
        self.why = ''; self.report = None; self.listeners = set(); self.on_control = None   # what it runs on, who hears its events, who moves the avatar
        atexit.register(self.close)

    def command(self, first):
        """What to run: the voice loop. A host with another worker of its own answers differently."""
        return [str(self.python), '-u', '-m', PACKAGE + '.loop']

    def start(self, first=None):
        """Start the worker; it reads `first` (its start line) before loading. Nothing starts without that Python."""
        if not self.python or not Path(self.python).is_file(): return
        self.why = ''
        env = {**os.environ, 'PYTHONPATH': os.pathsep.join(p for p in (str(TOP), os.environ.get('PYTHONPATH')) if p)}
        if self.log: Path(self.log).parent.mkdir(parents=True, exist_ok=True)
        with (open(self.log, 'ab') if self.log else open(os.devnull, 'wb')) as out:
            self.process = process = subprocess.Popen(self.command(first), cwd=self.cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=out, text=True, bufsize=1)
        if first: process.stdin.write(json.dumps(first) + '\n'); process.stdin.flush()
        threading.Thread(target=self._read, args=(process,), daemon=True).start()

    def restart(self, first=None):
        self.close(); self.start(first)

    def listen(self):
        """A queue that gets every voice loop event from now on; give it back with unlisten."""
        q = queue.Queue()
        with self.lock: self.listeners.add(q)
        return q

    def unlisten(self, q):
        with self.lock: self.listeners.discard(q)

    def _tell(self, event):
        with self.lock:
            for q in self.listeners: q.put(event)

    def _read(self, process=None):
        process = process or self.process
        for line in process.stdout:
            try: data = json.loads(line)
            except ValueError: continue
            if 'control' in data:   # the agent's avatar tool asks; the answer goes back on the same pipe
                c = data['control']
                try: result = self.on_control(c.get('op'), c.get('value'), c.get('ttl_ms'))
                except Exception as exc: result = {'ok': False, 'error': str(exc) if self.on_control else 'No avatar attached'}
                with self.lock: process.stdin.write(json.dumps({'op': 'control_result', 'id': data.get('id'), 'result': result}) + '\n'); process.stdin.flush()
                continue
            if 'event' in data: self._tell(data); continue
            with self.lock:
                if data.get('ready'): self.ready = True; self.schema = data.get('schema'); self.report = data.get('report')
                elif 'report' in data: self.report = data['report']   # the loop's voice changed: what it runs on now
                if data.get('why'): self.why = data['why']   # why the worker is about to stop, in its words
                if data.get('model'): self.model = data['model']   # the model file now loaded in the worker
                target = self.pending.get(data.get('id'))
                if target: target.put(data)
        with self.lock:
            if self.process is not process: return   # an old worker ending after a restart
            self.ready = False
            for target in self.pending.values(): target.put({'error': 'Agent worker stopped', 'done': True})
        self._tell({'event': 'status', 'text': 'The agent stopped'}); self._tell({'event': 'done'})

    def send(self, data):
        with self.lock:
            if not self.ready: raise RuntimeError('JaegerAgent is not ready')
            self.process.stdin.write(json.dumps(data) + '\n'); self.process.stdin.flush()

    def ask(self, data, timeout=60):
        """Send a line that is answered by one line with the same id, and return that answer."""
        rid = uuid.uuid4().hex; target = queue.Queue()
        with self.lock: self.pending[rid] = target
        try:
            self.send({**data, 'id': rid})
            try: item = target.get(timeout=timeout)
            except queue.Empty: raise RuntimeError('The agent did not answer')
            if item.get('error'): raise RuntimeError(item['error'])
            return item
        finally:
            with self.lock: self.pending.pop(rid, None)

    def say(self, text, voice=None, timeout=60):
        """One line through the voice loop's own voice (or `voice`, for this line), no model: {rate, pcm, marks, ms}."""
        return self.ask(dict(op='say', text=text, voice=voice), timeout)

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=3)
            except subprocess.TimeoutExpired: self.process.kill(); self.process.wait()
        self.ready = False
