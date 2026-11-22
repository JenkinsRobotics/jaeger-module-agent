# ruff: noqa: E701, E702
"""Jaeger Agent's voice loop as a body's talker: hearing, the model with its tools, and the voice, in one process.

Run it with the Python that has Jaeger Agent and its voice and hearing modules (host.AgentHost does):

    python -m jaeger_agent.body.loop

The same worker serves every host: Jaeger Avatar's app, Mochi, a robot. "The app" below is whichever host started it.
A host with no body ({body: false} in the start line: nothing fills the `animation` slot) gets the agent as it is with
no face: no avatar or scene tool, and its voice through the agent's own speaker.

Moved from JaegerAvatar (jaeger_avatar/embedded/agent_loop.py) on November 22, 2026. Until that copy is removed, a fix
made there is made here too.

The agent takes its hearing and voice from the Jaeger modules; this app copies none of it. The app sends typed
text and microphone sound in, and gets words and voice sound out (it plays them and moves the mouth itself).
The agent moves the body through its `avatar` tool, which asks the app over the same pipe.

A scene (lines with moves timed to them) is performed by the agent too: it reads the script, voices each line, and sends
each line's moves with its sound as cues. The app plays the sound and, when a cue's moment comes on the sound's own clock,
says so; the move is then made by the same `avatar` tool the model calls. The model can write a scene itself with its
`scene` tool; the agent performs it once that reply's words are spoken.

Turn-taking is the agent's own, with its own settings (its audio mode, its half duplex): when a phrase has ended, whether
the thought is finished or she should wait for the rest (HOLDING), and when the microphone is shut. This worker sets none of it.
The start line says only what the agent cannot know, {open_mic: true}: this host's microphone has an echo canceller (the
browser page), so sound goes straight to the engine, never behind a turn, and a person can talk over her. Every other host
(Mochi, a robot: a bare speaker and microphone) gets the agent's half duplex: the microphone is shut from her first word
until the body's speaker has played her last. The body plays the sound, so it says when: the worker sends {event: voiced, n}
when a part is all voiced, and the host answers {op: played, n} when its speaker has finished it. Until her first word the
agent goes on listening (listens_while_thinking), so a person who was only pausing can go on and is answered once, whole.
The engine says when each turn is over; that is this worker's 'done'.

In (one JSON line each): {op: start, ...} first, then {op: text | audio | persona | interrupt | control_result | say | scene | cue | played}.
Out: {ready, schema, model, report}, {event, text, data}, {event: audio, text, rate, pcm, marks, cues}, {event: done}, {control, id},
{id, rate, pcm, marks, ms} (the answer to say), {report} (after a voice change), {event: scene, text: begin | end} (around a scene's sound), {event: scene, text: refused, data} (a script it could not read),
{event: voiced, n} (half duplex only), {event: feeling, text} (what the voice is told a line feels like: the feeling the
model's avatar call set on the body, or a scene's).
"""
import base64
import json
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

CONTROL_WAIT = 3                          # seconds the avatar tool waits for the app to answer
PLAYED_WAIT = 3                           # seconds past a part's own length the engine waits to hear the body has played it
# How a turn goes, shown to the model at the start of every conversation (engine.prime): a plain reply is a feeling
# through the avatar tool and then the words; asked to act something out, the scene tool is the whole reply.
# A small model follows what is already in the conversation far more surely than an instruction: told only in the
# persona, Gemma 4 E4B called the avatar tool in 4 of 12 fresh conversations and wrote the call out as words in the other
# 8, which were then spoken; with the first exchange shown it called the tool in 24 of 24. The scene uses no named
# action, so every body can play it.
EXAMPLES = [
    ('Hello!', "Hi! It's so good to see you!", [('avatar', {'emotion': 'joy'}, {'ok': True, 'did': ['emotion joy']})]),
    ('Act out a tiny scene: look up, wait a second, then tell me what you saw.', '',
     [('scene', {'script': 'look up | hold 1 | look center | emotion surprised | say I just saw a shooting star!'}, {'ok': True, 'note': 'The scene plays now.'})]),
]


class Body:
    """The host's speaker, as one of the engine's outputs (jaeger_agent/core/audio.py: play, wait, stop). The sound itself
    goes to the host in the 'audio' events. This is how the engine's half duplex learns that the body has played it:
    the engine waits here with the microphone shut."""

    def __init__(self, emit):
        self.emit, self.done, self.n, self.given = emit, threading.Event(), 0, 0.0

    def play(self, chunk, rate):
        self.given += len(chunk) / rate

    def wait(self):
        """Everything given is voiced: tell the host, and return when it says its speaker has played it."""
        self.n += 1; self.done.clear()
        given, self.given = self.given, 0.0
        self.emit({'event': 'voiced', 'n': self.n})
        self.done.wait(given + PLAYED_WAIT)   # ponytail: a host that never says is taken to have played it by then

    def stop(self):                           # cut off: nothing more of it will be played
        self.done.set()

    def played(self, n):
        if n == self.n: self.done.set()       # an answer to an earlier part, come late, opens nothing


class Loop:
    """One engine. Typed lines, scenes and the rest arrive in order on one thread; microphone sound, an interrupt and a
    cue go in at once from the reader's thread (the engine gives each heard turn a thread of its own)."""

    def __init__(self, emit):
        self.emit, self.engine = emit, None
        self.work, self.busy, self.waiting = queue.Queue(), threading.Event(), {}
        self.body = None                      # half duplex: the host's speaker as the engine's output
        self.worker = self.ended = None       # the thread that runs the work in order; whether the engine ended a turn on it
        self.extra = {}                       # what this worker adds to the engine's report: which tool list, the load time
        self.examples = EXAMPLES              # what each conversation opens with; none for a voice with no face

    def report(self):
        """What each part runs on right now, as the engine says it."""
        return {**self.engine.report(), **self.extra}

    def on_event(self, e):
        mine = threading.current_thread() is self.worker
        if e.kind == 'turn':                  # the engine's word that a turn is over (typed, heard or a scene), before the next one's words
            self.ended = self.ended or mine
            return self.emit({'event': 'done'})
        if e.kind == 'user' and mine: return  # the page already showed what was typed; a heard turn runs on the engine's own thread
        if e.kind == 'audio':
            pcm = (np.clip(e.data, -1, 1) * 32767).astype('<i2').tobytes()
            self.emit({'event': 'audio', 'text': e.text, 'rate': self.engine.sample_rate, 'pcm': base64.b64encode(pcm).decode(),
                       'marks': [list(m) for m in getattr(e.data, 'marks', [])],   # when the lips close or tuck, from the voice module
                       'cues': list(getattr(e.data, 'cues', []))})                  # a scene's moves, timed from this sound's start
        else:
            self.emit({'event': e.kind, 'text': e.text, 'data': e.data})

    def control(self, op, value, ttl_ms):
        """The avatar tool's send: ask the app, wait for its answer."""
        rid = uuid.uuid4().hex
        box = self.waiting[rid] = queue.Queue()
        self.emit({'control': {'op': op, 'value': value, 'ttl_ms': ttl_ms}, 'id': rid})
        try: answer = box.get(timeout=CONTROL_WAIT)
        except queue.Empty: answer = {'ok': False, 'error': 'The avatar did not answer'}
        finally: self.waiting.pop(rid, None)
        self.emit({'event': 'tool', 'text': f'avatar {op}', 'data': {'value': value, 'result': answer}})   # shown in the Data flow window
        return answer

    def felt(self, op, value, ttl_ms):
        """The model's own avatar call: the body is asked, and a feeling it took is told to the voice too, so a voice that
        can act on it sounds as the face looks. (A scene's lines are voiced with their feelings by the engine itself.)"""
        answer = self.control(op, value, ttl_ms)
        if op == 'emotion' and answer.get('ok'): self.engine.feel(value)
        return answer

    def receive(self, r):
        """Called on the reader's thread for each line from the app; None means the app has gone."""
        op = r and r.get('op')
        if op == 'control_result':
            box = self.waiting.get(r.get('id'))
            if box: box.put(r.get('result') or {})
        elif op == 'interrupt':
            self.engine.end_scene()           # Stop also ends a scene that is waiting for the other person's line
            self.engine.force_listen()
        elif op == 'played':                  # the body's speaker has finished what was voiced: the engine opens the microphone
            if self.body: self.body.played(r.get('n'))
        elif op == 'audio':                   # heard now, not after the reply on its way: this is how a person goes on, or cuts in
            if r.get('pcm'): self.engine.push_audio(np.frombuffer(base64.b64decode(r['pcm']), '<i2').astype(np.float32) / 32768)
            if r.get('end'): self.engine.end_phrase()   # the page's mic went quiet or was gated: the phrase ends here
        elif op == 'cue':                     # its moment has come: it cannot wait behind the lines still being voiced
            threading.Thread(target=self.cue, args=(r.get('call'),), daemon=True).start()
        else:
            if op == 'text' and self.busy.is_set(): self.engine.force_listen()   # typed over a reply: stop it, then answer
            self.work.put(r)

    def fitter(self, strict):
        """The body's say on each call a scene will make, for the engine to ask before the scene starts, and the list
        it fills with what it left out. The body is asked once what it offers; with nobody on stage, calls pass as
        written. strict (your own scene): a call the body cannot make in full refuses the scene, so you fix the script.
        Not strict (a scene the model wrote): the scene plays without those parts, and the model is told which."""
        from .tools import fitted
        answer = self.control('offers', None, 0)
        offers, left = answer.get('offers') if answer.get('ok') else None, []
        def fit(call):
            if not offers: return call
            kept, wrong = fitted(call, offers)
            if wrong and strict: raise ValueError('This body cannot do ' + '; '.join(wrong) + '.')
            left.extend(wrong)
            return kept
        return fit, left

    def stage(self, script):
        """The scene tool's stage: the engine takes the model's script, to perform once this reply's words are spoken.
        Returns what was left out of it because the body cannot do it."""
        fit, left = self.fitter(strict=False)
        try:
            lines = self.engine.stage(script, fit=fit)
        except ValueError as exc:
            self.emit({'event': 'tool', 'text': 'scene', 'data': {'value': script, 'result': {'ok': False, 'error': str(exc)}}})
            raise
        self.emit({'event': 'tool', 'text': 'scene', 'data': {'value': script, 'result': {'ok': True, 'lines': lines, 'left_out': left}}})   # shown in the Data flow window
        return left

    def cue(self, call):
        """A scene's move: the avatar tool's own call, held for as long as the scene says (or as long as a call from the model)."""
        from .tools import avatar_tool
        call = dict(call or {})
        hold = call.pop('hold', None)
        try:
            send = self.control if hold is None else lambda op, value, _ttl: self.control(op, value, int(float(hold) * 1000))
            answer = avatar_tool(send)(**call)
        except Exception as exc:
            answer = {'ok': False, 'errors': [str(exc)]}
        if not answer.get('ok'):              # each move the body took or refused is already reported by control()
            self.emit({'event': 'tool', 'text': 'avatar cue', 'data': {'value': call, 'result': answer}})

    def say(self, r):
        """One line through the agent's voice alone, no model. The answer goes to whoever asked, by id; no turn happens."""
        began, voice, rate = time.time(), r.get('voice'), self.engine.sample_rate
        try:
            parts, marks, at = [], [], 0
            for part in self.engine.synth(str(r.get('text', '')), **({'voice': voice} if voice else {})):
                marks += [[round(a + at / rate, 3), round(b + at / rate, 3), kind] for a, b, kind in getattr(part, 'marks', [])]   # timed from the clip's start
                parts.append(np.asarray(part, dtype=np.float32)); at += len(part)
            pcm = (np.clip(np.concatenate(parts) if parts else np.zeros(0, np.float32), -1, 1) * 32767).astype('<i2').tobytes()
            self.emit({'id': r.get('id'), 'rate': rate, 'pcm': base64.b64encode(pcm).decode(), 'marks': marks,
                       'ms': round((time.time() - began) * 1000, 1)})
        except Exception as exc:
            self.emit({'id': r.get('id'), 'error': str(exc)})

    def run(self):
        self.worker = threading.current_thread()
        while (r := self.work.get()) is not None:
            op = r.get('op')
            self.ended = False
            self.busy.set()
            try:
                if op == 'text': self.engine.send_text(str(r.get('text', '')))
                elif op == 'persona':   # who it is now, and (when the app names one) the character's voice in the same voice module
                    voice = r.get('voice')
                    self.engine.switch_session(r['session'], r.get('persona', ''), **({'voice': voice} if voice else {}))
                    if self.examples: self.engine.prime(self.examples)
                    if voice: self.emit({'report': self.report()})
                elif op == 'say': self.say(r)
                elif op == 'scene': self.engine.perform(r.get('script'), fit=self.fitter(strict=True)[0])
            except Exception as exc:
                if op == 'scene' and isinstance(exc, ValueError): self.emit({'event': 'scene', 'text': 'refused', 'data': str(exc)})   # a script it could not read: the director is told, nothing is said
                elif not getattr(exc, 'told', False): self.emit({'event': 'status', 'text': f'error: {exc}'})   # the engine reports a turn's own failure
            finally:
                self.busy.clear()
                if op in ('text', 'scene') and not self.ended: self.emit({'event': 'done'})   # it never began, and the page is waiting


def model_choice(name, llms, default):
    """Which of Jaeger Agent's own models to run (its table, core/policy.py): the one named, by the agent's key or by its file's
    name as Settings shows it, when that file is on this machine; else the agent's default; None when that file is missing too."""
    key = name if name in llms else next((k for k, c in llms.items() if c.path.stem == name), default)
    return key if llms[key].present else default if llms[default].present else None


def schema():
    try:
        from importlib.metadata import version
        from jaeger_agent.core.config import AgentConfig, MultimodalConfig
        return {'version': version('jaeger-agent'), 'agent': AgentConfig.model_json_schema()['properties'],
                'multimodal': MultimodalConfig.model_json_schema()['properties']}
    except Exception as exc:
        return {'error': str(exc)}


def main():
    # The short tool list: the model reads ~15,000 tokens of tool text at start-up with every tool shown
    # (about 48 s here against about 15 s). JAEGER_FULL_TOOLS=1 shows every tool.
    os.environ.setdefault('JAEGER_TOOLSET_SCOPING', '1')
    wire = os.fdopen(os.dup(1), 'w')                         # the app's pipe; everything else that prints goes to the log
    os.dup2(2, 1); sys.stdout = sys.stderr
    lock = threading.Lock()

    def emit(data):
        with lock:
            wire.write(json.dumps(data, default=str) + '\n'); wire.flush()

    start = json.loads(sys.stdin.readline() or '{}')
    from jaeger_agent import register_tool_from_function
    from jaeger_agent.core.config import MultimodalConfig
    from jaeger_agent.core.engine import GemmaMultimodal
    from jaeger_agent.core.policy import DEFAULT_LLM, LLMS
    from jaeger_agent.core.workspace import ensure_bound
    from .tools import avatar_tool, scene_tool

    ensure_bound(Path(start.get('workspace') or Path.cwd() / '.jaeger_os/avatar'))   # the agent's own folder for this host
    # The model is Jaeger Agent's: its own table of models and its own default. The app only remembers which one was chosen.
    key = model_choice(start.get('model'), LLMS, DEFAULT_LLM)
    if not key:
        why = f"Jaeger Agent's model file is not on this machine: {LLMS[DEFAULT_LLM].path}"
        emit({'why': why})                                   # the app shows it; the download command is in dev/docs/runtime/SETUP.md
        raise SystemExit(why)
    name = LLMS[key].path.stem
    loop = Loop(emit)
    faced = start.get('body', True) is not False             # false: nothing fills the animation slot for this host
    if faced:
        register_tool_from_function(avatar_tool(loop.felt), name='avatar', side_effect='gesture')   # the agent ends the turn on words + a gesture
        register_tool_from_function(scene_tool(loop.stage), name='scene', side_effect='performance')   # the scene is the reply: no second model call to narrate it
    else: loop.examples = []                                 # the examples open with an avatar call: not for a voice with no face
    began = time.time()
    # Speech only, no wake word. The audio mode is the agent's own (its default), and so is its half duplex: the host's
    # speaker is the engine's output (Body), so the microphone is shut until the body has played her words. Only a host
    # whose microphone has an echo canceller (the browser page) asks for the agent's open mic, and then plays with no word back.
    # With no body the voice goes to the agent's own speaker (outputs=None: its default).
    open_mic = bool(start.get('open_mic'))
    if faced and not open_mic: loop.body = Body(emit)
    engine = loop.engine = GemmaMultimodal(
        on_event=loop.on_event, play=not open_mic, half_duplex=not open_mic, listens_while_thinking=True,
        outputs=[] if open_mic else [loop.body] if loop.body else None,
        output_mode='speech', extra_prompt=start.get('persona', ''),
        config=MultimodalConfig(output_mode='speech', wake_required=False,
                                stt_model=start.get('hearing', 'large-v3-turbo'), session_key=start.get('session', 'avatar'),
                                **({'kokoro_voice': start['voice']} if start.get('voice') else {})),   # the character's own voice, when the host names one
        runtime_config={**({} if key == DEFAULT_LLM else {'model_path': str(LLMS[key].path)}),   # no path: the agent loads its default
                        **{k: start[k] for k in ('max_tokens', 'temperature', 'timeout_s') if k in start}})
    engine.load()                                            # on this, the main thread: the MLX voice crashes at exit otherwise
    if loop.examples: engine.prime(loop.examples)            # the conversation it starts in; each later one is shown them as it opens
    loop.extra = {'tool_list': 'every tool' if os.environ.get('JAEGER_FULL_TOOLS') == '1' or os.environ['JAEGER_TOOLSET_SCOPING'] != '1' else 'short list',
                  'load_s': round(time.time() - began, 1), 'models': {c.path.stem: str(c.path) for c in LLMS.values() if c.present}}
    emit({'ready': True, 'schema': schema(), 'model': name, 'report': loop.report()})

    def read():
        for line in sys.stdin:
            try: loop.receive(json.loads(line))
            except Exception as exc: print('bad line from the app:', exc, file=sys.stderr)
        loop.receive(None)

    threading.Thread(target=read, daemon=True).start()
    loop.run()
    engine.close()


if __name__ == '__main__':
    main()
