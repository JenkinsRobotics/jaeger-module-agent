# ruff: noqa: E701, E702
"""The agent in a body: what Mochi and JP01 are. A host gives a bundle, a speaker, a microphone and a display.

    device = Device(AgentHost(), bundle, speaker=speaker)     # the body comes from the `animation` slot, opened here
    device.start()                                            # the agent loads; device.ready says when
    url = device.body.serve()                                 # the display: her face as a page (or device.body.panel() for a robot's panel)
    device.text('Hello')        device.scene('say Hi. | listen')        device.hear(mic_frame)        device.stop()

The body is whatever fills the JaegerOS `animation` slot, opened the way the voice and the hearing are opened
(core/parts.open_part). The agent has no body code of its own, and with nothing in the slot it is a voice with no
face: device.body is None, the avatar and scene tools are not offered, and it speaks through its own speaker.

The agent sends each spoken line as sound with its lip marks and its cues (a scene's moves, each with its moment in
the clip). The body's player plays the sound, opens the mouth with how loud the voice is, and says when a cue's
moment has come; the agent's avatar tool then makes the move through the same body.

Turn-taking is the agent's, not the host's. Every microphone frame goes to the agent. A bare speaker and microphone
have no echo canceller, so the agent runs its own half duplex: it shuts its ears from her first word until the body
has played her last ('voiced' from the agent, 'played' back). open_mic=True is for a device that has an echo
canceller: the agent then keeps hearing, and a person can talk over her.

Moved from JaegerAvatar (Device in jaeger_avatar/embedded/device.py) on November 22, 2026. Until that copy is removed,
a fix made there is made here too. The player and the mouth went to the animation module.
"""
import base64
import threading
import time

import numpy as np

from .persona import persona, voice_only

STATE = {'thinking': 'thinking', 'hearing you': 'listening'}   # the agent's state -> the activity the character's pack gives a feeling for


def open_body(bundle, **options):
    """The body from whatever fills the `animation` slot, and where it came from ({module, version, compute});
    (None, {}) when nothing fills it."""
    from jaeger_os.core.modules import resolve_slot_module
    from jaeger_agent.core.parts import open_part
    if getattr(resolve_slot_module('animation'), 'open_body', None) is None: return None, {}
    return open_part('animation', 'open_body', bundle=bundle, **options)


class Device:
    """The agent (through an AgentHost) in a body from the animation slot, with the host's speaker and microphone."""

    def __init__(self, host, bundle=None, *, checksum=None, who=None, name='', speaker=None, session='device', hearing='large-v3-turbo',
                 workspace=None, show=print, on_event=None, delay_s=0.0, open_mic=False, **start):
        self.host, self.show, self.on_event = host, show, on_event
        self.body, self.part = open_body(bundle, checksum=checksum, speaker=speaker, delay_s=delay_s) if bundle else (None, {})
        if open_mic: start['open_mic'] = True                # this device has an echo canceller: the agent keeps hearing while she speaks
        self.turns = threading.Semaphore(0)                  # one for each turn the agent has finished
        body = self.body
        self.name, self.activity = (body.name, body.activity) if body else (name or 'Agent', {})
        character = body.character if body else {}           # who the character is and its voice, as its bundle says (a host may say otherwise)
        who = who or character.get('who') or f'You are {self.name}, an AI character.'
        if character.get('voice'): start.setdefault('voice', character['voice'])
        if not body: start['body'] = False                   # the agent then offers no avatar or scene tool, and plays its own voice
        if character.get('manner'): start.setdefault('manner', character['manner'])   # how she talks, as her bundle's dials say it: the agent says it with each turn
        self.first = {'op': 'start', 'persona': persona(who, body.name, body.offers) if body else voice_only(who), 'session': session,
                      'hearing': hearing, **({'workspace': str(workspace)} if workspace else {}), **start}

    @property
    def ready(self): return self.host.ready

    def start(self):
        if self.body:
            self.host.on_control = self.body.control         # the agent's avatar tool moves this body
            self.body.on_cue = lambda call: self._send({'op': 'cue', 'call': call})   # a scene move's moment has come: the agent makes it
        self.events = self.host.listen()
        self.host.start(self.first)
        threading.Thread(target=self._listen, daemon=True).start()

    def _send(self, data):
        try: self.host.send(data)
        except RuntimeError: pass                            # the agent is still loading, or has stopped: the host shows which

    def _listen(self):
        while (e := self.events.get()) is not None:
            kind, text = e.get('event'), e.get('text') or ''
            if self.body: self._play(e, kind, text)
            if kind == 'user': self.show(f'You: {text}')
            elif kind == 'assistant' and text: self.show(f'{self.name}: {text}')
            elif kind == 'scene' and text == 'listen': self.show(f"Your line: “{e['data']['expect']}”" if (e.get('data') or {}).get('expect') else 'Your turn')
            elif kind == 'scene' and text == 'refused': self.show(f"Scene refused: {e.get('data')}")
            elif kind == 'status' and text.startswith('error'): self.show(text)
            if kind == 'done': self.turns.release()
            if self.on_event: self.on_event(e)

    def _play(self, e, kind, text):
        """What the agent's events are to the body: her voice to its player, and what she is doing to its face."""
        player = self.body.player
        if kind == 'audio':
            player.play(np.frombuffer(base64.b64decode(e['pcm']), '<i2').astype(np.float32) / 32768, e['rate'], e.get('marks') or [], e.get('cues') or [])
        elif kind == 'scene' and text == 'begin': player.hold()   # a scene's clips are held until every line is voiced, so no gap waits on a line
        elif (kind == 'scene' and text == 'end') or kind in ('done', 'voiced'): player.release()
        if kind == 'voiced': threading.Thread(target=self._played, args=(e.get('n'),), daemon=True).start()
        if kind == 'state' and self.activity.get(STATE.get(text)): self.body.control('emotion', self.activity[STATE[text]], 4000)

    def _stop_playing(self):
        if self.body: self.body.player.stop()

    def text(self, words):
        """Typed words: what she was saying stops, as it does when a person talks over her."""
        self._stop_playing(); self._send({'op': 'text', 'text': words})

    def scene(self, script):
        self._stop_playing(); self._send({'op': 'scene', 'script': script})

    def _played(self, n):
        """The agent has voiced a part and keeps its microphone shut until the speaker has played it: say when.
        The player's clock includes delay_s, so a slow speaker is waited for."""
        self.body.player.wait(); self._send({'op': 'played', 'n': n})

    def hear(self, frame):
        """Microphone sound: 16 kHz mono floats, any length. All of it goes to the agent, which shuts its own ears while she speaks."""
        self._send({'op': 'audio', 'pcm': base64.b64encode((np.clip(frame, -1, 1) * 32767).astype('<i2').tobytes()).decode()})

    def run(self, rows, wait_s=120):
        """A run with nobody at the keyboard: rows of "say: words", "scene: steps" or "sleep: seconds", each sent once the
        turn before it has ended and the last of it has been played."""
        for row in rows:
            kind, _, value = row.partition(':')
            kind, value = kind.strip(), value.strip()
            if not value or kind.startswith('#'): continue
            if kind == 'sleep': time.sleep(float(value)); continue
            while self.turns.acquire(blocking=False): pass
            (self.text if kind == 'say' else self.scene)(value)
            self.turns.acquire(timeout=wait_s)               # the agent has finished the turn
            time.sleep(0.2)
            if self.body: self.body.player.wait(wait_s)      # and the body has played the last of it

    def stop(self):
        """Stop: the reply or the scene ends, and a scene that was waiting for its other line is over."""
        self._send({'op': 'interrupt'}); self._stop_playing()

    def close(self):
        if self.body: self.body.close()
        self.host.close()
