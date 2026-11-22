"""Headless core multimodal orchestrator backed by an injected AgentRuntime.

The audio, turn-taking, and vision transport are the verified VoiceLLM
framework.  The brain substitution is intentional: JaegerAgent's runtime owns
tools, skills, prompts, and conversation memory.  The optional ``llm-gemma``
node remains available only as the playground's non-agentic baseline.
"""
from __future__ import annotations

import difflib
import importlib.util
import inspect
import queue
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .audio import Speaker
from .config import MultimodalConfig
from .contracts import AgentRuntime, normalize_turn_result
from .node import resolve_runtime_factory

from . import context as ctx
from . import scene

# Their line in a two-sided scene may begin this long before the end of hers.
SCENE_GRACE_S = 0.5
from .context import (
    CONTEXT_PROMPT_ADDON,
    HOLD_MAX_S,
    SELF_ECHO_WINDOW_S,
    TurnAnnotator,
    _SELF_VOICE,
    is_echo_garble,
    is_own_voice_echo,
    is_stock_hallucination,
    llm_user_text,
)
from .events import Event, _TurnShim
from .outputs import (
    DYNAMIC_OUTPUT_PROMPT, OutputMode, StreamedSpeech, decide_output,
    multimodal_output_scope,
)
from .policy import (
    EARLY_DECODE_MS,
    FRAME_MS,
    MIN_SPEECH_MS,
    PRE_ROLL_MS,
    SAMPLE_RATE,
    SILENCE_HANGOVER_MS,
    SILERO_CLOSE_P,
    SILERO_OPEN_P,
    SYSTEM_PROMPT,
    TTS_RATE,
    SileroVad,
    _match_wake,
    agent_asked_question,
    closes_conversation,
    committed_user_text,
    followup_window,
    is_non_speech,
    turn_started_in_followup,
)

HERE = Path(__file__).resolve().parent.parent




def _load_node(relative_path: str, name: str) -> Any:
    """Load a portable hyphen-named node folder without renaming it."""
    path = HERE / relative_path
    qualified = f"jaeger_agent.nodes.{name}"
    existing = sys.modules.get(qualified)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(qualified, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load multimodal node at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


dxr = _load_node("nodes/audio-duplex-io/runtime.py", "audio_duplex_io")
strm = _load_node("nodes/stt-streaming/runtime.py", "stt_streaming")
stt_mod = _load_node("nodes/stt-whisper/runtime.py", "stt_whisper")
vis_mod = _load_node("nodes/vision-mmproj/runtime.py", "vision_mmproj")
llm_mod = _load_node("nodes/llm-gemma/runtime.py", "llm_gemma")
tts_mod = _load_node("nodes/tts-kokoro/runtime.py", "tts_kokoro")

AudioIO = dxr.AudioIO
BargeMonitor = dxr.BargeMonitor
_agc_gain = dxr._agc_gain
DriftAnchor = strm.DriftAnchor
IncrementalTranscriber = strm.IncrementalTranscriber
StreamAnnotator = strm.StreamAnnotator
StreamGovernor = strm.StreamGovernor
UtteranceAggregator = strm.UtteranceAggregator
sentence_turn = strm.sentence_turn


def _plain_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in reversed(content):
            if isinstance(block, Mapping) and block.get("type") == "text":
                return str(block.get("text") or "")
    return str(content)


class AgentRuntimeBrain:
    """Present an AgentRuntime through the playground node interface.

    ``remember`` is intentionally empty: the runtime's per-session agent owns
    the transcript.  Keeping the old eight-exchange log as well would submit
    every prior turn twice.
    """

    def __init__(self, runtime: AgentRuntime, session_key: str) -> None:
        self.runtime = runtime
        self._base_session_key = session_key
        self._generation = 0
        self.llm = None

    @property
    def session_key(self) -> str:
        suffix = f":{self._generation}" if self._generation else ""
        return f"{self._base_session_key}{suffix}"

    @property
    def history(self) -> list:
        return []

    def load(
        self,
        chat_handler=None,
        say=print,
        *,
        system_prompt: str = "",
        **_kwargs: Any,
    ) -> None:
        configure = getattr(self.runtime, "configure_vision", None)
        if chat_handler is not None and callable(configure):
            configure(chat_handler)
        say("using injected AgentRuntime (runtime-owned conversation memory)")
        warmup = getattr(self.runtime, "warmup", None)
        if callable(warmup):
            say("warming AgentRuntime brain…")
            warmup(session_key=self.session_key, system_prompt=system_prompt)

    @property
    def streams(self) -> bool:
        """Whether the runtime can hand out a reply while it is written."""
        turn = getattr(self.runtime, "run_multimodal_turn", None) or self.runtime.run_turn
        try:
            return "on_delta" in inspect.signature(turn).parameters
        except (TypeError, ValueError):
            return False

    def answer(self, content: Any, system: str, on_delta: Any = None) -> str:
        extra = {} if on_delta is None else {"on_delta": on_delta}
        multimodal = getattr(self.runtime, "run_multimodal_turn", None)
        if callable(multimodal):
            value = multimodal(
                content,
                text=_plain_text(content),
                system_prompt=system,
                session_key=self.session_key,
                **extra,
            )
        elif isinstance(content, str):
            value = self.runtime.run_turn(
                content, session_key=self.session_key, **extra)
        else:
            raise RuntimeError(
                "the injected AgentRuntime does not accept image content; "
                "implement run_multimodal_turn() or disable vision"
            )
        result = normalize_turn_result(value)
        if result.error:
            raise RuntimeError(result.error)
        return result.text.strip()

    def remember(self, _user_text: str, _reply: str) -> None:
        """No-op: AgentRuntime is the sole conversation-memory owner."""

    def note(self, user_text: str, reply: str) -> None:
        """Something said outside a turn (a performed scene) joins the
        runtime's transcript, when the runtime keeps one that can take it."""
        note = getattr(self.runtime, "note", None)
        if callable(note):
            note(self.session_key, user_text, reply)

    def prime(self, examples: Any) -> None:
        """Open this conversation with example exchanges, when the runtime
        keeps a transcript that can take them and the conversation is empty."""
        prime = getattr(self.runtime, "prime", None)
        if callable(prime):
            prime(self.session_key, examples)

    def switch(self, session_key: str) -> None:
        """Move to another conversation; the one left behind stays as it is."""
        self._base_session_key = session_key
        self._generation = 0

    def clear(self) -> None:
        clear = getattr(self.runtime, "clear_session", None)
        if callable(clear):
            clear(self.session_key)
        self._generation += 1

    def close(self) -> None:
        self.runtime.close()

class GemmaMultimodal:
    """The engine. Feed it audio, text and images; it emits Events.

    It owns no thread and no window. `push_audio` is called with whatever
    frames the caller has — a microphone callback, a WAV replay, a Qt worker —
    and every decision comes back through `on_event`. The only blocking call
    is playback, which is intrinsic to half duplex: the mic is closed for
    exactly as long as the speaker is busy.
    """

    FRAME = SAMPLE_RATE * FRAME_MS // 1000
    # 2.5s, not 1.5: the endpoint hangover alone is 1.8s, so a turn built
    # from playback echo ARRIVES ~1.8s after playback ends — the old grace
    # expired 0.3s too early and the agent answered its own reply in a loop.
    SELF_ECHO_GRACE_S = 2.5
    MAX_TOKENS = 160

    def __init__(
        self,
        on_event=None,
        want_vision: bool = False,
        extra_prompt: str = "",
        play: bool = True,
        half_duplex: bool = True,
        audio_mode: str | None = None,
        output_mode: OutputMode | None = None,
        *,
        runtime: AgentRuntime | None = None,
        runtime_factory: str | Any | None = None,
        runtime_config: Mapping[str, Any] | None = None,
        config: MultimodalConfig | Mapping[str, Any] | None = None,
        use_offline_fallback: bool = False,
        outputs: Any = None,
        listens_while_thinking: bool = False,
    ) -> None:
        output_was_explicit = output_mode is not None
        if isinstance(config, Mapping):
            output_was_explicit = output_was_explicit or "output_mode" in config
        elif isinstance(config, MultimodalConfig):
            output_was_explicit = output_was_explicit or (
                "output_mode" in config.model_fields_set
            )
        self.config = (
            config
            if isinstance(config, MultimodalConfig)
            else MultimodalConfig.model_validate(config or {})
        )
        if output_mode is not None:
            if output_mode not in {"dynamic", "speech", "text", "mirror"}:
                raise ValueError("output_mode must be dynamic|speech|text|mirror")
            self.config = self.config.model_copy(update={"output_mode": output_mode})
        chatbot_requested = bool(
            runtime_config is not None
            and runtime_config.get("tools_enabled") is False
        )
        if not output_was_explicit and (use_offline_fallback or chatbot_requested):
            # Both lanes intentionally reproduce the imported Gemma baseline.
            # They cannot select an output tool, so dynamic would silently
            # turn the historical spoken assistant into a text-only one.
            self.config = self.config.model_copy(update={"output_mode": "speech"})
        audio_mode = audio_mode or self.config.audio_mode
        self.on_event = on_event or (lambda ev: None)
        # Half is the DEFAULT and the name of the file, because it is the mode
        # that needs no echo canceller. Setting this False leaves the mic open
        # so you can interrupt, and leans on the self-echo guard below — which
        # catches the runaway loop but not a voice talking OVER the agent.
        self.half_duplex = half_duplex
        # A heard turn has a thread of its own, and the mic goes on hearing
        # while she thinks: always with an open mic; with a closed one only
        # when the caller asks, because it must then bring the sound in real
        # time (a file replayed at full speed would run ahead of the answer).
        # The mic is still closed while she SPEAKS: that is what half duplex
        # is for. Hearing while she thinks is what lets a person who was only
        # pausing go on (see _picked_up).
        self._apart = not half_duplex or listens_while_thinking
        self.want_vision = want_vision
        self.extra_prompt = extra_prompt.strip()
        self.play = play
        # Where the voice goes (see core/audio.py). The app's list wins; with
        # none given it is this machine's speaker. The echo-cancelling modes
        # play through their own duplex device, so they get no default.
        if outputs is None:
            outputs = ([Speaker(self.config.output_device)]
                       if play and audio_mode in ("plain", "structured") else [])
        self.outputs = list(outputs)
        self.llm = self.tts = self.stt = None
        # the agent is COMPOSED of nodes — each capability its own folder
        self.node_stt = stt_mod.build()
        self.node_vision = vis_mod.build()
        if runtime is not None and use_offline_fallback:
            raise ValueError("runtime and use_offline_fallback are mutually exclusive")
        if use_offline_fallback:
            self.node_llm = llm_mod.build()
        else:
            if runtime is None:
                factory = resolve_runtime_factory(
                    runtime_factory or "jaeger_agent.core.runtime:create_runtime"
                )
                agent_config = dict(runtime_config or {})
                provider = str(agent_config.get("provider") or "llama_cpp").lower()
                if provider in {"llama_cpp", "llama-cpp", "local", "local-llama"}:
                    agent_config.setdefault(
                        "model_path", self.config.fallback_llm_model_path
                    )
                runtime = factory(config=agent_config)
            self.node_llm = AgentRuntimeBrain(runtime, self.config.session_key)
        self.node_tts = tts_mod.build()
        self.vision_ready = False

        self._vad = None
        self._frames: list[np.ndarray] = []
        self._pre: deque = deque(maxlen=PRE_ROLL_MS // FRAME_MS)   # the sound just before a phrase opens
        self._in_speech = False
        self._voiced = self._silent = 0
        self._started = 0.0
        self._hang = max(1, SILENCE_HANGOVER_MS // FRAME_MS)
        self._minb = max(1, MIN_SPEECH_MS // FRAME_MS)
        # Hearing starts this many quiet frames in — before the endpoint — and
        # its answer is used only if the quiet holds (see EARLY_DECODE_MS).
        self._early_at = EARLY_DECODE_MS // FRAME_MS
        self._early = None            # (pending words, "nobody needs this" flag)
        self._decoder = ThreadPoolExecutor(1, thread_name_prefix="hearing")

        self.speaking = threading.Event()
        self._pending_image: str | None = None
        self._last_spoken = ""
        self._last_spoken_at = 0.0
        self._followup_deadline = 0.0
        self._turns = 0
        self._closing = False
        self._paused = threading.Event()
        self._force_listen_evt = threading.Event()
        self._index = 0
        self._residue = np.zeros(0, dtype=np.float32)
        #: caller-registered: empty the input queue feeding push_audio.
        #: The original MicStream drains ITS queue on pause and on resume;
        #: this engine does not own the queue, so the owner hands us the drain.
        self.drain_input = lambda: None
        # ── audio pipeline: one of the four verified input modes ──
        assert audio_mode in ("plain", "structured", "quasi", "full"), audio_mode
        self.audio_mode = audio_mode
        self.annotator = TurnAnnotator() if audio_mode != "plain" else None
        self._prev_end = {"t": None}
        self._hold = {"text": "", "deadline": 0.0, "turns": []}
        self._agc = {"peak": 0.0, "noise": 0.02}
        self._io = None
        self._ear_stop = threading.Event()
        self._ear_thread = None
        self._preview_stop = threading.Event()
        self._preview_thread = None
        self._seng = self._sgov = self._sanchor = self._sann = None
        self._agg = UtteranceAggregator() if audio_mode == "full" else None
        self._label_q: list = []
        # one brain: typed, spoken, and held turns all respond through this
        # lock, so a typed message during duplex speech serializes instead
        # of racing the LLM
        self._respond_lock = threading.Lock()
        # One turn has the floor, from its words to its end. With an open mic
        # (half_duplex=False) turns come from more than one thread; this is
        # what keeps their events, the voice and the session apart.
        self._floor = threading.Lock()
        self._staged = None               # a scene a tool staged in this turn
        self._scene = None                # a two-sided scene's remaining lines, while it waits for the other person
        self._scene_expect = ""           # the line they are expected to say
        self._scene_from = 0.0            # when her own part will have been played (time.monotonic)
        self._sounded = 0.0               # seconds of sound this turn has put out
        self._feeling = ""                # how she feels this turn: the voice is told with each line
        self._answering = ""              # what was heard and is being answered now
        self._carry: list = []            # the sound of what they had said when they went on talking: heard with what they say next
        self._joining = False             # an answer they went on over is being dropped: its sound is on its way to _carry
        self._speech_n = 0                # phrases long enough to be speech, counted as each begins
        self._mark: int | None = None     # that count when the phrase now being answered ended
        self._judged = threading.Event()  # the phrase before has been made out: phrases are, in the order said
        self._judged.set()
        self._turn_t0: float | None = None   # for the "First audio" latency
        self._mode_lock = threading.Lock()
        self._pending_agentic_mode: bool | None = None

        self._preview_decoded = 0             # frames already previewed

    # -- events ------------------------------------------------------------
    def _emit(self, kind: str, text: str = "", data=None) -> None:
        self.on_event(Event(kind, text, data))

    @contextmanager
    def _turn(self, cut: bool = False):
        """One turn has the floor from its words to its end, and says when it
        is over: ``("turn", "end")``, after everything else it emits and before
        the next turn's words. ``cut``: a reply that is on its way gives way to
        this turn first (BARGE_MODE "stop"), as it does to a voice over it."""
        if (cut and dxr.BARGE_MODE == "stop"
                and (self._floor.locked() or self._respond_lock.locked())):
            self.force_listen()
        with self._floor:
            self._sounded = 0.0
            self._feeling = ""
            try:
                yield
            except Exception as exc:
                # said before the turn is over, so whoever shows the turn can
                # show that it failed
                self._emit("status", f"error: {exc}")
                exc.told = True
                raise
            finally:
                self._emit("turn", "end")

    # -- startup -----------------------------------------------------------
    def load(self) -> None:
        """Load perception/speech nodes and connect vision to the brain."""
        t0 = time.perf_counter()

        # Hearing loads beside the voice and the model, not after them.
        hearing = self._decoder.submit(
            self.node_stt.load, self._say, model_name=self.config.stt_model)
        # ponytail: the voice stays on this thread.  An MLX voice loaded on a
        # worker crashes the process at exit, and its stdout redirect races
        # llama's.  Load it in a process of its own if start-up must shrink.
        self.node_tts.load(
            self._say,
            voice=self.config.kokoro_voice,
            language=self.config.kokoro_language,
        )
        handler = self.node_vision.load(
            self.want_vision,
            self._say,
            model_path=self.config.vision_mmproj_path,
        )
        self.vision_ready = self.node_vision.ready
        # Warm the exact tool schema used for real turns. External TTS tools
        # stay hidden because final speech belongs to this engine's own node.
        with multimodal_output_scope():
            self.node_llm.load(
                handler,
                self._say,
                model_path=self.config.fallback_llm_model_path,
                system_prompt=self._system_prompt(),
            )
        self.llm = getattr(self.node_llm, "llm", None)
        hearing.result()                    # a part that failed fails start-up
        self.tts = self.node_tts.tts
        self.stt = self.node_stt.model
        self._vad = SileroVad(self.config.silero_model_path)

        if self.audio_mode in ("quasi", "full"):
            if dxr.pyaec is None:
                raise RuntimeError(
                    "pyaec is required for quasi/full audio_mode; "
                    "install jaeger-agent[multimodal-duplex]")
            self._say(f"opening 48 kHz AEC duplex device ({self.audio_mode} audio_mode)…")
            self._io = AudioIO(self.config.input_device,
                               self.config.output_device).__enter__()
            if self.audio_mode == "full":
                self._sann = StreamAnnotator()
                self._seng = IncrementalTranscriber(self.stt)

                def _on_commit(a):
                    self._sann.label(a)
                    self._label_q.append({**(self._sann.last or {}),
                                          "_fp": self._sann.last_fp})
                self._seng.on_commit_audio = _on_commit
                self._sgov = StreamGovernor()
                self._sanchor = DriftAnchor()
            self._ear_thread = threading.Thread(
                target=(self._ear_loop_full if self.audio_mode == "full"
                        else self._ear_loop_quasi), daemon=True)
            self._ear_thread.start()
        self._emit("session", "Audio", self.audio_mode)

        # Each node owns exactly one warmup inside ``load``. In particular the
        # AgentRuntime adapter warms directly, so no synthetic turn enters its
        # transcript. Do not repeat STT/TTS inference here.
        self._say(f"primed in {(time.perf_counter() - t0) * 1000:.0f} ms")
        self._preview_thread = threading.Thread(
            target=self._preview_loop,
            daemon=True,
            name="multimodal-live-preview",
        )
        self._preview_thread.start()
        self._emit("status", "ready")
        self._emit("state", "listening")

    # -- live preview -------------------------------------------------------
    def _preview_loop(self) -> None:
        """Decode the FORMING turn every second, like the CLI's LIVE panel.

        Silence between "you started talking" and "here is what you said" is
        where a voice app feels deaf — the console UI solved it by re-decoding
        the accumulating audio once a second and showing the churn. Same model
        as the final pass, behind one lock, and nothing shown here is ever
        committed: the endpoint flush is still the only writer of record.
        """
        while not self._preview_stop.wait(1.0):
            try:
                self._preview_pass()
            except Exception:
                pass                  # a failed preview must never disturb it

    def _preview_pass(self) -> None:
        if not self._in_speech or self.speaking.is_set() or self.stt is None:
            return
        frames = list(self._frames)                    # snapshot, no lock
        if len(frames) - self._preview_decoded < 16:   # <0.5s of new audio
            return
        self._preview_decoded = len(frames)
        audio = np.concatenate(frames[-500:])          # cap the window at 15s
        # Gives way the moment the person goes quiet: the early pass is next.
        text = self.node_stt.transcribe(
            audio, abort=lambda: self._early is not None or not self._in_speech)
        if text and self._in_speech:
            self._emit("live", text + " \u258c")

    def _say(self, msg: str) -> None:
        print(f"[gemma-multimodal] {msg}", flush=True)
        self._emit("status", msg)

    # -- input 1: text -----------------------------------------------------
    def send_text(self, text: str) -> None:
        """Typed input skips the wake gate: typing IS addressing it.

        FULL DUPLEX APPLIES TO EVERY INPUT: with duplex audio pipeline this can arrive
        while the agent is speaking. A typed message is an explicit
        interruption, so it follows the same floor policy as voice —
        dxr.BARGE_MODE stop cuts playback and answers; continue lets the agent
        finish, then answers (the respond lock serializes the brain)."""
        text = text.strip()
        if not text or self._closing:
            return
        if self.speaking.is_set() and dxr.BARGE_MODE == "stop":
            self.force_listen()
        with self._turn(cut=not self.half_duplex):
            self._emit("user", text)
            if closes_conversation(text):
                self._dismiss()
                return
            if self._scene is not None:   # their line in a scene that was waiting for it
                self._scene_reply(text)
                return
            self._respond(text, input_modality="text")

    # -- input 2: vision ---------------------------------------------------
    def attach_image(self, data_uri: str) -> None:
        """Hold one image for the next turn, whether typed or spoken."""
        if not data_uri:
            return
        # Only announce the TRANSITION. A live camera calls this once a
        # second, and a status line that repeats every second is a status
        # line nobody reads.
        first = self._pending_image is None
        self._pending_image = data_uri
        if first:
            self._emit("status", "image attached to your next message")

    # -- input 3: audio ----------------------------------------------------
    def push_audio(self, frame: np.ndarray) -> None:
        """One frame in. Endpointing, then the five stages on a full turn."""
        if self._closing or self._paused.is_set():
            return
        if self.half_duplex and self.speaking.is_set():
            return                    # half duplex: the mic is closed
        # Accept ANY block size and re-chunk internally. Requiring exactly one
        # frame length made this a silent no-op for every caller whose audio
        # device disagreed — sounddevice hands out whatever blocksize it was
        # opened with, and a mismatch dropped every frame without a word. An
        # engine that ignores its only input must say so or handle it; this
        # handles it.
        if self.audio_mode in ("quasi", "full"):
            return                    # duplex audio pipeline own their AEC device
        self._tick_hold()
        block = np.asarray(frame, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return
        rms = float(np.sqrt(np.mean(block ** 2)))
        self._agc["peak"] = max(rms, self._agc["peak"] * 0.998)
        self._agc["noise"] = min(max(rms, 1e-4),
                                 self._agc["noise"] * 1.0005)
        block = block * _agc_gain(self._agc["peak"], self._agc["noise"])
        self._residue = np.concatenate([self._residue, block])
        while self._residue.size >= self.FRAME:
            piece, self._residue = (self._residue[:self.FRAME],
                                    self._residue[self.FRAME:])
            self._push_frame(piece)

    def _push_frame(self, frame: np.ndarray) -> None:
        p = self._vad.probability(frame)
        bar = SILERO_CLOSE_P if self._in_speech else SILERO_OPEN_P
        if p >= bar:
            if not self._in_speech:
                self._in_speech = True
                self._voiced = self._silent = 0
                self._frames = list(self._pre)     # the start of the first word (PRE_ROLL_MS)
                self._pre.clear()
                self._started = time.time()
                self._emit("state", "hearing you")
                self._emit("latency", "Speech detected", 0.0)
            self._frames.append(frame)
            self._voiced += 1
            self._silent = 0
            if self._voiced == self._minb:     # long enough to be speech, not a click
                self._picked_up()
            if self._early is not None:        # they kept talking: not the end
                self._early[0].cancel()
                self._early[1].set()
                self._early = None
            return
        if not self._in_speech:
            self._pre.append(frame)
            return
        self._frames.append(frame)
        self._silent += 1
        if self._silent == self._early_at and self._voiced >= self._minb:
            unwanted = threading.Event()
            carried, said = list(self._carry), self._speech_n
            # Nobody needs it if they keep talking, in this phrase or a new one.
            self._early = (self._decoder.submit(
                self.node_stt.transcribe, np.concatenate([*carried, *self._frames]),
                abort=lambda: unwanted.is_set() or self._speech_n != said),
                unwanted, len(carried))
        if self._silent >= self._hang:
            if self._voiced >= self._minb:
                self._finish_phrase()
            else:
                # Too short to be a phrase (a click, or the detector's memory
                # of a voice from before the mic was shut). Left open it would
                # take in all the quiet up to the next thing said, and hearing
                # makes words up out of a long quiet ("Thank you.").
                self._drop_phrase()

    def _drop_phrase(self) -> None:
        if self._early is not None:
            self._early[0].cancel()
            self._early[1].set()
            self._early = None
        self._in_speech = False
        self._frames = []
        self._voiced = self._silent = 0
        self._preview_decoded = 0

    def _picked_up(self) -> None:
        """Someone is speaking again. If what they said before is still on its
        way to an answer, and she has made no sound of it, they were pausing,
        not finished: a phrase ends on half a second of quiet, and a person
        thinking of their next words is quiet for longer. The turn that is
        answering it sees this (:meth:`_gone`), drops the answer, and what they
        say now joins what they said. Once she has made a sound it is a cut-in
        instead, and stays one."""
        self._speech_n += 1
        if self._answering and not self._sounded:
            self._joining = True
            self.force_listen()           # stop writing an answer nobody will hear

    def _gone(self) -> bool:
        """They have gone on talking since the phrase now being answered
        ended, and she has made no sound of the answer."""
        return (self._mark is not None and self._speech_n != self._mark
                and not self._sounded)

    def _finish_phrase(self) -> None:
        """The phrase is over: on to the five stages. A closed mic answers it
        here, before it takes another frame. An open one (or a closed one that
        listens while she thinks) gives the turn a thread of its own and goes
        on hearing, so a person can go on, or cut in."""
        audio = np.concatenate(self._frames)
        self._in_speech = False
        self._frames = []
        self._voiced = self._silent = 0
        early, self._early = self._early, None
        early = (early[0], early[2]) if early else None   # the early pass, and how much carried sound it heard
        if not self._apart:
            return self._on_turn(audio, early)
        before, judged = self._judged, threading.Event()
        self._judged = judged
        threading.Thread(target=self._heard,
                         args=(audio, early, self._speech_n, before, judged),
                         daemon=True, name="heard-turn").start()

    def _heard(self, audio: np.ndarray, early, mark: int,
               before: threading.Event, judged: threading.Event) -> None:
        try:
            before.wait(30)               # made out in the order they were said
            self._on_turn(audio, early, mark, judged)
        except Exception as exc:          # nobody is waiting on this thread
            if not getattr(exc, "told", False):
                self._emit("status", f"error: {exc}")
        finally:
            judged.set()

    def end_phrase(self) -> None:
        """Whoever feeds :meth:`push_audio` says the sound has stopped: a mic
        that was closed, or gated while the agent talks. A phrase that is
        forming ends here, instead of waiting for a quiet that will not come.
        Call it from the thread that pushes the sound."""
        if not self._in_speech:
            return
        if self._voiced >= self._minb:
            return self._finish_phrase()
        self._drop_phrase()               # too short to be a phrase

    # -- the five stages ---------------------------------------------------
    def _carry_on(self, audio: np.ndarray) -> None:
        """``audio`` is the first part of what they are still saying: it is
        kept, and heard as one with the rest."""
        self._joining = True
        if (not self.half_duplex and dxr.BARGE_MODE == "stop"
                and (self._floor.locked() or self._respond_lock.locked())):
            self.force_listen()           # as a finished phrase cuts in on an open mic
        with self._floor:                 # after whatever was said before it
            self._carry.append(audio)

    def _take_carry(self) -> list:
        if self._joining:                 # an answer they went on over is being
            with self._floor:             # dropped: its sound comes first
                pass
        carried, self._carry, self._joining = self._carry, [], False
        return carried

    def _on_turn(self, audio: np.ndarray, early=None, mark: int | None = None,
                 judged: threading.Event | None = None) -> None:
        """One phrase, from its sound to its answer: the five stages. ``mark``
        and ``judged`` come with a phrase that has a thread of its own
        (:meth:`_heard`); ``early`` is the pass that started before the
        endpoint, with how much carried sound it heard."""
        if mark is not None and self._speech_n != mark:
            # They had gone on before this was even made out: it is heard
            # with the rest, in one pass.
            if early is not None:
                early[0].cancel()
            self._carry_on(audio)
            return
        carried = self._take_carry()
        if carried:                           # they had only paused: it is one thing they said
            audio = np.concatenate([*carried, audio])
        t_end = time.perf_counter()
        self._emit("latency", "Speech detected",
                   (time.time() - self._started) * 1000)
        self._emit("latency", "End-of-turn", float(SILENCE_HANGOVER_MS))
        self._emit("state", "transcribing")

        self._preview_decoded = 0
        if early is not None and early[1] == len(carried):
            heard = early[0].result()
        else:
            # ponytail: a joined sentence whose first part came late loses the
            # early pass's head start; start it again here if that wait shows.
            if early is not None:
                early[0].cancel()
            heard = self.node_stt.transcribe(
                audio, **({} if mark is None else
                          {"abort": lambda: self._speech_n != mark}))
        if mark is not None and self._speech_n != mark:
            # They went on while this was being made out (the pass stops as
            # soon as they do): it is heard with the rest.
            self._carry_on(audio)
            return
        if not heard:
            self._emit("state", "no words — ignored")
            return
        self._emit("live", heard)

        annot = None
        if self.annotator is not None:
            shim = _TurnShim(audio, self._started,
                             int(len(audio) / SAMPLE_RATE * 1000))
            annot = self.annotator.annotate(shim, heard, self._prev_end["t"])
            self._prev_end["t"] = time.time()
            for bad, kind, note in (
                    (is_stock_hallucination(heard, annot.get("wpm")),
                     "environment", "silence artifact — dropped"),
                    (is_echo_garble(heard, annot.get("wpm")),
                     "self", "echo garble — dropped"),
                    (is_own_voice_echo(annot.get("voiceprint"),
                                       self._started),
                     "self", "own voice — dropped")):
                if bad:
                    self._emit(kind, heard)
                    self._emit("state", note)
                    self._emit("live", "")
                    return

        # The ordered fingerprint stack above is authoritative. Keep the
        # playground's text-similarity check as a final backstop for plain
        # mode and replay/shared-device callers.
        if self._is_self_echo(heard):
            self._emit("self", heard)
            self._emit("state", "heard itself — discarded")
            return

        if is_non_speech(heard):            # stage 2: environment
            self._emit("environment", heard)
            self._emit("state", "environment — ignored")
            self._emit("live", "")
            return

        in_window = (not self.config.wake_required
                     or turn_started_in_followup(self._started,
                                                 self._followup_deadline))
        hit = _match_wake(heard)            # stage 4: addressed to us?
        if hit is None and not in_window:
            self._emit("overheard", heard)
            self._emit("state", "overheard — not addressed")
            self._emit("live", "")
            return

        # A scene that waits takes their line as it was said: held, it would
        # go to the model when the hold ran out and the scene would wait on.
        if (annot is not None and hit is None and in_window
                and annot.get("holding") and self._scene is None):
            self._hold["text"] = (self._hold["text"] + " " + heard).strip()
            self._hold["deadline"] = time.time() + HOLD_MAX_S
            self._hold["turns"].append(annot)
            self._emit("state", "HOLDING — you seem mid-thought")
            self._emit("live", self._hold["text"])
            return

        command = hit[1] if hit is not None else heard
        if hit is not None:
            self._turns = 0
            self._emit("session", "Wake", f"\u2713 heard {hit[0]!r}")
            if annot is not None:
                annot["wake"] = hit[0]
        self._followup_deadline = 0.0
        if not command:                       # wake phrase and nothing else
            self._emit("live", "")
            self._emit("status", f"{hit[0]} — listening…")
            self._followup_deadline = time.time() + followup_window(turns=1)
            return

        turns = self._hold["turns"] + ([annot] if annot else [])
        if self._hold["text"]:
            command = (self._hold["text"] + " " + command).strip()
            heard = (self._hold["text"] + " " + heard).strip()   # shown whole too
            self._hold = {"text": "", "deadline": 0.0, "turns": []}
        if self._scene is not None and time.monotonic() < self._scene_from - SCENE_GRACE_S:
            # Her own part of the scene is still being played: her voice in
            # the mic, or someone over her, is not their line.
            self._emit("state", "said over her line in the scene — ignored")
            self._emit("live", "")
            return
        if judged is not None:
            judged.set()                      # the next phrase may be made out while this one is answered
        # An open mic: the reply this was said over gives way first.
        with self._turn(cut=not self.half_duplex):
            self._emit("user", committed_user_text(heard, command))
            self._emit("live", "")
            if closes_conversation(command):    # conversation is a state
                self._dismiss()
            elif self._scene is not None:       # their line in a scene that was waiting for it
                self._scene_reply(command)
            else:
                self._answer_heard(command, mark, t_end, turns, audio)

    def _answer_heard(self, command: str, mark: int | None, t_end: float | None,
                      turns: list, audio: np.ndarray | None = None) -> None:
        """Answer what was heard. Until her first sound they may go on
        (:meth:`_picked_up`); then no answer is given, and its sound is kept
        to be heard as one with what they say next (a held thought, which has
        no sound left, goes back to being held)."""
        self._mark, self._answering = mark, command
        try:
            self._respond(command, t_end, turns=turns)
        finally:
            if self._gone():
                if audio is not None:
                    self._carry.append(audio)
                    self._joining = True
                else:
                    self._hold = {"text": command, "turns": list(turns),
                                  "deadline": time.time() + HOLD_MAX_S}
            self._mark, self._answering = None, ""

    def _respond(
        self,
        command: str,
        t_end: float | None = None,
        turns: list | None = None,
        *,
        input_modality: str = "speech",
    ) -> None:
        t_end = time.perf_counter() if t_end is None else t_end
        with self._respond_lock:
            self._respond_locked(command, t_end, turns, input_modality=input_modality)

    def _respond_locked(
        self,
        command: str,
        t_end: float,
        turns: list | None,
        *,
        input_modality: str = "speech",
    ) -> None:
        self._force_listen_evt.clear()
        if self._gone():                  # they went on before she began
            self._emit("state", "listening")
            return
        self._staged = None
        with self._mode_lock:
            enabled, self._pending_agentic_mode = self._pending_agentic_mode, None
        if enabled is not None:
            runtime = getattr(self.node_llm, "runtime", None)
            runtime.agentic_tools = enabled
            self.config = self.config.model_copy(
                update={"output_mode": "dynamic" if enabled else "speech"}
            )
        self._emit("state", "thinking")
        image, self._pending_image = self._pending_image, None
        user_text = command
        if self.audio_mode != "plain" and turns and ctx.STRUCTURED_CONTEXT:
            user_text = llm_user_text(command, turns, "turn")
        if self.config.output_mode == "dynamic":
            modality = "text" if input_modality == "text" else "speech"
            user_text = f"[input: {modality}]\n{user_text}"
        # the exact payload the agent receives — context line and all — so
        # the GUI can show what was SUBMITTED, not just what was heard
        self._emit("submitted",
                   user_text + ("  [+image]" if image is not None else ""))
        self._turn_t0 = t_end
        modality = "text" if input_modality == "text" else "speech"
        # Speak each sentence as soon as it is written — the voice runs beside
        # the model instead of waiting for the whole reply.
        sentences: queue.Queue = queue.Queue()
        speaker: threading.Thread | None = None
        stream = None
        if getattr(self.node_llm, "streams", False) and self.config.output_mode != "text":
            stream = StreamedSpeech(mode=self.config.output_mode, input_modality=modality)

        started = False

        def on_delta(piece: str) -> None:
            nonlocal speaker, started
            if not started:
                started = True
                self._emit("latency", "First token", (time.perf_counter() - t_end) * 1000)
            for sentence in stream.feed(piece):
                if speaker is None:
                    self._emit("latency", "First words", (time.perf_counter() - t_end) * 1000)
                    self._emit("state", "speaking")
                    speaker = threading.Thread(target=speak_all, daemon=True)
                    speaker.start()
                sentences.put(sentence)

        failed: list[BaseException] = []

        def speak_all() -> None:
            try:
                self._speak(iter(sentences.get, None))
            except Exception as exc:      # a voice failure must reach the turn
                failed.append(exc)

        with multimodal_output_scope():
            reply = (self._answer(user_text, image) if stream is None
                     else self._answer(user_text, image, on_delta))
        if self._force_listen_evt.is_set() or self._gone():
            if speaker is not None:
                sentences.put(None)
                speaker.join()
            self._emit("state", "listening")
            return
        decision = decide_output(
            mode=self.config.output_mode, input_modality=modality, reply=reply)
        self._remember(command, reply)
        self._emit("latency", "Reply ready", (time.perf_counter() - t_end) * 1000)
        self._emit(
            "session",
            "Output",
            "+".join(decision.channels).upper() if decision.channels else "SILENT",
        )
        self._emit(
            "assistant",
            decision.display_text,
            {
                "display": bool(decision.display_text),
                "channels": decision.channels,
                "source": decision.source,
            },
        )
        if speaker is not None:
            sentences.put(stream.rest(decision.speech_text) or None)
            sentences.put(None)
            speaker.join()
            if failed:
                raise failed[0]
            self._emit("latency", "Spoken", (time.perf_counter() - t_end) * 1000)
        elif decision.speech_text:
            self._emit("state", "speaking")
            self._speak(decision.speech_text)
            self._emit("latency", "Spoken", (time.perf_counter() - t_end) * 1000)
        else:
            self._emit("latency", "Spoken", None)
        staged, self._staged = self._staged, None
        if staged and not self._force_listen_evt.is_set():
            self._scene_part(staged)      # the scene a tool staged this turn
        self._emit("state", "listening")
        self._turns += 1
        visible_reply = decision.display_text or decision.speech_text or reply
        win = followup_window(turns=self._turns,
                              agent_asked=agent_asked_question(visible_reply))
        self._followup_deadline = time.time() + win
        self._emit("session", "Mode",
                   f"FOLLOW-UP \u2014 {win:.0f}s, no wake phrase needed")

    def _system_prompt(self) -> str:
        system = SYSTEM_PROMPT + (
            CONTEXT_PROMPT_ADDON
            if ctx.STRUCTURED_CONTEXT and self.audio_mode != "plain" else "")
        if self.extra_prompt:
            system = f"{system}\n{self.extra_prompt}"
        if self.config.output_mode == "dynamic":
            system = f"{system}{DYNAMIC_OUTPUT_PROMPT}"
        return system

    def _answer(self, user_text: str, image, on_delta=None) -> str:
        system = self._system_prompt()
        content = self.node_vision.content(user_text, image)
        if on_delta is None:
            return self.node_llm.answer(content, system)
        return self.node_llm.answer(content, system, on_delta)

    def _remember(self, user_text: str, reply: str) -> None:
        """Fallback history only; the AgentRuntime implementation is a no-op."""
        self.node_llm.remember(user_text, reply)

    @property
    def _history(self) -> list:
        return self.node_llm.history

    def _tick_hold(self) -> None:
        """An abandoned held thought eventually gets its answer (HOLD_MAX_S).
        As with a phrase (:meth:`_finish_phrase`), a closed mic answers it
        here and an open one gives the turn a thread of its own, so whoever
        brings the sound is not kept waiting for the reply."""
        if self._hold["text"] and time.time() >= self._hold["deadline"]:
            text, turns = self._hold["text"], self._hold["turns"]
            self._hold = {"text": "", "deadline": 0.0, "turns": []}
            if not self._apart:
                return self._held(text, turns)
            threading.Thread(target=self._held, args=(text, turns, True),
                             daemon=True, name="held-turn").start()

    def _held(self, text: str, turns: list, alone: bool = False) -> None:
        """Answer a held thought whose rest never came."""
        try:
            with self._turn():
                self._emit("user", text)
                self._emit("live", "")
                self._answer_heard(text, self._speech_n if self._apart else None, None, turns)
        except Exception:
            if not alone:                 # on its own thread nobody is waiting,
                raise                     # and the turn has said what failed

    def _dismiss(self) -> None:
        """Speak goodbye, then clear state and re-arm the wake gate."""
        if self._closing:
            return
        self._closing = True
        self._scene = None
        try:
            self._emit("assistant", "Goodbye.")
            self._emit("state", "speaking")
            self._speak("Goodbye.")
            self.node_llm.clear()
            self._turns = 0
            self._followup_deadline = 0.0
            self._hold = {"text": "", "deadline": 0.0, "turns": []}
            if self._agg is not None:
                self._agg.clear()
            if self._seng is not None:
                self._seng.reset()
            self._emit("session", "Mode", "WAKE \u2014 say the wake phrase")
            self._emit("session", "Wake", "\u2014")
            self._emit("state", "listening")
        finally:
            self._closing = False

    def set_paused(self, paused: bool) -> None:
        """Pause or resume input without moving turn policy into a caller."""
        if paused:
            self._paused.set()
            self._in_speech = False
            self._frames = []
            self._residue = np.zeros(0, dtype=np.float32)
            self.drain_input()
            if self._agg is not None:
                self._agg.clear()
        else:
            self.drain_input()
            self._paused.clear()
        self._emit("state", "paused" if paused else "listening")

    def set_barge_mode(self, mode: str) -> None:
        """Set the audio node's per-session floor policy."""
        if mode not in {"stop", "continue"}:
            raise ValueError("barge mode must be 'stop' or 'continue'")
        dxr.BARGE_MODE = mode
        self._emit("session", "Barge", mode)

    def set_agentic_tools(self, enabled: bool) -> None:
        """Apply a mode change atomically at the next turn, not mid-answer."""
        runtime = getattr(self.node_llm, "runtime", None)
        if runtime is None or not hasattr(runtime, "agentic_tools"):
            raise RuntimeError("runtime does not support live agentic/chatbot switching")
        with self._mode_lock:
            self._pending_agentic_mode = bool(enabled)

    def switch_session(self, session_key: str, extra_prompt: str = "", *,
                       voice: str | None = None) -> None:
        """Move to another conversation with its own persona and voice.

        For an app with several characters: a session's system prompt is fixed
        when the session is first built, so each character gets its own key —
        and keeps its own conversation when the app comes back to it. ``voice``
        is the character's voice in the loaded voice module; left out, the
        voice stays as it is.
        """
        with self._floor:                 # never under a turn that is on its way
            if voice:
                self.node_tts.set_voice(voice)
            self.extra_prompt = extra_prompt.strip()
            self._scene = None            # a scene waiting for the other person belonged to the conversation being left
            self.node_llm.switch(session_key)
            self.node_llm.load(None, self._say, system_prompt=self._system_prompt())

    def prime(self, examples: Any) -> None:
        """Open the current conversation with example exchanges, each
        ``(what was said to the agent, what it answered, the tool calls that
        answer made)``, a call being ``(name, arguments, result)``. They are
        shown once, while the conversation is still empty; call it after
        :meth:`load` and after each :meth:`switch_session`.

        For an app whose every reply should start with a tool call (an avatar
        that shows a feeling). A small model follows the form of what is
        already in the conversation far more surely than an instruction:
        told only in its prompt, Gemma 4 E4B called the tool in 4 of 12
        fresh conversations and wrote the call out as words in the other 8,
        and whichever it did first it went on doing. With one example first
        it called the tool in 12 of 12 (2026-11-21).
        """
        with self._floor:
            self.node_llm.prime(examples)

    # -- what an app needs without reaching into the nodes ------------------
    @property
    def sample_rate(self) -> int:
        """The rate of the sound in ``audio`` events and from :meth:`synth`."""
        return getattr(self.node_tts, "sample_rate", TTS_RATE)

    def synth(self, text: str, *, voice: str | None = None):
        """One line through the voice alone — no model, no turn, no events.
        Yields the voice module's chunks (sound, with ``marks`` for the lips),
        in the set voice or, for this line, in ``voice``. It waits for a turn
        that is on its way: the voice makes one sound at a time."""
        with self._floor:
            yield from self.node_tts.synth(text, **({"voice": voice} if voice else {}))

    def perform(self, script, fit=None) -> None:
        """Perform a scene (:mod:`.scene`): its lines in the agent's own voice,
        no model, each with its moves on it as cues (``audio`` events whose
        sound has ``cues`` beside its ``marks``) for whoever plays the sound to
        make on the sound's clock. A bad scene raises before a word is said.
        ``fit`` is the body's say on each call a cue will make (see
        :func:`.scene.lines`): it may leave parts out, or refuse the scene."""
        lines = scene.lines(script, fit)
        with self._turn(), self._respond_lock:
            self._force_listen_evt.clear()
            self._emit("state", "speaking")
            said = self._scene_part(lines)
            if said and not self._force_listen_evt.is_set():
                # The lines join the conversation, so she can be answered.
                getattr(self.node_llm, "note", self.node_llm.remember)(scene.DIRECTION, said)
                self._emit("assistant", said,
                           {"display": True, "channels": ("speech",), "source": "scene"})
            self._emit("state", "listening")

    def stage(self, script, fit=None) -> int:
        """From inside a turn (a tool the model called): take a scene to
        perform once this turn's reply has been spoken. Returns how many
        things it will voice; a bad scene, or one ``fit`` refuses (see
        :meth:`perform`), raises and stages nothing."""
        lines = scene.lines(script, fit)
        self._staged = lines
        return len(lines)

    def _scene_part(self, lines) -> str:
        """Voice a scene up to the other person's next line, and return what
        was said. The rest waits for them: the next thing typed or heard is
        their line, and the scene goes on from it (:meth:`_scene_reply`)."""
        mine, theirs, rest = scene.part(lines)
        if mine:
            self._speak_scene(mine)
        waits = theirs is not None and not self._force_listen_evt.is_set()   # a scene that was cut is over
        self._scene, self._scene_expect = (rest, str(theirs)) if waits else (None, "")
        if waits:
            # ponytail: her part is over once the sound this turn made has had
            # time to play, counted from now (a player starts a scene's clips
            # when the last is voiced). A "played" message from the player if
            # a slow one is seen to run past it.
            self._scene_from = time.monotonic() + (0 if self.play else self._sounded)
            self._emit("scene", "listen", {"expect": self._scene_expect})
        return " ".join(line for line in mine if line)

    def _scene_reply(self, said: str) -> None:
        """The other person has said their line in a waiting scene: go on."""
        with self._respond_lock:
            lines, expect, self._scene = self._scene, self._scene_expect, None
            self._force_listen_evt.clear()
            self._emit("scene", "heard", {"said": said, "expect": expect, "match": scene.match(expect, said)})
            self._emit("state", "speaking")
            mine = self._scene_part(lines)
            if mine:                      # both sides join the conversation
                getattr(self.node_llm, "note", self.node_llm.remember)(said, mine)
                self._emit("assistant", mine,
                           {"display": True, "channels": ("speech",), "source": "scene"})
            self._emit("state", "listening")

    def end_scene(self) -> None:
        """Leave a scene that is waiting for the other person's line (Stop).
        What is said next is ordinary conversation again."""
        self._scene = None

    def _speak_scene(self, lines) -> None:
        """Voice a scene's lines between ``scene`` "begin" and "end" events,
        so a player can hold the clips until every line is voiced."""
        self._emit("scene", "begin", {"lines": len(lines)})
        self._speak(lines)
        self._emit("scene", "end")

    def feel(self, feeling: str) -> None:
        """How she feels, for the rest of this turn. A body's tool says it when
        the body has taken the feeling, and the voice is told with every line
        after, so a voice that can act on it sounds as the face looks."""
        self._feeling = str(feeling or "")
        self._emit("feeling", self._feeling)

    def _voice(self, text: str):
        """The line from the voice, with the feeling when the voice has a
        place for one."""
        synth = self.node_tts.synth
        if self._feeling and "feeling" in inspect.signature(synth).parameters:
            return synth(text, feeling=self._feeling)
        return synth(text)

    def _sound_of(self, line):
        """A line's sound from the voice, chunk by chunk; a scene's line whole,
        with its cues on it."""
        if isinstance(line, scene.Line):
            for cue in line.cues:             # a feeling set before the line is the line's; one put on a word is a move
                if cue.get("emotion") and not cue.get("on"):
                    self.feel(cue["emotion"])
            return scene.voiced(line, self._voice, self.sample_rate)
        return self._voice(line)

    def call_example(self, name: str, args: dict) -> str:
        """One call of tool ``name`` the way the loaded model writes its calls,
        for an app's prompt; empty when the model's call form is not known."""
        adapter = getattr(getattr(self.node_llm, "runtime", None), "adapter", None)
        show = getattr(adapter, "tool_call_example", None)
        return show(name, args) if show else ""

    def report(self) -> dict:
        """What each part runs on right now: module, version, model, compute."""
        runtime = getattr(self.node_llm, "runtime", None)
        config = getattr(runtime, "config", None)
        return {
            "voice": dict(getattr(self.node_tts, "info", None) or {}),
            "hearing": dict(getattr(self.node_stt, "info", None) or {}),
            "agent": runtime.health() if hasattr(runtime, "health") else {},
            **{key: getattr(config, key)
               for key in ("ctx", "tools_enabled", "max_iterations") if hasattr(config, key)},
        }

    def force_listen(self) -> None:
        """Cut current playback/generation and return the floor to input."""
        self._force_listen_evt.set()
        runtime = getattr(self.node_llm, "runtime", None)
        interrupt = getattr(runtime, "interrupt", None)
        if callable(interrupt):
            try:
                interrupt(session_key=self.node_llm.session_key)
            except OSError as exc:
                self._emit("status", f"Agent connection lost during interruption: {exc}")
        if self._io is not None:
            self._io.interrupt_playback()
        for out in self.outputs:
            out.stop()
        self._emit("state", "listening")

    # -- duplex audio pipeline (quasi/full): engine-owned AEC device ------------------
    def _ear_loop_quasi(self) -> None:
        while not self._ear_stop.is_set():
            self._tick_hold()
            try:
                fr = self._io.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._paused.is_set():
                continue
            if self.speaking.is_set():
                continue                    # the speak watcher owns these
            f1 = fr.reshape(-1)
            self._emit("mic", data=f1)
            self._residue = np.concatenate([self._residue, f1])
            while self._residue.size >= self.FRAME:
                piece, self._residue = (self._residue[:self.FRAME],
                                        self._residue[self.FRAME:])
                self._push_frame(piece)

    def _ear_loop_full(self) -> None:
        eng, gov, anchor = self._seng, self._sgov, self._sanchor
        st = {"agg_at": 0.0, "considered": True}
        while not self._ear_stop.is_set():
            try:
                fr = self._io.q.get(timeout=0.2)
                f1 = fr.reshape(-1)
                self._emit("mic", data=f1)
                eng.feed(f1)
                gov.observe(f1)
            except queue.Empty:
                pass
            if self._paused.is_set():
                continue
            if self.speaking.is_set():
                continue
            now = time.monotonic()
            try:
                buf_s = float(eng.buffer_seconds)
            except TypeError:
                buf_s = float(eng.buffer_seconds())
            act = gov.action(now, eng.undecoded_seconds,
                             bool(eng.display_text()), buffer_s=buf_s)
            drift = False
            if act == "wait" and anchor.should_flush(eng, now):
                act, drift = "flush", True
            commits: list = []
            if act == "flush":
                commits = eng.flush()
                eng.reset_audio()
                anchor.reset()
            elif act != "wait":
                commits = eng.decode_pass()
            if act != "wait":
                gov.note_commits(len(commits), now)
                if not commits:
                    self._label_q.clear()
            quiet = (act == "wait" and self._agg.turns()
                     and not st["considered"]
                     and now - st["agg_at"] >= 1.5)
            if commits:
                st["agg_at"], st["considered"] = now, False
            for s_ in commits:
                label = (self._label_q.pop(0) if self._label_q
                         else (self._sann.last or {}))
                if (label.get("energy_db", 0.0) < -30.0
                        and is_stock_hallucination(s_, 0)):
                    continue
                fp = label.get("_fp")
                if fp is not None and is_own_voice_echo(
                        [float(x) for x in fp], time.time()):
                    continue
                self._agg.push(sentence_turn(s_, label))
                self._emit("live", self._agg.text())
            boundary = None
            wake = self._agg.wake() if commits else None
            if wake is not None:
                boundary = "addressed command"
            elif ((act == "flush" and not drift) or quiet)                     and self._agg.turns():
                boundary = "long pause"
                if quiet:
                    st["considered"] = True
            if boundary is None:
                continue
            in_window = time.time() < self._followup_deadline
            if boundary == "addressed command":
                if not in_window and len(self._agg.items) > 1:
                    self._agg.items = self._agg.items[-1:]
            elif not in_window:
                self._agg.clear()
                continue
            elif not self._agg.items[-1].get("complete"):
                continue
            utt = self._agg.text()
            turns = list(self._agg.turns())
            if wake is not None and turns:
                turns[-1]["wake"] = wake[0]
            self._agg.clear()
            self._emit("live", "")
            with self._turn():
                self._emit("user", utt)
                if closes_conversation(utt):
                    self._dismiss()
                    continue
                self._respond(utt, turns=turns)

    def _first_audio(self) -> None:
        if self._turn_t0 is not None:
            self._emit("latency", "First audio",
                       (time.perf_counter() - self._turn_t0) * 1000)
            self._turn_t0 = None

    def _speak_duplex(self, sentences) -> None:
        """03/04's continuous-watch speak on the engine-owned device: mic
        LIVE throughout, barge per dxr.BARGE_MODE, self-voice fingerprinted."""
        if self._force_listen_evt.is_set():
            return
        self._last_spoken = ""
        self._last_spoken_at = time.time()
        self.speaking.set()
        io = self._io
        io.clear_barge()
        monitor = BargeMonitor()
        played: list[np.ndarray] = []
        rate = getattr(self.node_tts, "sample_rate", TTS_RATE)
        state = {"barged": False, "stop": False}

        def _watch() -> None:
            while not state["stop"]:
                try:
                    fr = io.q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if (
                    not self._closing
                    and not self._paused.is_set()
                    and self.audio_mode == "full"
                    and self._seng is not None
                ):
                    f1 = fr.reshape(-1)
                    self._seng.feed(f1)
                    self._sgov.observe(f1)
                if (not self._closing and not self._paused.is_set()
                        and dxr.BARGE_MODE == "stop"
                        and not state["barged"]
                        and monitor.feed(fr)):
                    io.interrupt_playback()
                    state["barged"] = True

        w = threading.Thread(target=_watch, daemon=True)
        w.start()
        try:
            for text in sentences:
                if state["barged"] or self._force_listen_evt.is_set():
                    break
                self._last_spoken = f"{self._last_spoken} {text}".strip()
                for chunk24 in self._sound_of(text):
                    if state["barged"] or self._force_listen_evt.is_set():
                        break
                    self._first_audio()
                    played.append(chunk24)
                    self._sounded += len(chunk24) / rate
                    self._emit("audio", text, chunk24)
                    if self.play:
                        io.play_chunk(chunk24, rate)
                    for out in self.outputs:
                        out.play(chunk24, rate)
            while (self.play and io.is_playing() and not state["barged"]
                   and not self._force_listen_evt.is_set()):
                time.sleep(0.02)
            time.sleep(0.1)
        finally:
            state["stop"] = True
            w.join(timeout=0.5)
            if played:
                try:
                    from scipy.signal import resample_poly
                except ImportError as exc:
                    raise RuntimeError(
                        "scipy is required for duplex self-voice rejection; "
                        "install jaeger-agent"
                    ) from exc
                a16 = resample_poly(np.concatenate(played), up=SAMPLE_RATE, down=rate)
                _SELF_VOICE["fp"] = TurnAnnotator.fingerprint(
                    a16.astype(np.float32))
                _SELF_VOICE["until"] = time.time() + SELF_ECHO_WINDOW_S
            self.speaking.clear()
            self._last_spoken_at = time.time()

    # -- output 2: audio ---------------------------------------------------
    def _speak(self, text) -> None:
        """Faithful port of the shipped companion's speak() — the solved setup.

        ``text`` is the whole reply, or an iterable that hands out its
        sentences as the model finishes them (each is voiced on its own, so
        the first one is heard while the rest is still being written).

        Everything here exists because its absence was once a live bug:

        * capture closes for SYNTHESIS as well as playback (`paused_capture`
          wrapped the whole generator) — the flag goes up before Kokoro runs;
        * the input queue is DRAINED on entry and again on exit, so audio that
          leaked in around the closure is discarded, not transcribed later —
          the backlog behind the self-conversation loop;
        * chunk N plays while chunk N+1 is still generating, which is the
          difference between speech that starts in ~1s and a mute app that is
          "synthesising";
        * 120 ms after the last sample, because the room keeps saying it after
          the speaker stops.
        """
        if not text or self._force_listen_evt.is_set() or self._gone():
            return
        sentences = [text] if isinstance(text, str) else text
        if self._io is not None:
            return self._speak_duplex(sentences)
        self._last_spoken = ""
        self._last_spoken_at = time.time()
        self.speaking.set()
        if self.half_duplex:
            self.drain_input()
        started = False
        rate = getattr(self.node_tts, "sample_rate", TTS_RATE)
        try:
            for sentence in sentences:
                if self._force_listen_evt.is_set():
                    break
                self._last_spoken = f"{self._last_spoken} {sentence}".strip()
                for chunk in self._sound_of(sentence):
                    if self._force_listen_evt.is_set():
                        break
                    self._first_audio()
                    self._sounded += len(chunk) / rate
                    self._emit("audio", sentence, chunk)
                    for out in self.outputs:
                        out.play(chunk, rate)
                    started = True
            if started and self.outputs:
                for out in self.outputs:
                    out.wait()
                time.sleep(0.12)           # let the speaker and the room drain
        finally:
            if self.half_duplex:          # an open mic may be hearing someone
                self._in_speech = False
                self._frames = []
                self._pre.clear()
                self._residue = np.zeros(0, dtype=np.float32)
                self.drain_input()
            self.speaking.clear()
            self._last_spoken_at = time.time()

    def close(self) -> None:
        """Stop owned workers/devices and close the injected runtime."""
        self._ear_stop.set()
        self._preview_stop.set()
        self._decoder.shutdown(wait=False)
        if self._ear_thread is not None:
            self._ear_thread.join(timeout=1.0)
        if self._preview_thread is not None:
            self._preview_thread.join(timeout=1.0)
        if self._io is not None:
            self._io.__exit__(None, None, None)
            self._io = None
        # The mmproj MTMD context is a native child of the llama model and
        # must be released first.  llama-cpp-python's handler has no public
        # close method, so the vision node owns this lifecycle boundary.
        close_vision = getattr(self.node_vision, "close", None)
        if callable(close_vision):
            close_vision()
        close = getattr(self.node_llm, "close", None)
        if callable(close):
            close()

    def _is_self_echo(self, heard: str) -> bool:
        """Ours or theirs? Half duplex should never need this — but a file
        replay, a shared output device, or a future full-duplex caller can all
        deliver our own voice back, and answering yourself does not stop."""
        if not self._last_spoken or not heard.strip():
            return False
        if (time.time() - self._last_spoken_at > self.SELF_ECHO_GRACE_S
                and not self.speaking.is_set()):
            return False

        def norm(t: str) -> str:
            return " ".join("".join(c for c in t.lower()
                                    if c.isalnum() or c.isspace()).split())

        a, b = norm(heard), norm(self._last_spoken)
        if not a:
            return False
        return a in b or difflib.SequenceMatcher(None, a, b).ratio() >= 0.65

MultimodalAgent = GemmaMultimodal

__all__ = ["AgentRuntimeBrain", "GemmaMultimodal", "MultimodalAgent", "Event"]
