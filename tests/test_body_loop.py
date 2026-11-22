# ruff: noqa: E701, E702, E731
"""The voice loop worker a body's host starts (jaeger_agent/body/loop.py): what goes to the host for each thing the
engine does. Came with the worker from JaegerAvatar (tests/test_agent_loop.py), November 22, 2026."""
import base64
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as Event

import numpy as np

from jaeger_agent.body import loop as agent_loop


class Marked(np.ndarray):
    """A chunk as the voice module hands it over: sound, with its lip marks."""
    marks: list = []


def marked(samples, marks):
    chunk = np.asarray(samples, dtype=np.float32).view(Marked)
    chunk.marks = marks
    return chunk


class Engine:
    """Stands in for Jaeger Agent's voice loop: same calls, same events, no models."""

    sample_rate = 24000

    def __init__(self, loop):
        self.loop, self.calls, self.hears, self.voice, self.turn = loop, [], False, 'af_heart', None

    def report(self):
        return {'voice': {'module': 'jaeger_kokoro_tts', 'model': self.voice}}

    def send_text(self, text):
        self.calls.append(('text', text))
        self.loop.on_event(Event(kind='user', text=text, data=None))
        self.loop.on_event(Event(kind='audio', text='Hi.', data=np.array([0, 0.5, -2], dtype=np.float32)))
        self.loop.on_event(Event(kind='assistant', text='Hi.', data=None))
        self.loop.on_event(OVER)

    def push_audio(self, frame):
        self.calls.append(('audio', frame))
        if self.hears:                        # a heard turn runs on a thread of the engine's own: the mic stays open
            self.turn = threading.Thread(target=lambda: [self.loop.on_event(e) for e in (
                Event(kind='user', text='hello there', data=None), Event(kind='assistant', text='Hi.', data=None), OVER)])
            self.turn.start()

    def end_phrase(self):
        self.calls.append(('end',))

    def switch_session(self, key, persona, voice=None):
        self.calls.append(('persona', key, persona) + ((voice,) if voice else ()))
        self.voice = voice or self.voice

    def prime(self, examples):
        self.calls.append(('examples', list(examples)))

    def force_listen(self):
        self.calls.append(('stop',))

    def end_scene(self):
        self.calls.append(('scene over',))

    def feel(self, feeling):
        self.calls.append(('feel', feeling))


OVER = Event(kind='turn', text='end', data=None)   # the engine's word that a turn is over


def NOBODY(op, value, ttl_ms):
    """The app's answer when no body is on stage: a scene's moves cannot be checked."""
    return {'ok': False, 'error': 'No avatar attached'}


class LoopTest(unittest.TestCase):
    def setUp(self):
        self.out = []
        self.loop = agent_loop.Loop(self.out.append)
        self.engine = self.loop.engine = Engine(self.loop)

    def run_all(self, *requests):
        for r in requests:
            self.loop.receive(r)
        self.loop.receive(None)
        self.loop.run()

    def test_typed_turn_sends_voice_and_words_then_done(self):
        self.run_all({'op': 'text', 'text': 'hello'})
        self.assertEqual(self.engine.calls, [('text', 'hello')])
        kinds = [o.get('event') for o in self.out]
        self.assertEqual(kinds, ['audio', 'assistant', 'done'])   # no 'user': the page already showed what was typed
        audio = self.out[0]
        self.assertEqual((audio['text'], audio['rate']), ('Hi.', 24000))
        pcm = np.frombuffer(base64.b64decode(audio['pcm']), '<i2')
        self.assertEqual(pcm.tolist(), [0, 16383, -32767])       # float sound as 16-bit, clipped

    def test_microphone_sound_is_fed_in_and_only_a_heard_turn_ends_with_done(self):
        frame = base64.b64encode(np.array([0, 16384], '<i2').tobytes()).decode()
        self.loop.receive({'op': 'audio', 'pcm': frame})
        self.assertEqual(self.engine.calls[0][1].tolist(), [0, 0.5])
        self.assertEqual(self.out, [])
        self.engine.hears = True
        self.loop.receive({'op': 'audio', 'pcm': frame})
        self.engine.turn.join(2)
        self.assertEqual([o.get('event') for o in self.out], ['user', 'assistant', 'done'])
        self.assertEqual(self.out[0]['text'], 'hello there')

    def test_microphone_sound_does_not_wait_behind_a_turn(self):
        frame = base64.b64encode(np.array([0, 16384], '<i2').tobytes()).decode()
        self.loop.busy.set()                                      # a reply is being written
        self.loop.receive({'op': 'audio', 'pcm': frame})
        self.assertEqual([c[0] for c in self.engine.calls], ['audio'])   # heard now: this is how a person cuts in
        self.assertTrue(self.loop.work.empty())

    def test_the_app_saying_the_sound_stopped_ends_the_phrase(self):
        frame = base64.b64encode(np.array([0, 16384], '<i2').tobytes()).decode()
        self.loop.receive({'op': 'audio', 'pcm': frame, 'end': True})
        self.loop.receive({'op': 'audio', 'pcm': '', 'end': True})         # the mic went quiet with nothing left to send
        self.assertEqual([c[0] for c in self.engine.calls], ['audio', 'end', 'end'])

    def test_the_engine_waits_for_the_body_to_say_it_has_played_what_was_voiced(self):
        body = self.loop.body = agent_loop.Body(self.out.append)
        body.play(np.zeros(24000, dtype=np.float32), 24000)      # the sound itself goes out in the 'audio' events
        waited = threading.Thread(target=body.wait)
        waited.start()
        self.wait_for(1)
        self.assertEqual(self.out, [{'event': 'voiced', 'n': 1}])
        self.loop.receive({'op': 'played', 'n': 0})              # an answer to an earlier part, come late
        waited.join(0.1)
        self.assertTrue(waited.is_alive())                       # the microphone stays shut
        self.loop.receive({'op': 'played', 'n': 1})              # the speaker has finished: not queued behind a turn
        waited.join(2)
        self.assertFalse(waited.is_alive())

    def test_a_reply_that_is_cut_does_not_wait_to_be_played(self):
        body = agent_loop.Body(self.out.append)
        waited = threading.Thread(target=body.wait)
        waited.start()
        self.wait_for(1)
        body.stop()                                              # the engine's force_listen stops its outputs
        waited.join(2)
        self.assertFalse(waited.is_alive())

    def test_a_host_that_never_says_played_is_not_waited_for_forever(self):
        agent_loop.PLAYED_WAIT, before = 0.05, agent_loop.PLAYED_WAIT
        self.addCleanup(setattr, agent_loop, 'PLAYED_WAIT', before)
        body = agent_loop.Body(self.out.append)
        body.play(np.zeros(240, dtype=np.float32), 24000)        # 10 ms of sound
        began = time.time()
        body.wait()
        self.assertLess(time.time() - began, 1)

    def test_character_switch_and_interrupt(self):
        self.run_all({'op': 'persona', 'session': 's:mochi', 'persona': 'You are Mochi.'}, {'op': 'interrupt'})
        self.assertEqual([c[:3] for c in self.engine.calls if c[0] != 'examples'], [('scene over',), ('stop',), ('persona', 's:mochi', 'You are Mochi.')])   # an interrupt does not wait its turn, and ends a scene that was waiting for the other person
        self.assertEqual(self.out, [])

    def test_the_model_is_one_of_jaeger_agents_own_and_its_default_unless_another_is_chosen(self):
        llms = {'e4b': Event(path=Path('/m/small-Q4.gguf'), present=True), '26b': Event(path=Path('/m/big-Q4.gguf'), present=True)}   # stands in for the agent's own table
        pick = lambda name: agent_loop.model_choice(name, llms, 'e4b')
        self.assertEqual((pick('26b'), pick('big-Q4')), ('26b', '26b'))            # by the agent's key, or by the file's name as Settings shows it
        self.assertEqual((pick(None), pick(''), pick('a-file-the-app-once-found')), ('e4b', 'e4b', 'e4b'))
        llms['26b'].present = False
        self.assertEqual(pick('26b'), 'e4b')                                       # its file is gone: the agent's default
        llms['e4b'].present = False
        self.assertIsNone(pick('26b'))

    def test_each_conversation_opens_with_examples_of_a_reply_and_of_a_scene(self):
        # Told only in its prompt, Gemma called the avatar tool in 4 of 12 fresh conversations; shown an exchange first, in 24 of 24.
        self.run_all({'op': 'persona', 'session': 's:vita', 'persona': 'You are Vita.'})
        persona, (kind, examples) = self.engine.calls
        self.assertEqual(persona, ('persona', 's:vita', 'You are Vita.'))        # the persona goes in as the app wrote it: no model syntax added
        self.assertEqual(kind, 'examples')
        (said, reply, calls), (asked, quiet, show) = examples
        self.assertEqual([(n, a) for n, a, _ in calls], [('avatar', {'emotion': 'joy'})])   # a plain reply: one feeling, no move, then words
        self.assertTrue(said and reply)
        self.assertEqual((show[0][0], quiet), ('scene', ''))                     # asked to act something out: the scene is the whole reply
        steps = [step.split()[0] for step in show[0][1]['script'].split(' | ')]
        self.assertIn('say', steps)                                              # its script speaks,
        self.assertLessEqual(set(steps), {'say', 'wait', 'hold', 'look', 'emotion'})   # and names no action: every body can play it

    def test_typing_over_a_reply_stops_it_first(self):
        self.loop.busy.set()
        self.loop.receive({'op': 'text', 'text': 'wait'})
        self.assertEqual(self.engine.calls, [('stop',)])

    def test_the_avatar_tool_asks_the_app_and_gets_its_answer(self):
        answer = {}
        asked = threading.Thread(target=lambda: answer.update(self.loop.control('emotion', 'joy', 10000)))
        asked.start()
        for _ in range(200):
            if self.out: break
            threading.Event().wait(0.01)
        line = self.out[0]
        self.assertEqual(line['control'], {'op': 'emotion', 'value': 'joy', 'ttl_ms': 10000})
        self.loop.receive({'op': 'control_result', 'id': line['id'], 'result': {'ok': True}})
        asked.join(2)
        self.assertEqual(answer, {'ok': True})

    def answering(self, result):
        """The app's side: answer each control the worker sends with `result`."""
        def answer():
            seen = 0
            for _ in range(300):
                for line in self.out[seen:]:
                    if 'control' in line: self.loop.receive({'op': 'control_result', 'id': line['id'], 'result': result})
                seen = len(self.out)
                threading.Event().wait(0.01)
        threading.Thread(target=answer, daemon=True).start()

    def test_a_feeling_the_body_took_from_the_model_is_told_to_the_voice_too(self):
        self.answering({'ok': True})
        self.assertEqual(self.loop.felt('emotion', 'joy', 10000), {'ok': True})          # the model's own avatar call
        self.assertEqual(self.loop.felt('gaze', [1, 0], 10000), {'ok': True})            # a look is not a feeling
        self.assertEqual(self.engine.calls, [('feel', 'joy')])
        self.loop.control('emotion', 'sad', 10000)                                       # a scene's move at its moment: its line was voiced with it already
        self.assertEqual(self.engine.calls, [('feel', 'joy')])

    def test_a_feeling_the_body_refused_is_not_told_to_the_voice(self):
        self.answering({'ok': False, 'error': 'Unknown emotion'})
        self.assertFalse(self.loop.felt('emotion', 'furious', 10000)['ok'])
        self.assertEqual(self.engine.calls, [])

    def test_an_app_that_never_answers_is_a_refusal_not_a_hang(self):
        self.addCleanup(setattr, agent_loop, 'CONTROL_WAIT', agent_loop.CONTROL_WAIT)
        agent_loop.CONTROL_WAIT = 0.05
        self.assertEqual(self.loop.control('emotion', 'joy', 10000)['ok'], False)

    def test_an_avatar_call_is_reported_with_its_answer_so_the_data_flow_shows_tool_use(self):
        self.addCleanup(setattr, agent_loop, 'CONTROL_WAIT', agent_loop.CONTROL_WAIT)
        agent_loop.CONTROL_WAIT = 0.05
        answer = self.loop.control('emotion', 'joy', 10000)
        self.assertEqual(self.out[-1], {'event': 'tool', 'text': 'avatar emotion', 'data': {'value': 'joy', 'result': answer}})

    def test_a_line_is_spoken_by_the_agents_voice_without_a_turn(self):
        asked = []
        def synth(text, voice=None):
            asked.append((text, voice))
            yield marked([0, 0.5] * 250, [[0.1, 0.2, 'p']])       # half a second at 1000 Hz, lips shut at 0.1 s
            yield marked([-2] * 100, [[0.0, 0.1, 'f']])           # the next chunk times its marks from its own start
        self.engine.synth, self.engine.sample_rate = synth, 1000
        self.run_all({'op': 'say', 'id': 'r1', 'text': 'Hello there.'}, {'op': 'say', 'id': 'r2', 'text': 'Hello there.', 'voice': 'bf_emma'})
        self.assertEqual(self.engine.calls, [])                   # the model is not asked: this is the voice alone
        self.assertEqual(asked, [('Hello there.', None), ('Hello there.', 'bf_emma')])   # the set voice, or this line's own
        answer, other = self.out                                  # one answer to each asker; no 'done', since no turn happened
        self.assertEqual((answer['id'], other['id'], answer['rate']), ('r1', 'r2', 1000))
        pcm = np.frombuffer(base64.b64decode(answer['pcm']), '<i2')
        self.assertEqual((len(pcm), pcm[:2].tolist(), pcm[-1]), (600, [0, 16383], -32767))
        self.assertEqual(answer['marks'], [[0.1, 0.2, 'p'], [0.5, 0.6, 'f']])   # lip marks, timed from the start of the clip
        self.assertGreaterEqual(answer['ms'], 0)                  # how long the voice took to make the clip

    def test_lip_marks_go_out_with_the_voice_sound(self):
        self.loop.on_event(Event(kind='audio', text='Mm.', data=marked([0, 0.5], [[0.0, 0.05, 'p']])))
        self.loop.on_event(Event(kind='audio', text='Hi.', data=np.zeros(2, dtype=np.float32)))   # a voice with no timings
        self.assertEqual([o['marks'] for o in self.out], [[[0.0, 0.05, 'p']], []])

    def test_a_scene_is_performed_by_the_agent_and_its_cues_go_out_with_the_sound(self):
        cued = marked([0, 0.5], [[0.0, 0.05, 'p']])
        cued.cues = [{'at': 0.4, 'call': {'action': 'wave'}}]
        def perform(script, fit=None):
            self.engine.calls.append(('scene', script))
            self.loop.on_event(Event(kind='audio', text='Hi there.', data=cued))
            self.loop.on_event(Event(kind='assistant', text='Hi there.', data={'source': 'scene'}))
            self.loop.on_event(OVER)
        self.engine.perform = perform
        self.loop.control = NOBODY
        self.run_all({'op': 'scene', 'script': 'say Hi there. | action wave | on there'})
        self.assertEqual(self.engine.calls, [('scene', 'say Hi there. | action wave | on there')])   # the agent reads the script, not the app
        self.assertEqual([o.get('event') for o in self.out], ['audio', 'assistant', 'done'])       # a scene is a turn: the page waits for its end
        self.assertEqual(self.out[0]['cues'], [{'at': 0.4, 'call': {'action': 'wave'}}])
        self.loop.on_event(Event(kind='audio', text='Hi.', data=np.zeros(2, dtype=np.float32)))    # a reply's sound carries none
        self.assertEqual(self.out[-1]['cues'], [])

    def test_a_scene_the_agent_cannot_read_is_told_to_the_app_and_still_ends(self):
        def perform(script, fit=None): raise ValueError("Beat 1: 'goodbye' is not in the line")
        self.engine.perform = perform
        self.loop.control = NOBODY
        self.run_all({'op': 'scene', 'script': 'say Hello. | on goodbye'})
        self.assertEqual(self.out, [{'event': 'scene', 'text': 'refused', 'data': "Beat 1: 'goodbye' is not in the line"}, {'event': 'done'}])

    def test_the_models_scene_is_staged_with_the_engine_and_shown_as_tool_use(self):
        self.engine.stage = lambda script, fit=None: self.engine.calls.append(('stage', script)) or 2
        self.loop.control = NOBODY
        self.assertEqual(self.loop.stage('say Hi.\nwait 2'), [])     # nothing left out
        self.assertEqual(self.engine.calls, [('stage', 'say Hi.\nwait 2')])
        self.assertEqual(self.out, [{'event': 'tool', 'text': 'scene', 'data': {'value': 'say Hi.\nwait 2', 'result': {'ok': True, 'lines': 2, 'left_out': []}}}])
        def refuse(script, fit=None): raise ValueError('Beat 1: needs a line to say or a time to wait')
        self.engine.stage = refuse
        with self.assertRaises(ValueError): self.loop.stage('look left')
        self.assertEqual(self.out[-1]['data']['result'], {'ok': False, 'error': 'Beat 1: needs a line to say or a time to wait'})

    def test_a_scenes_moves_are_fitted_to_the_body_before_it_starts(self):
        OFFERS = {'emotions': ['neutral', 'joy'], 'reactions': [], 'actions': ['spin']}
        asked, made = [], []
        def control(op, value, ttl_ms):
            asked.append(op)
            return {'ok': True, 'offers': OFFERS}
        def engine_reads(script, fit=None):                       # the engine asks about each call the scene will make
            made.extend(fit(call) for call in ({'emotion': 'joy'}, {'emotion': 'wobbly', 'action': 'wave'}, {'emotion': 'joy', 'action': 'wave'}))
            return 1
        self.loop.control = control
        self.engine.stage = self.engine.perform = engine_reads
        # the model's scene still plays, less what the body lacks, and the model is told what that was
        self.assertEqual(self.loop.stage('...'), ['emotion wobbly (it shows: neutral, joy)', 'action wave (it does: spin)', 'action wave (it does: spin)'])
        self.assertEqual(made, [{'emotion': 'joy'}, {}, {'emotion': 'joy'}])
        self.assertEqual(asked, ['offers'])                       # the body is asked once a scene, not once a move
        self.assertEqual(self.out[-1]['data']['result']['left_out'][0], 'emotion wobbly (it shows: neutral, joy)')
        self.out.clear()
        self.run_all({'op': 'scene', 'script': 'emotion wobbly | say Hi.'})   # the director's scene is refused, to be fixed before it starts
        self.assertEqual([o.get('event') for o in self.out], ['scene', 'done'])
        self.assertEqual(self.out[0]['text'], 'refused')
        self.assertIn('This body cannot do emotion wobbly (it shows: neutral, joy)', self.out[0]['data'])

    def test_with_no_body_to_ask_a_scene_is_not_held_back(self):
        self.loop.control = NOBODY
        self.engine.stage = lambda script, fit=None: self.engine.calls.append(fit({'emotion': 'anything'})) or 3
        self.assertEqual(self.loop.stage('emotion anything | say Hi.'), [])
        self.assertEqual(self.engine.calls, [{'emotion': 'anything'}])

    def test_the_engines_scene_fence_reaches_the_app(self):
        self.loop.on_event(Event(kind='scene', text='begin', data={'lines': 3}))
        self.loop.on_event(Event(kind='scene', text='end', data=None))
        self.assertEqual(self.out, [{'event': 'scene', 'text': 'begin', 'data': {'lines': 3}}, {'event': 'scene', 'text': 'end', 'data': None}])

    def wait_for(self, n):
        for _ in range(200):
            if len(self.out) >= n: return
            threading.Event().wait(0.01)
        self.fail(f'expected {n} lines, got {self.out}')

    def test_a_cue_is_the_avatar_tools_own_call_made_at_once_and_held_as_the_scene_says(self):
        self.loop.busy.set()                                      # the agent is still voicing the scene's later lines
        self.loop.receive({'op': 'cue', 'call': {'look': 'left', 'hold': 3}})
        self.wait_for(1)                                          # it did not wait its turn behind them
        self.assertEqual(self.out[0]['control'], {'op': 'gaze', 'value': [-1, 0], 'ttl_ms': 3000})
        self.loop.receive({'op': 'control_result', 'id': self.out[0]['id'], 'result': {'ok': True}})
        self.wait_for(2)
        self.assertEqual(self.out[1], {'event': 'tool', 'text': 'avatar gaze', 'data': {'value': [-1, 0], 'result': {'ok': True}}})
        self.assertTrue(self.loop.work.empty())

    def test_a_cue_with_no_hold_holds_as_long_as_a_call_from_the_model(self):
        self.loop.receive({'op': 'cue', 'call': {'emotion': 'joy'}})
        self.wait_for(1)
        self.assertEqual(self.out[0]['control'], {'op': 'emotion', 'value': 'joy', 'ttl_ms': 10000})
        self.loop.receive({'op': 'control_result', 'id': self.out[0]['id'], 'result': {'ok': True}})
        self.wait_for(2)

    def test_a_cue_the_tool_cannot_make_is_reported_not_dropped(self):
        self.loop.receive({'op': 'cue', 'call': {'look': 'sideways'}})
        self.wait_for(1)
        self.assertEqual(self.out[0]['event'], 'tool')
        self.assertEqual(self.out[0]['data']['result']['ok'], False)
        self.assertIn('look sideways', self.out[0]['data']['result']['errors'][0])
        self.out.clear()
        self.loop.receive({'op': 'cue', 'call': {'moonwalk': 'now'}})   # a part the tool does not have
        self.wait_for(1)
        self.assertEqual(self.out[0]['data']['result']['ok'], False)

    def test_a_host_may_say_who_she_is_and_leave_the_rules_to_the_agent(self):
        """A host that cannot import the agent (another Python) names the character, the body and what it offers; what a
        character in a body is told is the agent's to write. A host that sends the whole text is taken at its word."""
        offers = {'emotions': ['neutral', 'joy'], 'reactions': ['wink'], 'actions': []}
        text = agent_loop.told({'who': 'You are Vita, an AI character.', 'body': 'Vita', 'offers': offers})
        self.assertTrue(text.startswith('You are Vita, an AI character.'))
        self.assertIn('It can show these feelings: neutral, joy.', text); self.assertIn('wink', text)
        self.assertIn('avatar tool', text)                                 # the agent's own rule for a body
        last = text.strip().splitlines()[-1]       # the body and its rule come last, and the call comes before the words
        for part in ('Your body right now: Vita.', 'begin every reply with one avatar tool call', 'call the scene tool instead'):
            self.assertIn(part, last)              # measured 2026-11-11: the model then moved the body on 42 of 42 turns
        # Gemma 4 E4B, 2026-11-21: 4 of 12 stories were told until it was asked for the whole story in the one reply (15 of 15).
        self.assertIn('tell the whole story in this reply', text); self.assertIn('Never only announce a story', text)
        odd = agent_loop.told({'who': 'You are Vita.', 'body': 'Vita', 'offers': {'emotions': ['joy', 'sad'], 'reactions': ['nod', 'not a name'], 'actions': ['wave']}})
        self.assertIn('It can show these feelings: joy, sad. ', odd)       # one name at a time: run together, Gemma took two for one
        self.assertIn('It can make these moves: nod, wave. ', odd); self.assertNotIn('not a name', odd)   # only plain names reach the prompt
        self.assertNotIn('<|', odd); self.assertNotIn('[joy]', odd)        # no model's own syntax and no inline tags: she moves with her tool
        # Her manner (the character's dials, as rules) is not in this text: it is said with each turn (tests/test_manner.py).
        self.assertEqual(agent_loop.told({'who': 'You are Vita, an AI character.', 'body': 'Vita', 'offers': offers, 'manner': 'Do not joke.'}), text)
        self.assertEqual(agent_loop.told({'persona': 'You are Mochi.', 'who': 'ignored'}), 'You are Mochi.')
        self.assertEqual(agent_loop.told({}), '')
        self.run_all({'op': 'persona', 'session': 's:vita', 'who': 'You are Vita, an AI character.', 'body': 'Vita', 'offers': offers})
        said = next(c for c in self.engine.calls if c[0] == 'persona')
        self.assertEqual(said[2], text)

    def test_her_manner_is_the_engines_to_say_with_each_turn_and_changes_at_once(self):
        """A host says how she talks right now with who she is, or on its own line whenever a dial is turned: no new
        conversation, and nothing for the model to read again."""
        self.run_all({'op': 'persona', 'session': 's:vita', 'who': 'You are Vita, an AI character.', 'manner': '  Do not joke.  '})
        self.assertEqual(self.engine.manner, 'Do not joke.')
        self.loop.receive({'op': 'manner', 'manner': 'Speak formally.'})          # a dial turned: not queued behind a reply
        self.assertEqual(self.engine.manner, 'Speak formally.')
        self.assertEqual([c[0] for c in self.engine.calls].count('persona'), 1)  # the conversation goes on
        self.loop.receive({'op': 'manner', 'manner': 'x' * 900})
        self.assertEqual(len(self.engine.manner), 600)                           # a host's words go into the prompt: bounded
        self.run_all({'op': 'persona', 'session': 's:mochi', 'who': 'You are Mochi.'})
        self.assertEqual(self.engine.manner, '')                                 # another character, who names none

    def test_a_character_brings_its_own_voice(self):
        self.loop.extra = {'load_s': 16.9}
        self.run_all({'op': 'persona', 'session': 's:mochi', 'persona': 'You are Mochi.', 'voice': 'af_bella'},
                     {'op': 'persona', 'session': 's:vita', 'persona': 'You are Vita.'})
        self.assertEqual([c for c in self.engine.calls if c[0] != 'examples'], [('persona', 's:mochi', 'You are Mochi.', 'af_bella'), ('persona', 's:vita', 'You are Vita.')])
        self.assertEqual(self.out, [{'report': {'voice': {'module': 'jaeger_kokoro_tts', 'model': 'af_bella'}, 'load_s': 16.9}}])   # the app is told what speaks now

    def test_the_worker_uses_only_the_engines_own_calls(self):
        source = Path(agent_loop.__file__).read_text()
        for inside in ('node_tts', 'node_stt', 'node_llm', '.runtime', '.adapter'):   # the agent's insides are the agent's to change
            self.assertNotIn(inside, source)

    def test_a_voice_that_fails_tells_the_asker_and_ends_no_turn(self):
        self.engine.synth = lambda text: 1 / 0
        self.run_all({'op': 'say', 'id': 'r2', 'text': 'Hello.'})
        (answer,) = self.out
        self.assertEqual(answer['id'], 'r2')
        self.assertIn('division', answer['error'])

    def test_a_failed_turn_is_reported_and_still_ends(self):
        self.engine.send_text = lambda text: 1 / 0
        self.run_all({'op': 'text', 'text': 'hello'})
        self.assertEqual([o.get('event') for o in self.out], ['status', 'done'])
        self.assertIn('division', self.out[0]['text'])

    def test_a_turn_the_engine_reported_and_ended_itself_is_not_reported_twice(self):
        def send_text(text):
            self.loop.on_event(Event(kind='status', text='error: the model stopped', data=None))
            self.loop.on_event(OVER)
            failure = RuntimeError('the model stopped'); failure.told = True
            raise failure
        self.engine.send_text = send_text
        self.run_all({'op': 'text', 'text': 'hello'})
        self.assertEqual([o.get('event') for o in self.out], ['status', 'done'])


if __name__ == '__main__':
    unittest.main()
