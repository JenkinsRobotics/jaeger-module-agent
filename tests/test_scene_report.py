"""The scene report (jaeger_agent/bench/scene_report.py): a run is checked beat by beat from the host's event log."""
from jaeger_agent.bench.scene_report import report

SCRIPT = 'emotion joy | say Hi there! | listen Hello. | say Bye now.'
GOOD = [{'event': 'audio', 'text': 'Hi there!', 'seconds': 1.0},
        {'event': 'tool', 'text': 'avatar emotion', 'data': {'value': 'joy', 'result': {'ok': True}}},
        {'event': 'scene', 'text': 'listen', 'data': {'expect': 'Hello.'}},
        {'event': 'scene', 'text': 'heard', 'data': {'said': 'Hey!', 'expect': 'Hello.', 'match': 0.0}},
        {'event': 'audio', 'text': 'Bye now.', 'seconds': 0.8}]


def test_a_run_that_did_everything_passes_every_beat_even_with_an_ad_lib():
    beats = report(SCRIPT, GOOD)
    assert [b['ok'] for b in beats] == [True] * 5, beats
    assert 'said “Hey!”' in beats[3]['note']


def test_a_missing_move_and_a_line_never_heard_fail_their_beats():
    beats = report(SCRIPT, [GOOD[0], GOOD[2]])
    assert [b['ok'] for b in beats] == [True, False, True, False, False]


def test_a_run_that_stopped_early_fails_what_it_never_reached_and_does_not_crash():
    beats = report(SCRIPT + ' | listen Again. | say Done.', GOOD[:2])
    assert [b['ok'] for b in beats] == [True, True, False, False, False, False, False, False]
