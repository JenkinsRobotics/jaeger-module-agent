"""A scene's run, beat by beat: did each line get voiced, each move made, each of the other person's lines waited for
and heard? Read from a host's event log (one JSON line per event the agent sent, as Mochi's --events writes).

    python -m jaeger_agent.bench.scene_report SCRIPT.scene RUN1.jsonl [RUN2.jsonl ...]

Prints each beat with how many runs it passed, and how many whole runs passed. A run passes when every beat does.
Their lines are no test of words (any words move the scene on), so 'heard' passes on being heard; how much of the
written line was said (match) is shown beside it.
"""
import json
import sys
from pathlib import Path

from jaeger_agent.core import scene


def expected(script):
    """The scene as beats to check: ('say', line, [cues]) and ('listen', their line), in order."""
    out = []
    for line in scene.lines(script):
        if isinstance(line, scene.Listen): out.append(('listen', str(line), []))
        elif str(line): out.append(('say', str(line), list(line.cues)))
        elif line.cues and out: out[-1][2].extend(line.cues)   # moves in a quiet stretch belong to the line before it
    return out


def _made(cue, events):
    """Whether the body took every move in this cue: for each, the avatar tool's report of it answered ok. (An action
    may be made as the body's own kind of move, a reaction say, so any avatar op with that value counts.)"""
    def took(value):
        return any(e.get('event') == 'tool' and str(e.get('text', '')).startswith('avatar ') and (e.get('data') or {}).get('value') == value
                   and ((e.get('data') or {}).get('result') or {}).get('ok') for e in events)
    return all(took(v) for k, v in cue.items() if k not in ('on', 'hold'))


def report(script, events):
    """[{beat, ok, note}] for one run."""
    beats, out = expected(script), []
    # Each part of the scene (her lines up to their next one) is looked for between the last 'heard' and the next.
    heard = [i for i, e in enumerate(events) if e.get('event') == 'scene' and e.get('text') == 'heard']
    edges = [-1] + heard + [len(events)]
    part = 0
    for kind, text, cues in beats:
        window = events[edges[part] + 1:edges[part + 1] + 1] if part + 1 < len(edges) else []
        if kind == 'say':
            said = [e for e in window if e.get('event') == 'audio' and (e.get('text') or '').strip() == text.strip()]
            out.append({'beat': f'say {text}', 'ok': bool(said), 'note': f"{said[0].get('seconds', '?')} s" if said else 'not voiced'})
            for cue in cues:
                made = _made(cue, window)
                out.append({'beat': f'  move {cue}', 'ok': made, 'note': '' if made else 'not made'})
        else:
            waited = [e for e in window if e.get('event') == 'scene' and e.get('text') == 'listen']
            out.append({'beat': f'listen for “{text}”', 'ok': bool(waited), 'note': '' if waited else 'never waited'})
            got = events[heard[part]] if part < len(heard) else None
            data = (got or {}).get('data') or {}
            out.append({'beat': '  their line heard', 'ok': bool(got), 'note': f"said “{data.get('said')}”, match {data.get('match')}" if got else 'not heard'})
            part += 1
    return out


def main(argv=sys.argv[1:]):
    script, runs = Path(argv[0]).read_text(), [[json.loads(x) for x in Path(f).read_text().splitlines() if x.strip()] for f in argv[1:]]
    results = [report(script, r) for r in runs]
    for i, beat in enumerate(results[0] if results else []):
        passed = sum(r[i]['ok'] for r in results)
        notes = '; '.join(sorted({r[i]['note'] for r in results if r[i]['note']}))[:160]
        print(f"{'✓' if passed == len(results) else '✗'} {passed}/{len(results)}  {beat['beat']}" + (f'   ({notes})' if notes else ''))
    whole = sum(all(b['ok'] for b in r) for r in results)
    print(f'\nWhole runs passed: {whole}/{len(results)}')
    return whole, len(results)


if __name__ == '__main__':
    main()
