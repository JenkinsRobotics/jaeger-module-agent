# ruff: noqa: E701, E702
"""The agent's hands on a body: the avatar tool and the scene tool.

The avatar tool keeps no list of a body's moves. It passes what the agent asked for
to the body (send: avatar-control/v1) and hands a refusal back with what this
body offers, so the agent can pick again. The one list it has is
MEANT: words a small model says for a feeling, tried only after the body has
refused the word, and only as names the body itself offers. send is whatever
reaches the body: the body opened from the `animation` slot (its control), or a
host's control link. Register it with the agent's own tool registry:

    register_tool_from_function(avatar_tool(send), name='avatar')

Moved from JaegerAvatar (jaeger_avatar/embedded/agent_tools.py) on November 22, 2026. Until that copy is removed, a fix
made there is made here too. The two transports that file also had are the body's own business and went to the
animation module (jaeger_animation/avatar).
"""

HOLD_MS = 10000   # ponytail: one hold time for a feeling, a move or a look; per call when a body needs it
LOOK = {'left': (-1, 0), 'right': (1, 0), 'up': (0, 1), 'down': (0, -1), 'center': (0, 0)}
# Feelings Gemma 4 E4B asked for that no body has, and the name it meant (4 of 39 turns, 2026-11-21).
MEANT = {'happy': 'joy', 'joyful': 'joy', 'cheerful': 'joy', 'glad': 'joy', 'loving': 'love', 'curious': 'thinking', 'thoughtful': 'thinking',
         'afraid': 'scared', 'mad': 'angry', 'tired': 'sleepy', 'embarrassed': 'shy', 'proud': 'smug', 'shocked': 'surprised', 'calm': 'neutral'}


def meant(emotion):
    """Other names to try for a feeling the body refused: the word it stands for, then each word of a run-together
    pair ("neutral joy", read off a list as one name), the last first."""
    word = emotion.strip().lower()
    return [n for n in dict.fromkeys([MEANT.get(word), *reversed(word.split())]) if n and n != word]


def fitted(call, offers):
    """A scene's call (the avatar tool's emotion, action, look, hold) as this body can make it, and what was left out,
    each with what the body does offer, in words for whoever wrote the scene."""
    left = []
    moves = [*offers.get('actions', []), *offers.get('reactions', [])]
    named = str(call.get('action', '')).strip()
    for slot, known in (('emotion', offers.get('emotions', [])), ('look', LOOK)):   # a move a scene named bare ("joy", "left") arrives as an action
        if named and named not in moves and named.lower() in known and slot not in call:
            call = {slot if k == 'action' else k: v for k, v in call.items()}
    emotion, action, look = (str(call.get(k, '')).strip() for k in ('emotion', 'action', 'look'))
    swap = {}
    if emotion and emotion not in offers.get('emotions', []):
        near = next((n for n in meant(emotion) if n in offers.get('emotions', [])), None)
        if near: swap['emotion'] = near
        else: left.append(('emotion', f"emotion {emotion} (it shows: {', '.join(offers.get('emotions', []))})"))
    if action and action not in moves:
        left.append(('action', f"action {action} (it does: {', '.join(moves)})"))
    if look and not all(w in LOOK for w in look.lower().replace('-', ' ').split()):
        left.append(('look', f"look {look} (left, right, up, down or center)"))
    left += [(name, f'{name} (a move is an emotion, an action or a look)') for name in call if name not in ('emotion', 'action', 'look', 'hold')]
    gone = {name for name, _ in left}
    return {k: swap.get(k, v) for k, v in call.items() if k not in gone}, [why for _, why in left]


def avatar_tool(send):
    """The tool function. send(op, value, ttl_ms) returns {'ok': True} or {'ok': False, 'error', 'offers'}."""
    def avatar(emotion: str = '', action: str = '', look: str = '') -> dict:
        """Move your avatar body: show a feeling, make a move, or look somewhere.

        emotion: a feeling to show on the face; it holds for about ten seconds.
        action: a gesture or reaction to perform once.
        look: where to look as the viewer sees it: left, right, up, down or center; two may combine ("up left").
        Use only the names this body offers. A refusal lists them.
        """
        did, errors, offers = [], [], {}
        emotion, action, look = ('' if str(v).strip().lower() in ('none', 'null') else v for v in (emotion, action, look))   # a small model fills a slot it does not want

        def tell(label, *tries):
            first = None
            for op, value in tries:
                answer = send(op, value, HOLD_MS)
                if answer.get('ok'):
                    did.append(label if first is None or op != 'emotion' else f'emotion {value} (for {emotion})')   # the body's own word, said back
                    return
                first = first or answer
            errors.append(f"{label}: {first.get('error', 'refused')}")
            offers.update(first.get('offers') or {})

        if emotion:
            tell(f'emotion {emotion}', ('emotion', emotion), *(('emotion', near) for near in meant(emotion)))
        if action:
            tell(f'action {action}', ('act', action), ('reaction', action))
        if look:
            words = look.lower().replace('-', ' ').split()
            if words and all(w in LOOK for w in words):
                tell(f'look {look}', ('gaze', [max(-1, min(1, sum(LOOK[w][i] for w in words))) for i in (0, 1)]))
            else:
                errors.append(f'look {look}: use ' + ', '.join(LOOK))
        if not (did or errors):
            errors.append('Nothing asked: give an emotion, an action or a look')
        out = {'ok': not errors, 'did': did}
        if errors:
            out['errors'] = errors
        if offers:
            out['offers'] = offers
        return out
    return avatar


def scene_tool(stage):
    """The tool function for a scene the agent writes itself. stage(script) hands the script to the agent's own
    scene performer, to play once this reply's words are spoken; it raises ValueError naming a step it cannot read,
    and returns what was left out because this body cannot do it (see fitted).
    Registered as a performance, the agent ends the turn on it: the scene is the reply, and a refused script
    goes back to the model to fix.

        register_tool_from_function(scene_tool(engine.stage), name='scene', side_effect='performance')
    """
    def scene(script: str) -> dict:
        """Perform a scripted scene with your avatar body: lines you speak, pauses, and moves timed to them. Use it when asked to act something out, or to do things in a set order or for a set time. The scene is your reply.

        script: steps in order, split by " | ".
        say <line>: speak a line. wait <seconds>: stay quiet.
        emotion <name>, action <name>, look <left, right, up, down or center>: a move; it happens as the next line or wait starts.
        hold <seconds>: keep the moves before it that long before going on.
        on <word>: put the moves before it on that word of the line just said.
        listen <their line>: the other person's turn in a scene for two. Write the actual words they could say, to be shown to them; the scene waits there until they have spoken, then goes on. Give them a turn after each of your lines.
        Example: "look left | hold 3 | look center | emotion joy | say So, what did you do today? | action bow | on today"
        Example for two: "emotion joy | say Welcome to my shop! What would you like? | emotion listening | listen I would like some bread | emotion excited | say Fresh bread, coming right up!"
        """
        try:
            left = stage(script)
        except ValueError as exc:
            return {'ok': False, 'error': str(exc)}
        return {'ok': True, 'note': 'The scene plays now' + (', without what this body cannot do: ' + '; '.join(left) if left else '') + '.'}
    return scene
