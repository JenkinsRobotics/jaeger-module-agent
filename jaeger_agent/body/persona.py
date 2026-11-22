# ruff: noqa: E701, E702
"""What a character in a body is told, the same for every host: who it is, the rules for replying, and what this body can do.

The rules are tuned to how Gemma 4 E4B behaves (JaegerAvatar dev/docs/reality/STATUS.md, November 21, 2026), together with the example
exchanges the worker opens each conversation with (EXAMPLES in loop.py).
"""
RULES = ("You know a lot: answer general questions from your own knowledge, right away. Do not search the web unless the user asks you "
    "to search or look something up. Use get_time for the time or date.\n"
    "If asked who you are, answer as this AI character by name, and never call yourself Gemma or mention Google. "
    "Match the length to the question: a sentence or two for small talk, three or four to explain something. Asked for a story, "
    "the short-answer rule does not apply: tell the whole story in this reply, about six short sentences, with its ending. "
    "Never only announce a story or begin one and stop.")
# Measured on gemma-4-E4B. 2026-11-11: told to call "together with your words" it moved the body on 1 to 3 of 7 turns; told to say the
# words first and call last, on 4 to 6 of 7 (first sound about 0.4 s sooner, one turn ran a web search). 2026-11-21: told to call first,
# the first turn of a fresh conversation still called in only 4 of 12, so each conversation also opens with example exchanges
# (EXAMPLES in loop.py): 24 of 24. Last in the prompt, where the small model heeds it.
BODY_RULE = ("Show how you feel with this body: begin every reply with one avatar tool call (an emotion; add an action only when the user "
    "asks you to move or a big moment calls for one), then say your words. Never write the moves as words. "
    "When asked to act something out, put on a show, or do things in a set order or for a set time, call the scene tool instead: the scene is your whole reply.")


def profile(character):
    """A character sheet (a pack profile's `character`, or the character.yaml a bundle carries) as the words the model is given."""
    prompt, identity, lore = character.get('prompt', {}), character.get('identity', {}), character.get('lore', {})
    lines = [str(prompt[k]) for k in ('custom_instructions', 'soul', 'backstory') if prompt.get(k)]
    if identity.get('role'): lines.insert(0, f"You are {identity['role']}.")
    if identity.get('voice_tone'): lines.append(f"Your voice: {identity['voice_tone']}.")
    if prompt.get('speech_patterns'): lines.append('How you talk: ' + '; '.join(map(str, prompt['speech_patterns'])) + '.')
    if lore.get('quotes'): lines.append('Things you have said: ' + ' / '.join(map(str, lore['quotes'][:4])))
    # Only the traits far from the middle change how a reply sounds; the rest would be noise in the prompt.
    style = character.get('traits', {}).get('expression', {})
    high = [k for k, v in style.items() if isinstance(v, (int, float)) and v >= .7]
    low = [k for k, v in style.items() if isinstance(v, (int, float)) and v <= .15]
    if high or low: lines.append('Style: ' + ', '.join([f'very high {k}' for k in high] + [f'almost no {k}' for k in low]) + '.')
    return '\n'.join(lines)


def who(character, name):
    """The first line of the instructions: an AI character, by name, then its profile."""
    return f"You are {name}, an AI character. {profile(character)}"


def persona(who, body='', controls=None):
    """The instructions for one character in one body. who: "You are Mochi, ..."; body: the body's name;
    controls: what it offers, {emotions, reactions, actions}, as the body itself lists them.
    How she talks right now (her manner: a pack's dials as rules) is not here. Measured on gemma-4-E4B, 2026-11-22: in
    this text several rules at once changed almost nothing, and every change meant a new conversation and the whole
    prompt read again (11 to 14 s to her next word). The engine says it with each turn (core/engine.py, tests/test_manner.py)."""
    controls = controls if isinstance(controls, dict) else {}
    def names(*keys):
        return ', '.join([n for k in keys for n in (controls.get(k) if isinstance(controls.get(k), list) else []) if isinstance(n, str) and n.isidentifier()][:16])   # one name at a time: run together, Gemma took two for one
    feelings, moves, shell = names('emotions'), names('reactions', 'actions'), str(body or '')[:60]
    return (f"{who}\n{RULES}\n" + (f"Your body right now: {shell}. " if shell else "You have a body. ")
            + (f"It can show these feelings: {feelings}. " if feelings else '') + (f"It can make these moves: {moves}. " if moves else '') + BODY_RULE)


def voice_only(who):
    """The instructions for a character with no body (nothing fills the `animation` slot): who it is and how to reply."""
    return f"{who}\n{RULES}"
