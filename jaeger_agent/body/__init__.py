"""The agent in a body: a face and moves from the `animation` slot, with a host's speaker, microphone and display.

    from jaeger_agent.body import AgentHost, Device
    device = Device(AgentHost(), bundle, speaker=speaker)

The agent opens the body the way it opens its voice and its hearing: whatever
Jaeger module fills the slot (core/parts.open_part). What is the agent's own
lives here: the avatar and scene tools (tools.py), what a character in a body
is told (persona.py), the voice loop started for a body (loop.py), a host's
side of it (host.py) and the two joined (device.py). With nothing in the slot
the agent is a voice with no face.

Moved from JaegerAvatar's jaeger_avatar/embedded on November 22, 2026, and not
yet run from here: see jaeger-module-animation
dev/docs/20261122_avatar_engine_move.md.
"""
from .device import Device, open_body
from .host import AgentHost

__all__ = ["AgentHost", "Device", "open_body"]
