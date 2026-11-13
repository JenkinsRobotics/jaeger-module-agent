"""The agent's voice and hearing come from the Jaeger modules.

The agent has no speech loader of its own and no fallback copy.  It asks
JaegerOS which installed module fills a slot (``tts``, ``stt``) and opens
that, so a faster or different module reaches the agent by being installed.
"""
from __future__ import annotations

import threading
from typing import Any

# ponytail: one lock for every hearing pass in the process (voice loop,
# preview, the listen tool). Per-model locks if two hearing models ever run
# side by side.
HEARING_LOCK = threading.Lock()

_shared: dict[tuple, Any] = {}
_shared_lock = threading.Lock()


def open_part(slot: str, opener: str, *, shared: bool = False,
              **options: Any) -> tuple[Any, dict[str, str]]:
    """Open whatever fills ``slot``; return it and where it came from.

    ``shared`` hands every caller with the same options the same part, so the
    process holds one copy of the model.
    """
    from jaeger_os.core.modules import resolve_slot_module

    package = resolve_slot_module(slot)
    open_ = getattr(package, opener, None)
    if open_ is None:
        raise RuntimeError(
            f"Nothing usable fills the '{slot}' slot: no installed Jaeger module "
            f"offers {opener}() (none installed, too old, or it failed to import). "
            "Install jaeger-agent[multimodal].")
    if shared:
        key = (open_, tuple(sorted(options.items())))
        with _shared_lock:
            if key not in _shared:
                _shared[key] = open_(**options)
            part = _shared[key]
    else:
        part = open_(**options)
    return part, {"module": package.__name__,
                  "version": getattr(package, "__version__", "unknown"),
                  "compute": getattr(part, "compute", "unknown")}
