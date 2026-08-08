"""Named `BeamSearchConfig` presets ("search modes") - depth/width/top_k tradeoffs
between decision latency and search quality, selectable by name from `runner`'s
entry points/CLIs without hand-constructing a `BeamSearchConfig`.

`resolve_search_mode()` is the single entry point: a mode name, an already-built
`BeamSearchConfig`, or `None` (default mode) in, optionally with `max_depth`
overridden on top regardless of which mode it came from.
"""

from __future__ import annotations

from dataclasses import replace

from sts2_training.decision.beam_search import BeamSearchConfig

__all__ = ["DEFAULT_SEARCH_MODE", "SEARCH_MODES", "resolve_search_mode"]

# Kept small - add a mode only for a genuinely different tradeoff, not every
# parameter combination (use an explicit BeamSearchConfig for anything bespoke).
SEARCH_MODES: dict[str, BeamSearchConfig] = {
    "shallow": BeamSearchConfig(max_depth=1, beam_width=4, top_k_actions=3),  # throughput over quality
    "standard": BeamSearchConfig(max_depth=2, beam_width=8, top_k_actions=4),  # BeamSearchConfig's own defaults
    "deep": BeamSearchConfig(max_depth=4, beam_width=8, top_k_actions=4),  # more lookahead, same width
    "wide": BeamSearchConfig(max_depth=2, beam_width=16, top_k_actions=6),  # more candidates, same depth
}

DEFAULT_SEARCH_MODE = "standard"


def resolve_search_mode(
    mode: str | BeamSearchConfig | None = None,
    *,
    max_depth: int | None = None,
) -> BeamSearchConfig:
    """`None` -> default mode; `str` -> looked up in `SEARCH_MODES` (`ValueError` if
    unknown); `BeamSearchConfig` -> used as-is. `max_depth`, if given, overrides just
    that field on the resolved config.
    """
    if mode is None:
        mode = DEFAULT_SEARCH_MODE
    if isinstance(mode, BeamSearchConfig):
        config = mode
    else:
        try:
            config = SEARCH_MODES[mode]
        except KeyError:
            raise ValueError(
                f"unknown search mode {mode!r}; choose one of {sorted(SEARCH_MODES)} "
                "or pass a BeamSearchConfig directly"
            ) from None
    if max_depth is None:
        return config
    return replace(config, max_depth=max_depth)
