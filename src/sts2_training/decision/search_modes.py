"""Named `BeamSearchConfig` presets ("search modes") - depth/width/top_k tradeoffs
between decision latency and search quality, selectable by name from the top level
(`runner` entry points / their CLIs) without hand-constructing a `BeamSearchConfig`.

`resolve_search_mode()` is the single entry point every caller should go through: it
accepts a mode name, an already-built `BeamSearchConfig` (for full manual control), or
`None` (the default mode), and optionally overrides just `max_depth` on top - the one
knob explicitly called out as needing direct top-level control regardless of mode.
"""

from __future__ import annotations

from dataclasses import replace

from sts2_training.decision.beam_search import BeamSearchConfig

__all__ = ["DEFAULT_SEARCH_MODE", "SEARCH_MODES", "resolve_search_mode"]

# Kept intentionally small - add a new mode only once a real use case needs a
# genuinely different tradeoff point, not for every parameter combination someone
# might want (use an explicit BeamSearchConfig for anything more bespoke).
SEARCH_MODES: dict[str, BeamSearchConfig] = {
    # Cheapest: minimal lookahead, narrow beam. Fast per-decision data collection
    # where search quality matters less than throughput.
    "shallow": BeamSearchConfig(max_depth=1, beam_width=4, top_k_actions=3),
    # BeamSearchConfig's own defaults - a reasonable default lookahead/width balance.
    "standard": BeamSearchConfig(max_depth=2, beam_width=8, top_k_actions=4),
    # More lookahead at the same width - for evaluation/debugging runs where decision
    # latency matters less than finding a genuinely better line.
    "deep": BeamSearchConfig(max_depth=4, beam_width=8, top_k_actions=4),
    # Same depth as standard but a wider beam/candidate set - trades latency for
    # considering more distinct root actions before committing to one.
    "wide": BeamSearchConfig(max_depth=2, beam_width=16, top_k_actions=6),
}

DEFAULT_SEARCH_MODE = "standard"


def resolve_search_mode(
    mode: str | BeamSearchConfig | None = None,
    *,
    max_depth: int | None = None,
) -> BeamSearchConfig:
    """Resolves `mode` to a `BeamSearchConfig`:

    - `None` -> `SEARCH_MODES[DEFAULT_SEARCH_MODE]`.
    - a `str` -> looks it up in `SEARCH_MODES` (raises `ValueError` if unknown).
    - a `BeamSearchConfig` -> used as-is, for full manual control alongside the
      preset system rather than instead of it.

    `max_depth`, if given, overrides just that field on top of whichever config was
    resolved (via `dataclasses.replace`), independent of which mode/config it came
    from - this is the direct depth control requested at the top level, without
    needing a whole new mode per depth value.
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
