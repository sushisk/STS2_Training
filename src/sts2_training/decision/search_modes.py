"""Named `BeamSearchConfig` presets ("search modes") - depth/width/top_k tradeoffs
between decision latency and search quality, selectable by name from `runner`'s
entry points/CLIs without hand-constructing a `BeamSearchConfig`.

`resolve_search_mode()` is the single entry point: a mode name, an already-built
`BeamSearchConfig`, or `None` (default mode) in, optionally with `max_depth`
overridden on top regardless of which mode it came from. Named presets are templates:
resolving one returns an isolated copy so mutating an engine's config cannot corrupt the
global preset used by later runs. The public `SEARCH_MODES` registry is also read-only
and returns a fresh config copy for every lookup, so introspection cannot mutate the
private templates either.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace

from sts2_training.decision.beam_search import BeamSearchConfig

__all__ = ["DEFAULT_SEARCH_MODE", "SEARCH_MODES", "resolve_search_mode"]


def _copy_config(config: BeamSearchConfig, *, max_depth: int | None = None) -> BeamSearchConfig:
    simulation_options = (
        None if config.simulation_options is None else dict(config.simulation_options)
    )
    return replace(
        config,
        max_depth=config.max_depth if max_depth is None else max_depth,
        simulation_options=simulation_options,
    )


# Kept small - add a mode only for a genuinely different tradeoff, not every
# parameter combination (use an explicit BeamSearchConfig for anything bespoke).
_SEARCH_MODE_TEMPLATES: dict[str, BeamSearchConfig] = {
    "shallow": BeamSearchConfig(max_depth=1, beam_width=4, top_k_actions=3),  # throughput over quality
    "standard": BeamSearchConfig(),
    "deep": BeamSearchConfig(max_depth=4, beam_width=8, top_k_actions=4),  # more lookahead, same width
    "wide": BeamSearchConfig(max_depth=2, beam_width=16, top_k_actions=6),  # more candidates, same depth
}


class _SearchModeRegistry(Mapping[str, BeamSearchConfig]):
    """Read-only public view that never exposes the mutable preset templates."""

    def __getitem__(self, name: str) -> BeamSearchConfig:
        return _copy_config(_SEARCH_MODE_TEMPLATES[name])

    def __iter__(self) -> Iterator[str]:
        return iter(_SEARCH_MODE_TEMPLATES)

    def __len__(self) -> int:
        return len(_SEARCH_MODE_TEMPLATES)


SEARCH_MODES: Mapping[str, BeamSearchConfig] = _SearchModeRegistry()
DEFAULT_SEARCH_MODE = "standard"


def resolve_search_mode(
    mode: str | BeamSearchConfig | None = None,
    *,
    max_depth: int | None = None,
) -> BeamSearchConfig:
    """Resolve a named preset or explicit config.

    `None` selects `DEFAULT_SEARCH_MODE`. Named presets are copied before returning so
    callers may safely mutate the resolved config. An explicit `BeamSearchConfig` is
    used as-is when no override is requested; supplying `max_depth` returns a copied
    config with only that field changed.
    """
    if mode is None:
        mode = DEFAULT_SEARCH_MODE
    if isinstance(mode, BeamSearchConfig):
        if max_depth is None:
            return mode
        return _copy_config(mode, max_depth=max_depth)

    try:
        config = _SEARCH_MODE_TEMPLATES[mode]
    except KeyError:
        raise ValueError(
            f"unknown search mode {mode!r}; choose one of {sorted(SEARCH_MODES)} "
            "or pass a BeamSearchConfig directly"
        ) from None
    return _copy_config(config, max_depth=max_depth)
