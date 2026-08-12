"""Named `BeamSearchConfig` presets for latency/quality tradeoffs.

Decision-phase semantics are intentionally not encoded by the mode name. The Combat
engine applies its domain scope separately; shallow/standard/deep/wide only vary search
budget knobs such as depth, width, and top-k. ``none`` keeps the standard budget shape
for compatibility but disables Beam execution through ``search_mode_uses_beam``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace

from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.decision.combat_decision import COMBAT_BEAM_ACTION_TYPES

__all__ = [
    "COMBAT_BEAM_ACTION_TYPES",
    "DEFAULT_SEARCH_MODE",
    "SEARCH_MODES",
    "resolve_search_mode",
    "search_mode_uses_beam",
]


def _copy_config(config: BeamSearchConfig, *, max_depth: int | None = None) -> BeamSearchConfig:
    simulation_options = (
        None if config.simulation_options is None else dict(config.simulation_options)
    )
    return replace(
        config,
        max_depth=config.max_depth if max_depth is None else max_depth,
        simulation_options=simulation_options,
    )


_SEARCH_MODE_TEMPLATES: dict[str, BeamSearchConfig] = {
    "none": BeamSearchConfig(),
    "shallow": BeamSearchConfig(max_depth=1, beam_width=4, top_k_actions=3),
    "standard": BeamSearchConfig(),
    "deep": BeamSearchConfig(max_depth=4, beam_width=8, top_k_actions=4),
    "wide": BeamSearchConfig(max_depth=2, beam_width=16, top_k_actions=6),
}


class _SearchModeRegistry(Mapping[str, BeamSearchConfig]):
    """Read-only public view that never exposes mutable preset templates."""

    def __getitem__(self, name: str) -> BeamSearchConfig:
        return _copy_config(_SEARCH_MODE_TEMPLATES[name])

    def __iter__(self) -> Iterator[str]:
        return iter(_SEARCH_MODE_TEMPLATES)

    def __len__(self) -> int:
        return len(_SEARCH_MODE_TEMPLATES)


SEARCH_MODES: Mapping[str, BeamSearchConfig] = _SearchModeRegistry()
DEFAULT_SEARCH_MODE = "none"


def search_mode_uses_beam(mode: str | BeamSearchConfig | None = None) -> bool:
    """Return whether a runner search mode should execute Beam Search.

    Explicit ``BeamSearchConfig`` objects remain authoritative configurations and are
    therefore considered Beam-enabled. ``None`` resolves to ``DEFAULT_SEARCH_MODE``.
    """
    if isinstance(mode, BeamSearchConfig):
        return True
    resolved_mode = DEFAULT_SEARCH_MODE if mode is None else mode
    if resolved_mode not in _SEARCH_MODE_TEMPLATES:
        raise ValueError(
            f"unknown search mode {resolved_mode!r}; choose one of {sorted(SEARCH_MODES)} "
            "or pass a BeamSearchConfig directly"
        )
    return resolved_mode != "none"


def resolve_search_mode(
    mode: str | BeamSearchConfig | None = None,
    *,
    max_depth: int | None = None,
) -> BeamSearchConfig:
    """Resolve a performance preset or explicit config.

    This function does not alter decision scope. Combat continuation support is a domain
    capability applied by `CombatDecisionEngine`, not a property of the preset name.
    Use ``search_mode_uses_beam`` for the runner-level execution toggle.
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
