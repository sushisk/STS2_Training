"""Policy + Beam Search + Value Function decision-making, layered on
`sts2_training.api.AsyncTrainingApiClient` (DTO v0.7). See `how_to_use.md`.
"""

from sts2_training.decision.beam_search import (
    BeamSearchConfig,
    BeamSearchEngine,
    BeamSearchResult,
)
from sts2_training.decision.engine import CombatDecisionEngine, DecisionOutcome
from sts2_training.decision.oracle_log import (
    ORACLE_RECORD_SCHEMA_VERSION,
    OracleJsonlWriter,
    oracle_collection_record,
)
from sts2_training.decision.oracle_search import (
    BudgetedOracleCollector,
    OracleCollectionConfig,
    OracleCollectionResult,
    OracleRngOutcome,
    OracleTargetMetadata,
    OracleTargets,
    RootActionOracleTarget,
    StableNodeOracleTarget,
    build_oracle_targets,
)
from sts2_training.decision.policy import ActionCandidate, PolicyModel, PriorHeuristicPolicy
from sts2_training.decision.search_modes import DEFAULT_SEARCH_MODE, SEARCH_MODES, resolve_search_mode
from sts2_training.decision.search_trace import (
    InMemorySearchTraceCollector,
    PolicyCandidateTrace,
    PolicyProposalTrace,
    ResolvedNodeTrace,
    SearchTraceCollector,
    SearchTraceEnd,
    SearchTraceStart,
    StablePruneNodeTrace,
    StablePruneTrace,
)
from sts2_training.decision.stable_pruner import (
    StableFrontierPruner,
    StablePruneContext,
    ValueTopKPruner,
)
from sts2_training.decision.value import HeuristicValueFunction, ValueModel

__all__ = [
    "ActionCandidate",
    "BeamSearchConfig",
    "BeamSearchEngine",
    "BeamSearchResult",
    "BudgetedOracleCollector",
    "CombatDecisionEngine",
    "DecisionOutcome",
    "DEFAULT_SEARCH_MODE",
    "HeuristicValueFunction",
    "InMemorySearchTraceCollector",
    "ORACLE_RECORD_SCHEMA_VERSION",
    "OracleCollectionConfig",
    "OracleCollectionResult",
    "OracleJsonlWriter",
    "OracleRngOutcome",
    "OracleTargetMetadata",
    "OracleTargets",
    "PolicyCandidateTrace",
    "PolicyModel",
    "PolicyProposalTrace",
    "PriorHeuristicPolicy",
    "ResolvedNodeTrace",
    "RootActionOracleTarget",
    "SEARCH_MODES",
    "SearchTraceCollector",
    "SearchTraceEnd",
    "SearchTraceStart",
    "StableFrontierPruner",
    "StableNodeOracleTarget",
    "StablePruneContext",
    "StablePruneNodeTrace",
    "StablePruneTrace",
    "ValueModel",
    "ValueTopKPruner",
    "build_oracle_targets",
    "oracle_collection_record",
    "resolve_search_mode",
]
