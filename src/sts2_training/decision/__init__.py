"""Policy + Beam Search + Value Function decision-making, layered on
`sts2_training.api.AsyncTrainingApiClient` (DTO v0.7). See `how_to_use.md`.
"""

from sts2_training.decision.beam_search import (
    BeamSearchConfig,
    BeamSearchEngine,
    BeamSearchResult,
)
from sts2_training.decision.engine import CombatDecisionEngine, DecisionOutcome
from sts2_training.decision.learned_pruner import (
    LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION,
    LinearStableFrontierPruner,
)
from sts2_training.decision.oracle_log import (
    ORACLE_RECORD_SCHEMA_VERSION,
    OracleJsonlWriter,
    oracle_collection_record,
)
from sts2_training.decision.oracle_search import (
    BudgetedOracleCollector,
    OracleCollectionConfig,
    OracleCollectionResult,
    OracleProvenance,
    OracleRngOutcome,
    OracleTargetMetadata,
    OracleTargets,
    RootActionOracleTarget,
    StableNodeOracleTarget,
    build_oracle_targets,
)
from sts2_training.decision.policy import ActionCandidate, PolicyModel, PriorHeuristicPolicy
from sts2_training.decision.pruner_features import (
    PRUNER_FEATURE_NAMES,
    PRUNER_FEATURE_SCHEMA_VERSION,
    stable_pruner_feature_matrix,
)
from sts2_training.decision.pruner_training_data import (
    PairwisePrunerExample,
    PrunerFrontierTrainingExample,
    PrunerNodeTrainingExample,
    build_pairwise_examples,
    load_pruner_frontiers,
)
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
    STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
    StableFrontierPruner,
    StablePruneContext,
    StablePruneNodeView,
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
    "LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION",
    "LinearStableFrontierPruner",
    "ORACLE_RECORD_SCHEMA_VERSION",
    "OracleCollectionConfig",
    "OracleCollectionResult",
    "OracleJsonlWriter",
    "OracleProvenance",
    "OracleRngOutcome",
    "OracleTargetMetadata",
    "OracleTargets",
    "PRUNER_FEATURE_NAMES",
    "PRUNER_FEATURE_SCHEMA_VERSION",
    "PairwisePrunerExample",
    "PolicyCandidateTrace",
    "PolicyModel",
    "PolicyProposalTrace",
    "PriorHeuristicPolicy",
    "PrunerFrontierTrainingExample",
    "PrunerNodeTrainingExample",
    "ResolvedNodeTrace",
    "RootActionOracleTarget",
    "SEARCH_MODES",
    "STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION",
    "SearchTraceCollector",
    "SearchTraceEnd",
    "SearchTraceStart",
    "StableFrontierPruner",
    "StableNodeOracleTarget",
    "StablePruneContext",
    "StablePruneNodeTrace",
    "StablePruneNodeView",
    "StablePruneTrace",
    "ValueModel",
    "ValueTopKPruner",
    "build_oracle_targets",
    "build_pairwise_examples",
    "load_pruner_frontiers",
    "oracle_collection_record",
    "resolve_search_mode",
    "stable_pruner_feature_matrix",
]
