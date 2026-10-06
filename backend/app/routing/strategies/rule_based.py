from typing import List

from app.core.constants import RoutingStrategy, TaskCategory
from app.routing.base import BaseRouter
from app.schemas.chat import ChatCompletionRequest
from app.schemas.features import RequestFeatures
from app.schemas.provider import ModelMetadata


class RuleBasedRouter(BaseRouter):
    """Deterministic expert heuristic routing strategy."""

    @property
    def strategy_name(self) -> RoutingStrategy:
        return RoutingStrategy.RULE_BASED

    def select_model(
        self,
        request: ChatCompletionRequest,
        features: RequestFeatures,
        candidates: List[ModelMetadata],
    ) -> ModelMetadata:
        self.validate_candidates(candidates)

        # Rule 1: Latency Sensitivity
        if features.latency_sensitive:
            fast_candidates = sorted(candidates, key=lambda m: m.expected_latency_ms)
            return fast_candidates[0]

        # Rule 2: Long Context Requirement (>2000 estimated tokens)
        if features.is_long_context:
            large_ctx = [m for m in candidates if m.context_window >= 16384]
            if large_ctx:
                return max(large_ctx, key=lambda m: m.context_window)

        # Rule 3: Coding Tasks
        if features.task_category == TaskCategory.CODING or features.requires_code:
            coding_matches = [
                m for m in candidates if "code" in m.model_name.lower() or "llama-3" in m.model_name.lower() or "gemini" in m.model_name.lower()
            ]
            if coding_matches:
                return max(coding_matches, key=lambda m: m.quality_score)

        # Rule 4: Reasoning & Multi-step Math
        if features.task_category == TaskCategory.REASONING or features.requires_reasoning:
            reasoning_matches = [
                m for m in candidates if "pro" in m.model_name.lower() or "r1" in m.model_name.lower() or m.quality_score >= 0.85
            ]
            if reasoning_matches:
                return max(reasoning_matches, key=lambda m: m.quality_score)

        # Rule 5: Summarization & Extraction
        if features.task_category in (TaskCategory.SUMMARIZATION, TaskCategory.EXTRACTION):
            fast_extractors = [
                m for m in candidates if "flash" in m.model_name.lower() or "mistral" in m.model_name.lower() or "fast" in m.model_name.lower()
            ]
            if fast_extractors:
                return min(fast_extractors, key=lambda m: m.expected_latency_ms)

        # Default fallback: highest score-to-latency ratio
        return max(candidates, key=lambda m: m.quality_score / max(m.expected_latency_ms, 1.0))
