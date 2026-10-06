from typing import List, Optional
from pydantic import BaseModel, Field

from app.core.constants import TaskCategory


class RequestFeatures(BaseModel):
    """Structured feature vector extracted deterministically from an incoming request."""

    prompt_length_chars: int = 0
    estimated_prompt_tokens: int = 0
    message_count: int = 1
    task_category: TaskCategory = TaskCategory.GENERAL
    requires_code: bool = False
    requires_reasoning: bool = False
    requires_structured_output: bool = False
    is_long_context: bool = False
    latency_sensitive: bool = False
    tenant_id: str = "default"

    def to_feature_vector(self) -> List[float]:
        """Encodes features into a normalized float vector for Contextual Bandit algorithms."""
        # 10-dimensional continuous/binary normalized feature vector:
        # [bias, log(tokens), msg_count/10, is_coding, is_reasoning, is_structured, is_long_ctx, is_latency_sens, is_summary, is_extract]
        import math

        norm_tokens = math.log1p(self.estimated_prompt_tokens) / 10.0  # Normalized ~[0.0, 1.0]
        norm_msg_count = min(self.message_count / 10.0, 1.0)

        return [
            1.0,  # Bias term
            round(norm_tokens, 4),
            round(norm_msg_count, 4),
            1.0 if self.requires_code else 0.0,
            1.0 if self.requires_reasoning else 0.0,
            1.0 if self.requires_structured_output else 0.0,
            1.0 if self.is_long_context else 0.0,
            1.0 if self.latency_sensitive else 0.0,
            1.0 if self.task_category == TaskCategory.SUMMARIZATION else 0.0,
            1.0 if self.task_category == TaskCategory.EXTRACTION else 0.0,
        ]
