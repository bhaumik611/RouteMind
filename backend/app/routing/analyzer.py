import re
from typing import List, Tuple

from app.core.constants import TaskCategory
from app.schemas.chat import ChatCompletionRequest, ChatMessage
from app.schemas.features import RequestFeatures


class RequestAnalyzer:
    """Sub-millisecond deterministic request feature extractor."""

    # Regex patterns for fast category detection
    CODE_PATTERNS = re.compile(
        r"(def\s+\w+|class\s+\w+|import\s+\w+|from\s+\w+\s+import|```|function\s*\(|=>\s*\{|"
        r"SELECT\s+.*FROM|CREATE\s+TABLE|public\s+static\s+void|#include\s+<|\b(python|javascript|typescript|c\+\+|golang|sql|rust)\b|"
        r"debug\s+this|fix\s+the\s+bug|syntax\s+error|unit\s+test|refactor)",
        re.IGNORECASE,
    )

    REASONING_PATTERNS = re.compile(
        r"(step[\s\-]by[\s\-]step|explain\s+why|derive|proof|prove\s+that|calculate|solve\s+for|"
        r"logical\s+deduction|chain\s+of\s+thought|analyze\s+the\s+implications|trade[\s\-]off|"
        r"mathematical|physics|probability|theorem)",
        re.IGNORECASE,
    )

    STRUCTURED_PATTERNS = re.compile(
        r"(\bjson\b|\byaml\b|\bcsv\b|\bxml\b|schema|structured\s+output|format:\s*\{|format\s+as\s+json|"
        r"return\s+as\s+array|json\s+object)",
        re.IGNORECASE,
    )

    SUMMARIZATION_PATTERNS = re.compile(
        r"(summarize|summary|tl;?dr|briefly\s+explain|give\s+a\s+concise\s+overview|condense|digest)",
        re.IGNORECASE,
    )

    EXTRACTION_PATTERNS = re.compile(
        r"(extract|parse|pull\s+out|identify\s+all\s+entities|find\s+all\s+names|list\s+key\s+points|"
        r"tabularize|scrape\s+data)",
        re.IGNORECASE,
    )

    LATENCY_PATTERNS = re.compile(
        r"(fast|urgent|quick|asap|brief|one\s+sentence|in\s+short|short\s+answer)",
        re.IGNORECASE,
    )

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """Fast token heuristic (~4 chars per token for English text + word split heuristic)."""
        if not text:
            return 0
        words = len(text.split())
        chars = len(text)
        return max(1, int((words * 1.3 + chars / 4.0) / 2))

    @classmethod
    def analyze(cls, request: ChatCompletionRequest) -> RequestFeatures:
        """Extracts deterministic features from ChatCompletionRequest in < 0.5ms."""
        total_chars = 0
        combined_text_parts: List[str] = []

        for msg in request.messages:
            content = msg.content or ""
            total_chars += len(content)
            combined_text_parts.append(content)

        full_prompt = "\n".join(combined_text_parts)
        est_tokens = cls.estimate_tokens(full_prompt)
        msg_count = len(request.messages)

        # Keyword matching
        has_code = bool(cls.CODE_PATTERNS.search(full_prompt))
        has_reasoning = bool(cls.REASONING_PATTERNS.search(full_prompt))
        has_structured = bool(cls.STRUCTURED_PATTERNS.search(full_prompt))
        has_summary = bool(cls.SUMMARIZATION_PATTERNS.search(full_prompt))
        has_extraction = bool(cls.EXTRACTION_PATTERNS.search(full_prompt))
        has_latency_cue = bool(cls.LATENCY_PATTERNS.search(full_prompt))

        # Categorize dominant task
        category = TaskCategory.GENERAL
        if has_code:
            category = TaskCategory.CODING
        elif has_reasoning:
            category = TaskCategory.REASONING
        elif has_summary:
            category = TaskCategory.SUMMARIZATION
        elif has_extraction:
            category = TaskCategory.EXTRACTION
        elif has_structured:
            category = TaskCategory.CLASSIFICATION

        is_long = est_tokens > 2000
        latency_sens = request.latency_sensitive or has_latency_cue

        return RequestFeatures(
            prompt_length_chars=total_chars,
            estimated_prompt_tokens=est_tokens,
            message_count=msg_count,
            task_category=category,
            requires_code=has_code,
            requires_reasoning=has_reasoning,
            requires_structured_output=has_structured,
            is_long_context=is_long,
            latency_sensitive=latency_sens,
            tenant_id=request.tenant_id or "default",
        )
