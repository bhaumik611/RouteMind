import asyncio
import time
import uuid
from typing import AsyncIterator, List, Optional

from app.core.constants import ProviderType
from app.providers.base import BaseLLMProvider
from app.schemas.chat import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    UsageInfo,
)
from app.schemas.provider import ModelMetadata, ProviderResponse


class LocalMockProvider(BaseLLMProvider):
    """Local high-fidelity Mock / Offline provider for unit testing, offline development, and zero-cost simulation."""

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.LOCAL_MOCK

    @property
    def name(self) -> str:
        return "Local Mock Provider"

    def get_available_models(self) -> List[ModelMetadata]:
        return [
            ModelMetadata(
                id="local_mock/mock-free-model",
                provider_id=ProviderType.LOCAL_MOCK,
                model_name="mock-free-model",
                is_free=True,
                context_window=8192,
                quality_score=0.75,
                expected_latency_ms=120.0,
            ),
            ModelMetadata(
                id="local_mock/mock-fast-model",
                provider_id=ProviderType.LOCAL_MOCK,
                model_name="mock-fast-model",
                is_free=True,
                context_window=4096,
                quality_score=0.65,
                expected_latency_ms=60.0,
            ),
            ModelMetadata(
                id="local_mock/mock-coding-model",
                provider_id=ProviderType.LOCAL_MOCK,
                model_name="mock-coding-model",
                is_free=True,
                context_window=16384,
                quality_score=0.88,
                expected_latency_ms=200.0,
            ),
        ]

    async def _execute_complete(
        self, request: ChatCompletionRequest, model_name: str
    ) -> ProviderResponse:
        # Simulate slight network/compute latency
        await asyncio.sleep(0.05)

        last_user_message = request.messages[-1].content if request.messages else "Hello"
        mock_response = f"[Mock response from {model_name}]: Successfully processed message '{last_user_message[:50]}...'"

        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        completion_tokens = len(mock_response.split())

        return ProviderResponse(
            content=mock_response,
            model=model_name,
            provider=self.provider_type,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            finish_reason="stop",
        )

    async def _execute_stream(
        self, request: ChatCompletionRequest, model_name: str
    ) -> AsyncIterator[ChatCompletionChunk]:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        tokens = ["This ", "is ", "a ", "streamed ", "mock ", "response ", "from ", f"{model_name}."]

        for token in tokens:
            await asyncio.sleep(0.01)
            yield ChatCompletionChunk(
                id=chunk_id,
                model=model_name,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(content=token),
                        finish_reason=None,
                    )
                ],
            )

        yield ChatCompletionChunk(
            id=chunk_id,
            model=model_name,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(),
                    finish_reason="stop",
                )
            ],
        )

    async def health_check(self) -> bool:
        return True
