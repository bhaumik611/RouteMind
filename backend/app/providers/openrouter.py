import json
from typing import AsyncIterator, List, Optional
import httpx

from app.core.config import settings
from app.core.constants import OPENROUTER_VERIFIED_FREE_MODELS, ProviderType
from app.core.exceptions import FreeTierViolationError, ProviderUnavailableError
from app.core.logging import logger
from app.providers.base import BaseLLMProvider
from app.schemas.chat import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    UsageInfo,
)
from app.schemas.provider import ModelMetadata, ProviderResponse


class OpenRouterProvider(BaseLLMProvider):
    """Adapter for OpenRouter API supporting verified free-tier models."""

    BASE_URL = "https://openrouter.ai/api/v1"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OPENROUTER

    @property
    def name(self) -> str:
        return "OpenRouter Free Tier"

    def _get_headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://routemind.internal",
            "X-Title": "RouteMind LLM Gateway",
        }
        if settings.OPENROUTER_API_KEY:
            headers["Authorization"] = f"Bearer {settings.OPENROUTER_API_KEY}"
        return headers

    def get_available_models(self) -> List[ModelMetadata]:
        models = []
        for model_id in OPENROUTER_VERIFIED_FREE_MODELS:
            models.append(
                ModelMetadata(
                    id=f"openrouter/{model_id}",
                    provider_id=ProviderType.OPENROUTER,
                    model_name=model_id,
                    is_free=True,
                    context_window=131072 if "128k" in model_id or "3.1" in model_id else 8192,
                    supports_streaming=True,
                    quality_score=0.85 if "llama-3" in model_id or "r1" in model_id else 0.78,
                    expected_latency_ms=650.0,
                    is_active=True,
                )
            )
        return models

    async def _execute_complete(
        self, request: ChatCompletionRequest, model_name: str
    ) -> ProviderResponse:
        url = f"{self.BASE_URL}/chat/completions"
        payload = {
            "model": model_name,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }

        response = await self._http_client.post(url, json=payload, headers=self._get_headers())
        if response.status_code != 200:
            error_text = response.text
            logger.error(
                f"OpenRouter returned status {response.status_code}: {error_text}",
                extra={"extra_data": {"status_code": response.status_code, "model": model_name}},
            )
            raise ProviderUnavailableError(
                f"OpenRouter API error (HTTP {response.status_code}): {error_text}"
            )

        data = response.json()
        choice = data["choices"][0]
        usage_data = data.get("usage", {})

        return ProviderResponse(
            content=choice["message"]["content"],
            model=data.get("model", model_name),
            provider=self.provider_type,
            usage=UsageInfo(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
            finish_reason=choice.get("finish_reason", "stop"),
            raw_response=data,
        )

    async def _execute_stream(
        self, request: ChatCompletionRequest, model_name: str
    ) -> AsyncIterator[ChatCompletionChunk]:
        url = f"{self.BASE_URL}/chat/completions"
        payload = {
            "model": model_name,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
        }

        async with self._http_client.stream(
            "POST", url, json=payload, headers=self._get_headers()
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise ProviderUnavailableError(f"OpenRouter stream error ({response.status_code}): {body.decode()}")

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    choice = data["choices"][0]
                    delta_data = choice.get("delta", {})
                    yield ChatCompletionChunk(
                        id=data.get("id", "chunk"),
                        model=data.get("model", model_name),
                        choices=[
                            ChatCompletionChunkChoice(
                                index=choice.get("index", 0),
                                delta=ChatCompletionChunkDelta(
                                    role=delta_data.get("role"),
                                    content=delta_data.get("content"),
                                ),
                                finish_reason=choice.get("finish_reason"),
                            )
                        ],
                    )
                except Exception:
                    continue

    async def health_check(self) -> bool:
        if not settings.OPENROUTER_API_KEY:
            return False
        try:
            res = await self._http_client.get(
                f"{self.BASE_URL}/auth/key", headers=self._get_headers(), timeout=5.0
            )
            return res.status_code == 200
        except Exception:
            return False
