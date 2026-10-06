import json
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx

from app.core.config import settings
from app.core.constants import GEMINI_VERIFIED_FREE_MODELS, ProviderType
from app.core.exceptions import ProviderUnavailableError
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


class GeminiProvider(BaseLLMProvider):
    """Adapter for Google Gemini Free Tier REST API."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.GEMINI

    @property
    def name(self) -> str:
        return "Google Gemini Free Tier"

    def get_available_models(self) -> List[ModelMetadata]:
        models = []
        for model_id in GEMINI_VERIFIED_FREE_MODELS:
            models.append(
                ModelMetadata(
                    id=f"gemini/{model_id}",
                    provider_id=ProviderType.GEMINI,
                    model_name=model_id,
                    is_free=True,
                    context_window=1048576 if "1.5" in model_id or "2.0" in model_id else 32768,
                    supports_streaming=True,
                    quality_score=0.92 if "pro" in model_id else 0.86,
                    expected_latency_ms=450.0,
                    is_active=True,
                )
            )
        return models

    def _convert_messages_to_gemini_payload(
        self, request: ChatCompletionRequest
    ) -> Dict[str, Any]:
        """Translates standard OpenAI messages to Google Gemini contents schema."""
        contents: List[Dict[str, Any]] = []
        system_instruction: Optional[Dict[str, Any]] = None

        for msg in request.messages:
            if msg.role == "system":
                system_instruction = {"parts": [{"text": msg.content}]}
            else:
                role = "user" if msg.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg.content}]})

        # Ensure at least one content entry exists
        if not contents and system_instruction:
            contents.append({"role": "user", "parts": [{"text": "Hello"}]})

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
                "topP": request.top_p,
            },
        }

        if system_instruction:
            payload["system_instruction"] = system_instruction

        if request.stop:
            stops = [request.stop] if isinstance(request.stop, str) else request.stop
            payload["generationConfig"]["stopSequences"] = stops

        return payload

    async def _execute_complete(
        self, request: ChatCompletionRequest, model_name: str
    ) -> ProviderResponse:
        url = f"{self.BASE_URL}/{model_name}:generateContent"
        params = {"key": settings.GEMINI_API_KEY or ""}
        payload = self._convert_messages_to_gemini_payload(request)

        response = await self._http_client.post(url, params=params, json=payload)
        if response.status_code != 200:
            error_text = response.text
            logger.error(
                f"Gemini API returned status {response.status_code}: {error_text}",
                extra={"extra_data": {"status_code": response.status_code, "model": model_name}},
            )
            raise ProviderUnavailableError(
                f"Gemini API error (HTTP {response.status_code}): {error_text}"
            )

        data = response.json()
        try:
            candidate = data["candidates"][0]
            text_content = candidate["content"]["parts"][0]["text"]
            finish_reason = candidate.get("finishReason", "STOP").lower()
            usage_meta = data.get("usageMetadata", {})

            return ProviderResponse(
                content=text_content,
                model=model_name,
                provider=self.provider_type,
                usage=UsageInfo(
                    prompt_tokens=usage_meta.get("promptTokenCount", 0),
                    completion_tokens=usage_meta.get("candidatesTokenCount", 0),
                    total_tokens=usage_meta.get("totalTokenCount", 0),
                ),
                finish_reason="stop" if finish_reason == "stop" else finish_reason,
                raw_response=data,
            )
        except (KeyError, IndexError) as e:
            raise ProviderUnavailableError(f"Unexpected Gemini API response format: {str(e)}")

    async def _execute_stream(
        self, request: ChatCompletionRequest, model_name: str
    ) -> AsyncIterator[ChatCompletionChunk]:
        url = f"{self.BASE_URL}/{model_name}:streamGenerateContent"
        params = {"key": settings.GEMINI_API_KEY or "", "alt": "sse"}
        payload = self._convert_messages_to_gemini_payload(request)

        async with self._http_client.stream(
            "POST", url, params=params, json=payload
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise ProviderUnavailableError(f"Gemini stream error ({response.status_code}): {body.decode()}")

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                try:
                    data = json.loads(data_str)
                    candidate = data["candidates"][0]
                    text_delta = candidate["content"]["parts"][0].get("text", "")
                    finish_reason = candidate.get("finishReason")
                    yield ChatCompletionChunk(
                        id=f"gemini-{model_name}",
                        model=model_name,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(content=text_delta),
                                finish_reason="stop" if finish_reason == "STOP" else finish_reason,
                            )
                        ],
                    )
                except Exception:
                    continue

    async def health_check(self) -> bool:
        if not settings.GEMINI_API_KEY:
            return False
        try:
            url = f"{self.BASE_URL}?key={settings.GEMINI_API_KEY}"
            res = await self._http_client.get(url, timeout=5.0)
            return res.status_code == 200
        except Exception:
            return False
