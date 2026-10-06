import abc
import time
from typing import AsyncIterator, List, Optional
import httpx

from app.core.config import settings
from app.core.constants import ProviderType
from app.core.exceptions import FreeTierViolationError, ProviderUnavailableError
from app.core.logging import logger
from app.schemas.chat import ChatCompletionChunk, ChatCompletionRequest
from app.schemas.provider import ModelMetadata, ProviderMetadata, ProviderResponse


class BaseLLMProvider(abc.ABC):
    """Abstract Base Class for all RouteMind LLM Provider Adapters."""

    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        self._http_client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.REQUEST_TIMEOUT_SECONDS, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=50, max_connections=200),
        )

    @property
    @abc.abstractmethod
    def provider_type(self) -> ProviderType:
        """Returns the unique ProviderType enum."""
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable provider display name."""
        pass

    @abc.abstractmethod
    def get_available_models(self) -> List[ModelMetadata]:
        """Returns the list of configured models supported by this provider."""
        pass

    @abc.abstractmethod
    async def _execute_complete(
        self, request: ChatCompletionRequest, model_name: str
    ) -> ProviderResponse:
        """Internal provider-specific completion execution logic."""
        pass

    @abc.abstractmethod
    async def _execute_stream(
        self, request: ChatCompletionRequest, model_name: str
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Internal provider-specific streaming execution logic."""
        pass

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Returns True if the provider endpoint is reachable and responsive."""
        pass

    def validate_model_safety(self, model_name: str) -> None:
        """Enforces FREE_ONLY constraint on this provider's model execution."""
        if settings.FREE_ONLY:
            settings.validate_model_safety(self.provider_type.value, model_name)

    async def complete(
        self, request: ChatCompletionRequest, model_name: Optional[str] = None
    ) -> ProviderResponse:
        """Executes a non-streaming chat completion with safety checks and latency tracking."""
        target_model = model_name or request.model
        self.validate_model_safety(target_model)

        start_time = time.perf_counter()
        try:
            response = await self._execute_complete(request, target_model)
            response.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return response
        except FreeTierViolationError:
            raise
        except Exception as e:
            latency = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                f"Provider [{self.name}] execution failed after {latency}ms: {str(e)}",
                extra={"extra_data": {"provider": self.provider_type.value, "model": target_model}},
            )
            raise ProviderUnavailableError(f"Provider {self.name} failed: {str(e)}")

    async def stream(
        self, request: ChatCompletionRequest, model_name: Optional[str] = None
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Executes a streaming chat completion."""
        target_model = model_name or request.model
        self.validate_model_safety(target_model)

        try:
            async for chunk in self._execute_stream(request, target_model):
                yield chunk
        except Exception as e:
            logger.error(f"Provider [{self.name}] streaming failed: {str(e)}")
            raise ProviderUnavailableError(f"Provider {self.name} stream failed: {str(e)}")

    def get_metadata(self) -> ProviderMetadata:
        """Returns structured metadata including all active models."""
        return ProviderMetadata(
            id=self.provider_type,
            name=self.name,
            is_enabled=settings.is_provider_available(self.provider_type),
            is_free_only_compatible=True,
            models=self.get_available_models(),
        )

    async def close(self):
        """Closes the underlying HTTP client session."""
        await self._http_client.aclose()
