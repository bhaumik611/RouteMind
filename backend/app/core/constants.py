from enum import Enum
from typing import Set


class ProviderType(str, Enum):
    OPENROUTER = "openrouter"
    GEMINI = "gemini"
    LOCAL_MOCK = "local_mock"
    OLLAMA = "ollama"


class RoutingStrategy(str, Enum):
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    CHEAPEST = "cheapest"
    RULE_BASED = "rule_based"
    QUALITY_FIRST = "quality_first"
    LATENCY_FIRST = "latency_first"
    BANDIT = "bandit"


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class TaskCategory(str, Enum):
    GENERAL = "general"
    CODING = "coding"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    REWRITING = "rewriting"


class UserRole(str, Enum):
    ADMIN = "ADMIN"
    DEVELOPER = "DEVELOPER"
    USER = "USER"
    VIEWER = "VIEWER"


# Standard HTTP Header Keys
HEADER_REQUEST_ID = "X-Request-ID"
HEADER_TENANT_ID = "X-Tenant-ID"
HEADER_API_KEY = "X-API-Key"
HEADER_ROUTED_PROVIDER = "X-RouteMind-Provider"
HEADER_ROUTED_MODEL = "X-RouteMind-Model"
HEADER_ROUTING_STRATEGY = "X-RouteMind-Strategy"
HEADER_LATENCY_MS = "X-RouteMind-Latency-MS"


# Whitelist of Verified Free Tier Models
# OpenRouter requires the ':free' suffix to guarantee zero cost
OPENROUTER_VERIFIED_FREE_MODELS: Set[str] = {
    "meta-llama/llama-3-8b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "meta-llama/llama-3.2-1b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-2-9b-it:free",
    "qwen/qwen-2-7b-instruct:free",
    "deepseek/deepseek-r1-distill-llama-70b:free",
    "microsoft/phi-3-mini-128k-instruct:free",
}

# Google Gemini Free Tier Models (Subject to standard free tier rate limits)
GEMINI_VERIFIED_FREE_MODELS: Set[str] = {
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash-8b",
}

# Local / Mock Verified Models
LOCAL_VERIFIED_MODELS: Set[str] = {
    "mock-free-model",
    "mock-coding-model",
    "mock-fast-model",
    "llama3:8b",
    "mistral:7b",
    "qwen2.5:7b",
}
