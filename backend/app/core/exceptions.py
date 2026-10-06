from typing import Any, Dict, Optional


class RouteMindException(Exception):
    """Base exception for all RouteMind gateway errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_SERVER_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_code,
                "status_code": self.status_code,
                "details": self.details,
            }
        }


class FreeTierViolationError(RouteMindException):
    """Raised when a non-free model is requested while FREE_ONLY=true."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=403,
            error_code="FREE_TIER_VIOLATION",
            details=details,
        )


class QuotaExceededError(RouteMindException):
    """Raised when tenant or system virtual budget/quota is exceeded."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=429,
            error_code="QUOTA_EXCEEDED",
            details=details,
        )


class RateLimitExceededError(RouteMindException):
    """Raised when a rate limit is exceeded."""

    def __init__(self, message: str, retry_after: int = 60, details: Optional[Dict[str, Any]] = None):
        d = details or {}
        d["retry_after_seconds"] = retry_after
        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            details=d,
        )


class AuthenticationError(RouteMindException):
    """Raised when API key or tenant credentials are invalid."""

    def __init__(self, message: str = "Invalid or missing API key"):
        super().__init__(
            message=message,
            status_code=401,
            error_code="UNAUTHORIZED",
        )


class PolicyViolationError(RouteMindException):
    """Raised when a request violates tenant-defined policies."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=403,
            error_code="POLICY_VIOLATION",
            details=details,
        )


class ModelNotFoundError(RouteMindException):
    """Raised when a requested model is not found in the active catalog."""

    def __init__(self, model_id: str):
        super().__init__(
            message=f"Model '{model_id}' is not registered or is currently disabled.",
            status_code=404,
            error_code="MODEL_NOT_FOUND",
            details={"model_id": model_id},
        )


class ProviderUnavailableError(RouteMindException):
    """Raised when all candidate providers fail or their circuit breakers are OPEN."""

    def __init__(self, message: str = "All candidate providers are currently unavailable."):
        super().__init__(
            message=message,
            status_code=503,
            error_code="PROVIDER_UNAVAILABLE",
        )


class ConfigurationError(RouteMindException):
    """Raised on invalid system or gateway configuration."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            status_code=500,
            error_code="CONFIGURATION_ERROR",
        )
