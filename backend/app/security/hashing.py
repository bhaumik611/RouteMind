import hashlib
import hmac
import secrets


def hash_api_key(api_key: str) -> str:
    """Computes a secure SHA-256 digest of the API key for persistent storage."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def generate_api_key(prefix: str = "rm-live") -> str:
    """Generates a high-entropy cryptographically secure API key."""
    token = secrets.token_urlsafe(32)
    return f"{prefix}-{token}"


def verify_api_key(plain_key: str, hashed_key: str) -> bool:
    """Constant-time comparison of hashed keys."""
    computed_hash = hash_api_key(plain_key)
    return hmac.compare_digest(computed_hash, hashed_key)
