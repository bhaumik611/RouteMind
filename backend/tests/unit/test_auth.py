from app.security.hashing import generate_api_key, hash_api_key, verify_api_key


def test_api_key_generation_and_hashing():
    key = generate_api_key(prefix="rm-test")
    assert key.startswith("rm-test-")
    assert len(key) > 20

    hashed = hash_api_key(key)
    assert len(hashed) == 64  # SHA-256 hex digest length
    assert verify_api_key(key, hashed) is True
    assert verify_api_key("wrong-key", hashed) is False
