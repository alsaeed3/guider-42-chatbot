from functools import lru_cache
import hashlib

def generate_cache_key(prompt: str) -> str:
    """Generate a consistent hash key for the prompt."""
    return hashlib.md5(prompt.lower().strip().encode()).hexdigest()

@lru_cache(maxsize=100)
def get_cached_response(cache_key: str) -> str:
    """Retrieve cached response using LRU cache decorator."""
    return None  # This will only store responses after they're generated 