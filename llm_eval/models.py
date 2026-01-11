"""
Ollama client wrapper with caching and retry logic.

Design decisions:
- File-based cache: Simple, persistent, inspectable (no DB overhead)
- Retry with exponential backoff: Handles transient Ollama failures
- Hash-based cache keys: Ensures deterministic lookups
- Structured responses: Always return dict with metadata for debugging
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """Production-grade Ollama client with caching and fault tolerance."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        cache_dir: str = "results/.cache",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Args:
            base_url: Ollama server endpoint
            cache_dir: Directory for response cache (created if missing)
            max_retries: Number of retry attempts on failure
            retry_delay: Initial delay between retries (exponential backoff)

        Why caching: Avoids re-running expensive inferences during scoring phase.
        Why retries: Ollama can have transient failures under load.
        """
        self.client = ollama.Client(host=base_url)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        use_cache: bool = True,
    ) -> Optional[dict[str, Any]]:
        """
        Generate response from Ollama model with caching.

        Args:
            prompt: Input text
            model: Model identifier (e.g., 'qwen2.5:3b')
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum response length
            use_cache: Whether to use/store cached responses

        Returns:
            {
                'response': str,           # Generated text
                'model': str,              # Model used
                'prompt_tokens': int,      # Approximate input tokens
                'response_tokens': int,    # Approximate output tokens
                'total_duration': float,   # Generation time (seconds)
                'cached': bool             # Whether response was cached
            }

        Why this return structure: Provides all metadata needed for analysis
        without coupling to Ollama's internal response format.
        """
        cache_key = self._cache_key(prompt, model, temperature, max_tokens)

        # Check cache first
        if use_cache:
            cached = self._load_cache(cache_key)
            if cached:
                logger.info(f"Cache hit for {model}")
                return cached

        # Generate with retry logic
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Querying {model} (attempt {attempt + 1}/{self.max_retries})"
                )
                start_time = time.time()

                response = self.client.generate(
                    model=model,
                    prompt=prompt,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                )

                duration = time.time() - start_time

                # Structure response
                result = {
                    "response": response["response"],
                    "model": model,
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "response_tokens": response.get("eval_count", 0),
                    "total_duration": duration,
                    "cached": False,
                }

                # Cache for future use
                if use_cache:
                    self._save_cache(cache_key, result)

                return result

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)  # Exponential backoff
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All retries exhausted for {model}")
                    raise

    def _cache_key(
        self, prompt: str, model: str, temperature: float, max_tokens: int
    ) -> str:
        """
        Generate deterministic cache key from parameters.

        Why MD5: Fast, collision-resistant for our use case, human-readable length.
        """
        content = f"{model}|{temperature}|{max_tokens}|{prompt}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_cache(self, key: str) -> Optional[dict[str, Any]]:
        """Load cached response if exists."""
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    data["cached"] = True
                    return data
            except Exception as e:
                logger.warning(f"Failed to load cache {key}: {e}")
                return None

        return None

    def _save_cache(self, key: str, response: dict[str, Any]) -> None:
        """Persist response to cache."""
        cache_file = self.cache_dir / f"{key}.json"

        try:
            with open(cache_file, "w") as f:
                json.dump(response, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache {key}: {e}")

    def clear_cache(self) -> int:
        """
        Delete all cached responses.

        Returns:
            Number of files deleted

        Use case: Force fresh generation after model updates or prompt changes.
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1

        logger.info(f"Cleared {count} cache entries")
        return count


# Example usage
if __name__ == "__main__":
    client = OllamaClient()

    # Test with both models
    for model in ["ministral-3:3b", "deepseek-r1:7b"]:
        result = client.generate(
            prompt="What is 2+2? Answer briefly.", model=model, temperature=0.0
        )

        if result:
            print(f"\n{model}:")
            print(f"Response: {result['response']}")
            print(f"Tokens: {result['response_tokens']}")
            print(f"Duration: {result['total_duration']:.2f}s")
            print(f"Cached: {result['cached']}")
        else:
            print(f"Model- {model} did not generate any response")
