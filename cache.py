"""
Response cache for reproducible batch benchmarking.

Keyed by a hash of the canonical (sorted-key) JSON of the context dict
sent to the LLM, so identical transaction states always map to the
same cache entry regardless of dict ordering.

Two-phase workflow this enables:
  1. Live run populates the cache as it goes (mode="live" in ai_agent.py).
  2. Cached run (mode="cached") only ever reads the cache -- a miss is
     treated as an LLM failure and falls back to the rule engine,
     rather than silently calling out to a live model mid-benchmark.
This is what makes a 1000-case AI benchmark reproducible: after the
first live pass, every subsequent run against the same cache file is
byte-identical.
"""

import hashlib
import json
import os


def make_key(context: dict) -> str:
    canonical = json.dumps(context, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


class ResponseCache:
    def __init__(self, path: str = None):
        self.path = path
        self._store: dict[str, dict] = {}
        if path and os.path.exists(path):
            with open(path) as f:
                self._store = json.load(f)

    def get(self, key: str):
        return self._store.get(key)

    def set(self, key: str, value: dict):
        self._store[key] = value

    def __len__(self):
        return len(self._store)

    def __contains__(self, key: str):
        return key in self._store

    def save(self, path: str = None):
        target = path or self.path
        if not target:
            raise ValueError("No path given to save the cache to.")
        with open(target, "w") as f:
            json.dump(self._store, f, indent=2)
