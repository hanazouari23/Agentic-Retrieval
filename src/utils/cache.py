"""Simple persistent disk cache for pickleable objects.

The cache is organised under ``cache_root / version``.  Changing the version
string creates a fresh namespace and invalidates old entries automatically.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional


class DiskCache:
    """File-based cache keyed by arbitrary strings.

    Each key is hashed with SHA-256 and stored as a pickle file.  The version
    component is baked into the storage directory so that component or prompt
    changes can start from a clean slate without deleting historical caches.
    """

    def __init__(self, cache_root: Path, version: str):
        self.cache_dir = Path(cache_root) / version
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Include the key itself in the hash input so different keys never
        # collide within the same version directory.
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.pkl"

    def get(self, key: str) -> Optional[Any]:
        path = self._path(key)
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def put(self, key: str, value: Any) -> None:
        path = self._path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f)

    def clear(self) -> int:
        """Remove all entries for this version.  Returns the number deleted."""
        removed = 0
        for path in self.cache_dir.glob("*.pkl"):
            path.unlink()
            removed += 1
        return removed
