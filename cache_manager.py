"""Cache analysis results to avoid reprocessing."""

import hashlib
import pickle
import shutil
from pathlib import Path
from typing import Optional


class CacheManager:
    """Pickle-file cache keyed by audio path + analysis parameters."""

    def __init__(self, cache_dir: str = '.cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _path(self, filepath: str, token: str) -> Path:
        key = hashlib.md5(f"{filepath}_{token}".encode()).hexdigest()
        return self.cache_dir / f"{key}.pkl"

    def load(self, filepath: str, token: str) -> Optional[dict]:
        """Return cached data for (file, parameters), or None."""
        path = self._path(filepath, token)
        if not path.exists():
            return None
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def save(self, filepath: str, token: str, data: dict):
        """Cache `data`; failures are silently ignored."""
        try:
            with open(self._path(filepath, token), 'wb') as f:
                pickle.dump(data, f)
        except Exception:
            pass

    def clear(self):
        """Delete the entire cache directory."""
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir(exist_ok=True)
