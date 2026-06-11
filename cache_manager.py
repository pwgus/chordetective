"""Cache analysis results to avoid reprocessing."""

import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, List
import pickle


class CacheManager:
    """Manage cached audio analyses."""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    @staticmethod
    def _hash_file(filepath: str, n_fft: int, method: str) -> str:
        """Generate cache key from file + parameters."""
        key_str = f"{filepath}_{n_fft}_{method}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_cache_key(self, filepath: str, n_fft: int, method: str) -> str:
        """Get cache filename for given parameters."""
        hash_key = self._hash_file(filepath, n_fft, method)
        return str(self.cache_dir / f"{hash_key}.pkl")

    def load_analysis(self, filepath: str, n_fft: int, method: str) -> Optional[Dict]:
        """Load cached analysis if exists."""
        cache_file = self.get_cache_key(filepath, n_fft, method)

        if Path(cache_file).exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    def save_analysis(self, filepath: str, n_fft: int, method: str, data: Dict):
        """Cache analysis results."""
        cache_file = self.get_cache_key(filepath, n_fft, method)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception:
            pass  # Fail silently

    def clear_cache(self):
        """Clear all cached analyses."""
        import shutil
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir(exist_ok=True)
