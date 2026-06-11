"""User configuration management."""

import json
from pathlib import Path
from typing import Any, Dict


class Config:
    """Store and retrieve user preferences."""

    DEFAULT_CONFIG = {
        'n_fft': 2048,
        'note_confidence_threshold': 0.4,
        'chord_confidence_threshold': 0.3,
        'sample_rate': 22050,
        'detect_notes': True,
        'detect_chords': True,
        'smooth_chords': True,
        'last_open_dir': str(Path.home()),
        'use_cache': True,
        'video_fps': 30,
        'video_resolution': '720p',
    }

    def __init__(self, config_file: str = '.music_analyzer.json'):
        self.config_file = Path(config_file)
        self.config = self.DEFAULT_CONFIG.copy()
        self.load()

    def load(self):
        """Load config from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    saved = json.load(f)
                    self.config.update(saved)
            except Exception:
                pass

    def save(self):
        """Save config to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception:
            pass

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """Set config value."""
        self.config[key] = value
        self.save()

    def to_dict(self) -> Dict:
        """Export current config."""
        return self.config.copy()
