"""User configuration management."""

import json
from pathlib import Path
from typing import Any, Dict


class Config:
    """Store and retrieve user preferences (.music_analyzer.json)."""

    DEFAULT_CONFIG = {
        # Analysis
        'n_fft': 2048,
        'transform': 'cqt',               # 'cqt' | 'stft'
        'segmentation': 'onsets',         # 'onsets' | 'fixed' | 'adaptive'
        'window_size': 2048,              # samples; fixed mode only
        'flux_sensitivity': 0.5,          # 0.0-1.0; adaptive mode only
        'note_method': 'pitch',           # 'pitch' | 'chroma'
        'note_confidence_threshold': 0.4,
        'chord_confidence_threshold': 0.3,
        # Post-analysis smoothing
        'min_note_duration_ms': 100,      # 0 disables
        'median_frames': 3,               # 1 disables (range 1-15)
        'majority_frames': 0,             # 0 disables; gaussian-weighted vote
        'hmm_smoothing': False,           # Viterbi smoothing (slow)
        # Video
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
                    self.config.update(json.load(f))
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
        """Set one config value and save."""
        self.config[key] = value
        self.save()

    def update(self, values: Dict[str, Any]):
        """Set several config values, saving once."""
        if values:
            self.config.update(values)
            self.save()

    def to_dict(self) -> Dict:
        """Export current config."""
        return self.config.copy()
