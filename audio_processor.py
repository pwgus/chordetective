"""Audio loading and preprocessing module."""

import numpy as np
import librosa
import librosa.display
from pathlib import Path
from typing import Tuple, Optional


class AudioProcessor:
    """Handle audio file loading and spectrogram computation."""

    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

    def __init__(self, sr: int = 22050):
        """
        Args:
            sr: Sample rate (Hz). Default 22050 for fast processing.
        """
        self.sr = sr
        self.y = None  # Audio time series
        self.duration = None

    def load(self, filepath: str) -> bool:
        """Load audio file. Return True if successful."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {path.suffix}")

        try:
            self.y, _ = librosa.load(filepath, sr=self.sr)
            self.duration = librosa.get_duration(y=self.y, sr=self.sr)
            return True
        except Exception as e:
            raise RuntimeError(f"Error loading audio: {e}")

    def compute_spectrogram(self, n_fft: int = 2048, hop_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute magnitude spectrogram.

        Returns:
            (S, freqs, times) where S is magnitude spectrogram [freq × time]
        """
        if self.y is None:
            raise RuntimeError("No audio loaded. Call load() first.")

        if hop_length is None:
            hop_length = n_fft // 4

        # STFT
        D = librosa.stft(self.y, n_fft=n_fft, hop_length=hop_length)
        S = np.abs(D)

        # Frequency and time axes
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        times = librosa.frames_to_time(np.arange(S.shape[1]), sr=self.sr, hop_length=hop_length)

        return S, freqs, times

    def get_audio_chunk(self, start_time: float, duration: float) -> np.ndarray:
        """Extract audio chunk from [start_time, start_time + duration] in seconds."""
        if self.y is None:
            raise RuntimeError("No audio loaded.")

        start_sample = int(start_time * self.sr)
        end_sample = int((start_time + duration) * self.sr)
        return self.y[start_sample:end_sample]

    def get_duration(self) -> Optional[float]:
        """Return total duration in seconds."""
        return self.duration
