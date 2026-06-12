"""Audio loading, segmentation and shared signal helpers."""

import numpy as np
import librosa
from pathlib import Path
from typing import List, Tuple, Optional

# Segments quieter than this fraction of the loudest segment count as silence
SILENCE_RMS_RATIO = 0.02


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

    def compute_cqt(self, hop_length: int = 512, n_bins: int = 84,
                    bins_per_octave: int = 12, fmin: Optional[float] = None
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Constant-Q Transform magnitude.

        CQT uses geometrically spaced frequency bins (one per semitone by
        default), so resolution adapts per frequency — better suited to
        tonal music than the fixed-resolution STFT.

        Returns:
            (C, freqs, times) where C is magnitude [n_bins × time]
        """
        if self.y is None:
            raise RuntimeError("No audio loaded. Call load() first.")

        if fmin is None:
            fmin = librosa.note_to_hz('C1')

        # librosa.cqt requires hop_length divisible by 2**(n_octaves - 1)
        n_octaves = int(np.ceil(n_bins / bins_per_octave))
        required = 2 ** (n_octaves - 1)
        hop_length = max(required, (hop_length // required) * required)

        C = np.abs(librosa.cqt(self.y, sr=self.sr, hop_length=hop_length,
                               fmin=fmin, n_bins=n_bins,
                               bins_per_octave=bins_per_octave))
        freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin,
                                        bins_per_octave=bins_per_octave)
        times = librosa.frames_to_time(np.arange(C.shape[1]), sr=self.sr,
                                       hop_length=hop_length)
        return C, freqs, times

    def compute_transform(self, transform: str = 'cqt', n_fft: int = 2048,
                          hop_length: Optional[int] = None
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute magnitude spectrum with selected method ('cqt' or 'stft')."""
        if transform == 'stft':
            return self.compute_spectrogram(n_fft=n_fft, hop_length=hop_length)
        if hop_length is None:
            hop_length = n_fft // 4
        return self.compute_cqt(hop_length=hop_length)

    def detect_onsets(self, hop_length: int = 512, backtrack: bool = True) -> np.ndarray:
        """Detect note onset times (seconds) via onset strength peak picking."""
        if self.y is None:
            raise RuntimeError("No audio loaded.")
        return librosa.onset.onset_detect(
            y=self.y, sr=self.sr, hop_length=hop_length,
            backtrack=backtrack, units='time'
        )

    def compute_spectral_flux(self, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """
        Spectral flux envelope (frame-to-frame spectral change).

        Returns:
            (flux, times) — high values mark transitions, low values stability
        """
        if self.y is None:
            raise RuntimeError("No audio loaded.")
        flux = librosa.onset.onset_strength(y=self.y, sr=self.sr, hop_length=hop_length)
        times = librosa.frames_to_time(np.arange(len(flux)), sr=self.sr,
                                       hop_length=hop_length)
        return flux, times

    def segment_by_onsets(self, hop_length: int = 512,
                          min_segment: float = 0.05) -> List[Tuple[float, float]]:
        """Split audio at detected onsets: each segment is one musical event."""
        onsets = self.detect_onsets(hop_length=hop_length)
        return self._boundaries_to_segments(onsets, min_segment)

    def segment_fixed(self, window_size: int = 2048) -> List[Tuple[float, float]]:
        """Split audio into fixed windows of `window_size` samples."""
        if self.y is None:
            raise RuntimeError("No audio loaded.")
        win_t = max(window_size, 256) / self.sr
        segments = []
        t = 0.0
        while t < self.duration:
            end = min(t + win_t, self.duration)
            segments.append((t, end))
            t = end
        # Merge a tiny trailing remainder into the previous window
        if len(segments) > 1 and (segments[-1][1] - segments[-1][0]) < win_t * 0.5:
            last = segments.pop()
            segments[-1] = (segments[-1][0], last[1])
        return segments

    def segment_adaptive(self, sensitivity: float = 0.5, hop_length: int = 512,
                         min_segment: float = 0.05,
                         max_segment: float = 1.0) -> List[Tuple[float, float]]:
        """
        Flux-driven adaptive segmentation.

        Boundaries are placed at spectral-flux peaks above a threshold derived
        from `sensitivity` (0.0–1.0): low flux yields long windows (capped at
        `max_segment`), high flux yields short ones. Higher sensitivity means
        more boundaries / smaller windows.
        """
        if self.y is None:
            raise RuntimeError("No audio loaded.")

        flux, times = self.compute_spectral_flux(hop_length=hop_length)
        if flux.size == 0 or flux.max() <= 0:
            return [(0.0, self.duration)]

        norm = flux / flux.max()
        sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
        threshold = 1.0 - 0.95 * sensitivity

        rising = np.r_[True, norm[1:] >= norm[:-1]]
        falling = np.r_[norm[:-1] > norm[1:], True]
        boundaries = times[(norm >= threshold) & rising & falling]

        segments = self._boundaries_to_segments(boundaries, min_segment)

        # Split overly long stable stretches so the window keeps adapting
        out = []
        for t0, t1 in segments:
            n = int(np.ceil((t1 - t0) / max_segment))
            if n <= 1:
                out.append((t0, t1))
            else:
                step = (t1 - t0) / n
                out.extend((t0 + i * step, t0 + (i + 1) * step) for i in range(n))
        return out

    def _boundaries_to_segments(self, boundary_times, min_segment: float
                                ) -> List[Tuple[float, float]]:
        """Convert boundary times into contiguous segments covering the audio."""
        dur = self.duration
        bounds = sorted({0.0, dur} | {float(t) for t in boundary_times if 0.0 < t < dur})

        segments = []
        start = bounds[0]
        for b in bounds[1:]:
            if b - start >= min_segment:
                segments.append((start, b))
                start = b
        if not segments:
            return [(0.0, dur)]
        if segments[-1][1] < dur:
            segments[-1] = (segments[-1][0], dur)
        return segments

    def get_duration(self) -> Optional[float]:
        """Return total duration in seconds."""
        return self.duration


def compute_chroma(y: np.ndarray, sr: int, n_fft: int = 2048,
                   hop_length: Optional[int] = None,
                   transform: str = 'cqt') -> Tuple[np.ndarray, np.ndarray]:
    """Chroma features [12 × frames] plus frame times, via CQT or STFT."""
    if hop_length is None:
        hop_length = n_fft // 4
    if transform == 'stft':
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft,
                                             hop_length=hop_length)
    else:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr,
                                   hop_length=hop_length)
    return chroma, times


def segment_rms(y: np.ndarray, t0: float, t1: float, sr: int) -> float:
    """RMS energy of the audio between t0 and t1 (seconds)."""
    s0, s1 = int(t0 * sr), max(int(t1 * sr), int(t0 * sr) + 1)
    chunk = y[s0:s1]
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))


def segment_frames(times: np.ndarray, t0: float, t1: float) -> Tuple[int, int]:
    """Frame index range [i0, i1) covering segment [t0, t1); never empty."""
    i0 = int(np.searchsorted(times, t0, side='left'))
    i1 = int(np.searchsorted(times, t1, side='left'))
    i0 = min(i0, len(times) - 1)
    i1 = min(max(i1, i0 + 1), len(times))
    i0 = min(i0, i1 - 1)
    return i0, i1
