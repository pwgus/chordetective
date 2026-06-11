"""Chord detection from chroma features."""

import numpy as np
import librosa
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class ChordDetection:
    """Single chord detection result."""
    time: float
    chord_name: str
    confidence: float  # 0-100


class ChordDetector:
    """Detect chords from chroma features."""

    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Chord templates (normalized chroma patterns)
    TEMPLATES = {
        # Major triads
        'C': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
        'Cm': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
        'C7': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]),
        'Cmaj7': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]),
        'Cm7': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
    }

    def __init__(self, sr: int = 22050):
        self.sr = sr
        self._generate_all_templates()

    def _generate_all_templates(self):
        """Generate templates for all 12 keys."""
        base_templates = {
            'maj': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
            'min': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
            '7': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]),
            'maj7': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]),
            'min7': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
        }

        self.templates = {}
        for root_idx in range(12):
            root_name = self.NOTES[root_idx]
            for chord_type, pattern in base_templates.items():
                # Rotate pattern for different roots
                rotated = np.roll(pattern, root_idx)
                chord_name = f"{root_name}{chord_type}"
                self.templates[chord_name] = rotated

    def detect_chords(self, y: np.ndarray, n_fft: int = 2048, hop_length: None = None,
                      confidence_threshold: float = 0.3) -> List[ChordDetection]:
        """
        Detect chords from audio.

        Args:
            confidence_threshold: Min correlation score (0-1)

        Returns:
            List of ChordDetection objects
        """
        if hop_length is None:
            hop_length = n_fft // 4

        # Extract chroma
        chroma = librosa.feature.chroma_cqt(y=y, sr=self.sr, hop_length=hop_length)

        # Normalize chroma frames
        chroma_norm = chroma / (np.sum(chroma, axis=0, keepdims=True) + 1e-8)

        times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=self.sr, hop_length=hop_length)

        detections = []
        for i in range(chroma.shape[1]):
            frame = chroma_norm[:, i]

            # Correlate with all chord templates
            best_chord = None
            best_score = 0

            for chord_name, template in self.templates.items():
                # Cosine similarity
                score = np.dot(frame, template) / (np.linalg.norm(frame) * np.linalg.norm(template) + 1e-8)
                if score > best_score:
                    best_score = score
                    best_chord = chord_name

            if best_score >= confidence_threshold:
                detections.append(ChordDetection(
                    time=float(times[i]),
                    chord_name=best_chord,
                    confidence=float(best_score * 100)
                ))

        return detections

    def smooth_detections(self, detections: List[ChordDetection], window_size: int = 3) -> List[ChordDetection]:
        """
        Smooth chord detections by merging repeated chords within time window.

        Args:
            window_size: Time window in frames to consider for smoothing
        """
        if not detections:
            return []

        smoothed = []
        i = 0
        while i < len(detections):
            current = detections[i]
            same_chord_group = [current]

            # Collect all detections with same chord nearby
            j = i + 1
            while j < len(detections) and detections[j].time - current.time < window_size * 0.02:
                if detections[j].chord_name == current.chord_name:
                    same_chord_group.append(detections[j])
                j += 1

            # Average confidence and time
            avg_time = np.mean([d.time for d in same_chord_group])
            avg_conf = np.mean([d.confidence for d in same_chord_group])

            smoothed.append(ChordDetection(
                time=avg_time,
                chord_name=current.chord_name,
                confidence=avg_conf
            ))

            i = j if j > i + 1 else i + 1

        return smoothed
