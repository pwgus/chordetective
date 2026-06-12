"""Chord detection from chroma features."""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from audio_processor import (SILENCE_RMS_RATIO, compute_chroma,
                             segment_frames, segment_rms)
from notes import NOTES, chord_root


@dataclass
class ChordDetection:
    """Single chord detection result."""
    time: float
    chord_name: str
    confidence: float  # 0-100
    duration: float = 0.0  # seconds; 0 for legacy frame-wise detections


class ChordDetector:
    """Detect chords from chroma features using template matching."""

    # Interval patterns from the root (semitones)
    PATTERNS = {
        # Triads
        'maj': [0, 4, 7],
        'min': [0, 3, 7],
        'aug': [0, 4, 8],
        'dim': [0, 3, 6],
        # Seventh chords
        'maj7': [0, 4, 7, 11],
        'min7': [0, 3, 7, 10],
        'dom7': [0, 4, 7, 10],
        'min7b5': [0, 3, 6, 10],   # half-diminished
        # Sixth chords
        'maj6': [0, 4, 7, 9],
        'min6': [0, 3, 7, 9],
        # Sus chords
        'sus2': [0, 2, 7],
        'sus4': [0, 5, 7],
    }

    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.templates = self._build_templates()

    @classmethod
    def _build_templates(cls) -> dict:
        """Binary chroma template per (root, chord type) combination."""
        templates = {}
        for root_idx, root_name in enumerate(NOTES):
            for chord_type, intervals in cls.PATTERNS.items():
                template = np.zeros(12)
                template[[(i + root_idx) % 12 for i in intervals]] = 1.0
                templates[f"{root_name}{chord_type}"] = template
        return templates

    def _best_match(self, vec: np.ndarray) -> Tuple[Optional[str], float]:
        """Best chord template for a chroma vector by cosine similarity."""
        vec_norm = np.linalg.norm(vec)
        best_chord, best_score = None, 0.0
        for chord_name, template in self.templates.items():
            score = np.dot(vec, template) / (
                vec_norm * np.linalg.norm(template) + 1e-8)
            if score > best_score:
                best_score = score
                best_chord = chord_name
        return best_chord, float(best_score)

    def detect_chords_segmented(self, y: np.ndarray, segments,
                                transform: str = 'cqt', n_fft: int = 2048,
                                hop_length: Optional[int] = None,
                                confidence_threshold: float = 0.3) -> List[ChordDetection]:
        """
        Detect one chord per segment by matching the segment's mean chroma.

        Quiet segments and matches below the threshold become 'N.C.'
        (no chord), keeping the output timeline contiguous.

        Args:
            segments: list of (start_s, end_s) from AudioProcessor segmentation
            transform: 'cqt' (default) or 'stft' chroma source
        """
        if not segments:
            return []
        if hop_length is None:
            hop_length = n_fft // 4

        chroma, times = compute_chroma(y, self.sr, n_fft, hop_length,
                                       transform=transform)

        seg_rms = [segment_rms(y, t0, t1, self.sr) for t0, t1 in segments]
        peak_rms = max(seg_rms) if seg_rms else 0.0

        detections = []
        for (t0, t1), rms in zip(segments, seg_rms):
            if peak_rms <= 0 or rms < SILENCE_RMS_RATIO * peak_rms:
                detections.append(ChordDetection(
                    time=float(t0), chord_name='N.C.', confidence=90.0,
                    duration=float(t1 - t0)))
                continue

            i0, i1 = segment_frames(times, t0, t1)
            vec = chroma[:, i0:i1].mean(axis=1)
            vec = vec / (vec.max() + 1e-8)

            best_chord, best_score = self._best_match(vec)
            if best_score >= confidence_threshold:
                name, conf = best_chord, best_score
            else:
                # Confidence reflects certainty that no chord matches
                name, conf = 'N.C.', 1.0 - best_score
            detections.append(ChordDetection(
                time=float(t0), chord_name=name,
                confidence=float(conf * 100), duration=float(t1 - t0)))

        return detections

    def get_key_estimate(self, chords: List[ChordDetection]) -> tuple:
        """
        Estimate key from chord sequence (confidence-weighted root count).

        Returns:
            (key_note, confidence) where key_note is 'C', 'D', etc.
        """
        if not chords:
            return None, 0.0

        root_counts = np.zeros(12)
        for chord in chords:
            root = chord_root(chord.chord_name)
            if root is not None:
                root_counts[NOTES.index(root)] += chord.confidence

        if root_counts.sum() == 0:
            return None, 0.0

        best_idx = int(np.argmax(root_counts))
        confidence = root_counts[best_idx] / root_counts.sum()
        return NOTES[best_idx], float(confidence)
