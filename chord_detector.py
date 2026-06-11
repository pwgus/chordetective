"""Chord detection from chroma features."""

import numpy as np
import librosa
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ChordDetection:
    """Single chord detection result."""
    time: float
    chord_name: str
    confidence: float  # 0-100
    duration: float = 0.0  # seconds; 0 for legacy frame-wise detections


class ChordDetector:
    """Detect chords from chroma features using template matching."""

    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.templates = self._build_templates()

    def _build_templates(self) -> dict:
        """Build comprehensive chord templates for all keys and types."""
        templates = {}

        # Base patterns (intervals from root)
        patterns = {
            # Triads
            'maj': [0, 4, 7],      # Major: root, major 3rd, perfect 5th
            'min': [0, 3, 7],      # Minor: root, minor 3rd, perfect 5th
            'aug': [0, 4, 8],      # Augmented: root, major 3rd, augmented 5th
            'dim': [0, 3, 6],      # Diminished: root, minor 3rd, diminished 5th

            # Seventh chords
            'maj7': [0, 4, 7, 11],      # Major 7
            'min7': [0, 3, 7, 10],      # Minor 7
            'dom7': [0, 4, 7, 10],      # Dominant 7
            'min7b5': [0, 3, 6, 10],    # Half-diminished

            # Sixth chords
            'maj6': [0, 4, 7, 9],
            'min6': [0, 3, 7, 9],

            # Sus chords
            'sus2': [0, 2, 7],
            'sus4': [0, 5, 7],
        }

        # Generate templates for each root note
        for root_idx in range(12):
            root_name = self.NOTES[root_idx]

            for chord_type, intervals in patterns.items():
                template = np.zeros(12)
                for interval in intervals:
                    chroma_idx = (interval + root_idx) % 12
                    template[chroma_idx] = 1.0

                chord_name = f"{root_name}{chord_type}"
                templates[chord_name] = template

        return templates

    def _compute_chroma(self, y: np.ndarray, n_fft: int, hop_length: int,
                        transform: str = 'cqt') -> np.ndarray:
        """Chroma features via CQT (default) or STFT."""
        if transform == 'stft':
            return librosa.feature.chroma_stft(y=y, sr=self.sr, n_fft=n_fft,
                                               hop_length=hop_length)
        return librosa.feature.chroma_cqt(y=y, sr=self.sr, hop_length=hop_length)

    def detect_chords(self, y: np.ndarray, n_fft: int = 2048, hop_length: Optional[int] = None,
                      confidence_threshold: float = 0.3, smooth: bool = True,
                      transform: str = 'cqt') -> List[ChordDetection]:
        """
        Detect chords from audio using chroma feature matching.

        Args:
            y: Audio time series
            n_fft: FFT window size
            hop_length: Number of samples per frame
            confidence_threshold: Minimum correlation score (0-1)
            smooth: Apply temporal smoothing
            transform: 'cqt' (default) or 'stft' chroma source

        Returns:
            List of ChordDetection objects
        """
        if hop_length is None:
            hop_length = n_fft // 4

        # Extract chroma features
        chroma = self._compute_chroma(y, n_fft, hop_length, transform)
        chroma = librosa.power_to_db(chroma + 1e-8)

        # Normalize
        chroma_norm = (chroma - chroma.min(axis=0, keepdims=True)) / (chroma.max(axis=0, keepdims=True) - chroma.min(axis=0, keepdims=True) + 1e-8)

        times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=self.sr, hop_length=hop_length)

        detections = []
        for i in range(chroma.shape[1]):
            frame = chroma_norm[:, i]

            # Find best matching chord
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

        if smooth:
            detections = self._smooth_detections(detections)

        return detections

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

        from music_analyzer import (_segment_frames, _segment_rms,
                                    SILENCE_RMS_RATIO)

        chroma = self._compute_chroma(y, n_fft, hop_length, transform)
        times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=self.sr,
                                       hop_length=hop_length)

        seg_rms = [_segment_rms(y, t0, t1, self.sr) for t0, t1 in segments]
        peak_rms = max(seg_rms) if seg_rms else 0.0

        detections = []
        for (t0, t1), rms in zip(segments, seg_rms):
            if peak_rms <= 0 or rms < SILENCE_RMS_RATIO * peak_rms:
                detections.append(ChordDetection(
                    time=float(t0), chord_name='N.C.', confidence=90.0,
                    duration=float(t1 - t0)))
                continue

            i0, i1 = _segment_frames(times, t0, t1)
            vec = chroma[:, i0:i1].mean(axis=1)
            vec = vec / (vec.max() + 1e-8)

            best_chord, best_score = None, 0.0
            for chord_name, template in self.templates.items():
                score = np.dot(vec, template) / (
                    np.linalg.norm(vec) * np.linalg.norm(template) + 1e-8)
                if score > best_score:
                    best_score = score
                    best_chord = chord_name

            if best_score >= confidence_threshold:
                name, conf = best_chord, best_score
            else:
                # Confidence reflects certainty that no chord matches
                name, conf = 'N.C.', 1.0 - best_score
            detections.append(ChordDetection(
                time=float(t0), chord_name=name,
                confidence=float(conf * 100), duration=float(t1 - t0)))

        return detections

    def _smooth_detections(self, detections: List[ChordDetection], window_time: float = 0.5) -> List[ChordDetection]:
        """
        Smooth chord detections by grouping temporally close identical chords.

        Args:
            window_time: Time window in seconds to consider for grouping
        """
        if not detections:
            return []

        smoothed = []
        i = 0

        while i < len(detections):
            current = detections[i]
            group = [current]
            j = i + 1

            # Collect same chord within window
            while j < len(detections):
                if (detections[j].time - current.time < window_time and
                    detections[j].chord_name == current.chord_name):
                    group.append(detections[j])
                    j += 1
                else:
                    break

            # Average group
            avg_time = np.mean([d.time for d in group])
            avg_conf = np.mean([d.confidence for d in group])

            smoothed.append(ChordDetection(
                time=avg_time,
                chord_name=current.chord_name,
                confidence=avg_conf
            ))

            i = j

        return smoothed

    def get_key_estimate(self, chords: List[ChordDetection]) -> tuple:
        """
        Estimate key from chord sequence (simple heuristic).

        Returns:
            (key_note, confidence) where key_note is 'C', 'D', etc.
        """
        if not chords:
            return None, 0.0

        # Count root notes in detected chords
        root_counts = np.zeros(12)
        for chord in chords:
            root_note = chord.chord_name[:-1] if len(chord.chord_name) > 1 else chord.chord_name[0]
            try:
                idx = self.NOTES.index(root_note)
                root_counts[idx] += chord.confidence
            except ValueError:
                pass

        if root_counts.sum() == 0:
            return None, 0.0

        best_idx = np.argmax(root_counts)
        confidence = root_counts[best_idx] / root_counts.sum()

        return self.NOTES[best_idx], float(confidence)
