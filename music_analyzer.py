"""Musical note and chord detection."""

import numpy as np
import librosa
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Detection:
    """Single note/chord detection result."""
    time: float
    note_name: str
    frequency: float
    confidence: float  # 0-100


class NoteDetector:
    """Detect dominant notes in audio using chroma features and pitch."""

    # Equal temperament note mapping
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def __init__(self, sr: int = 22050):
        self.sr = sr

    @staticmethod
    def freq_to_note(freq: float) -> Tuple[str, int]:
        """Convert frequency (Hz) to note name and octave."""
        if freq <= 0:
            return "silence", -1

        # A4 = 440 Hz, index 57 in semitones
        semitones_from_a4 = 12 * np.log2(freq / 440)
        semitone = int(np.round(semitones_from_a4)) + 57

        octave = semitone // 12 - 1
        note_idx = semitone % 12
        note_name = NoteDetector.NOTES[note_idx]

        return note_name, octave

    @staticmethod
    def note_to_freq(note_name: str, octave: int) -> float:
        """Convert note name + octave to frequency (Hz)."""
        if note_name == "silence":
            return 0
        semitone = NoteDetector.NOTES.index(note_name) + (octave + 1) * 12 - 57
        return 440 * (2 ** (semitone / 12))

    def detect_chroma(self, y: np.ndarray, n_fft: int = 2048, hop_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract chroma features (pitch class distribution).

        Returns:
            (chroma, times) where chroma is [12 × time]
        """
        if hop_length is None:
            hop_length = n_fft // 4

        chroma = librosa.feature.chroma_cqt(y=y, sr=self.sr, hop_length=hop_length)
        times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=self.sr, hop_length=hop_length)

        return chroma, times

    def detect_pitch(self, y: np.ndarray, n_fft: int = 2048, hop_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect fundamental frequency using PYIN.

        Returns:
            (f0, times) where f0 is Hz per frame, times in seconds
        """
        if hop_length is None:
            hop_length = n_fft // 4

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
            sr=self.sr, hop_length=hop_length
        )

        times = librosa.frames_to_time(np.arange(len(f0)), sr=self.sr, hop_length=hop_length)

        return f0, voiced_probs, times

    def detect_notes_from_pitch(self, y: np.ndarray, n_fft: int = 2048, hop_length: Optional[int] = None,
                                 confidence_threshold: float = 0.5) -> List[Detection]:
        """
        Detect notes from pitch tracking.

        Args:
            confidence_threshold: Only return detections with confidence >= this (0-1)

        Returns:
            List of Detection objects
        """
        f0, voiced_probs, times = self.detect_pitch(y, n_fft, hop_length)

        detections = []
        for i, (freq, conf) in enumerate(zip(f0, voiced_probs)):
            if conf >= confidence_threshold and not np.isnan(freq):
                note_name, octave = self.freq_to_note(freq)
                detections.append(Detection(
                    time=float(times[i]),
                    note_name=f"{note_name}{octave}",
                    frequency=float(freq),
                    confidence=float(conf * 100)
                ))

        return detections

    def detect_notes_from_chroma(self, y: np.ndarray, n_fft: int = 2048, hop_length: Optional[int] = None,
                                  confidence_threshold: float = 0.3) -> List[Detection]:
        """
        Detect notes from chroma features (pitch class energy).

        Returns:
            List of Detection objects (simplified, no octave info)
        """
        chroma, times = self.detect_chroma(y, n_fft, hop_length)

        detections = []
        for i in range(chroma.shape[1]):
            energy = chroma[:, i]
            max_idx = np.argmax(energy)
            confidence = energy[max_idx] / (np.sum(energy) + 1e-8)

            if confidence >= confidence_threshold:
                note_name = self.NOTES[max_idx]
                detections.append(Detection(
                    time=float(times[i]),
                    note_name=note_name,
                    frequency=0.0,  # Chroma doesn't provide frequency
                    confidence=float(confidence * 100)
                ))

        return detections
