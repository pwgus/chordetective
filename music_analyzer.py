"""Musical note and chord detection."""

import numpy as np
import librosa
from typing import List, Tuple, Optional
from dataclasses import dataclass, field, asdict

from smoother import Smoother, SmoothingSettings


@dataclass
class Detection:
    """Single note/chord detection result."""
    time: float
    note_name: str
    frequency: float
    confidence: float  # 0-100
    duration: float = 0.0  # seconds; 0 for legacy frame-wise detections


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

    def detect_chroma(self, y: np.ndarray, n_fft: int = 2048, hop_length: Optional[int] = None,
                      transform: str = 'cqt') -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract chroma features (pitch class distribution).

        Args:
            transform: 'cqt' (default, variable-resolution) or 'stft'

        Returns:
            (chroma, times) where chroma is [12 × time]
        """
        if hop_length is None:
            hop_length = n_fft // 4

        if transform == 'stft':
            chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, n_fft=n_fft,
                                                 hop_length=hop_length)
        else:
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
                                  confidence_threshold: float = 0.3,
                                  transform: str = 'cqt') -> List[Detection]:
        """
        Detect notes from chroma features (pitch class energy).

        Returns:
            List of Detection objects (simplified, no octave info)
        """
        chroma, times = self.detect_chroma(y, n_fft, hop_length, transform=transform)

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

    def detect_notes_segmented(self, y: np.ndarray, segments: List[Tuple[float, float]],
                               method: str = 'pitch', transform: str = 'cqt',
                               n_fft: int = 2048, hop_length: Optional[int] = None,
                               confidence_threshold: float = 0.4) -> List[Detection]:
        """
        Detect one note per segment by aggregating frame-wise analysis.

        Frame features are computed once over the whole signal, then frames
        inside each segment are aggregated (median pitch / mean chroma) into
        a single Detection spanning the segment. Quiet or unvoiced segments
        become 'silence' detections so the output timeline stays contiguous.

        Args:
            segments: list of (start_s, end_s) from AudioProcessor segmentation
            method: 'pitch' (PYIN, with octave) or 'chroma' (pitch class only)
            transform: 'cqt' or 'stft' (chroma method only; PYIN is time-domain)
        """
        if not segments:
            return []
        if hop_length is None:
            hop_length = n_fft // 4

        seg_rms = [_segment_rms(y, t0, t1, self.sr) for t0, t1 in segments]
        peak_rms = max(seg_rms) if seg_rms else 0.0

        detections = []
        if method == 'pitch':
            f0, voiced_probs, times = self.detect_pitch(y, n_fft, hop_length)
            for (t0, t1), rms in zip(segments, seg_rms):
                i0, i1 = _segment_frames(times, t0, t1)
                seg_f0 = f0[i0:i1]
                seg_prob = voiced_probs[i0:i1]
                voiced = (~np.isnan(seg_f0)) & (seg_prob >= 0.3)
                conf = float(np.mean(seg_prob[voiced])) if voiced.any() else 0.0

                if (peak_rms <= 0 or rms < SILENCE_RMS_RATIO * peak_rms
                        or voiced.mean() < 0.3 or conf < confidence_threshold):
                    # Confidence reflects certainty that this IS silence
                    quiet = peak_rms <= 0 or rms < SILENCE_RMS_RATIO * peak_rms
                    sil_conf = 90.0 if quiet else float(np.clip(1.0 - conf, 0.0, 1.0)) * 100
                    detections.append(Detection(
                        time=float(t0), note_name='silence', frequency=0.0,
                        confidence=sil_conf, duration=float(t1 - t0)))
                else:
                    freq = float(np.median(seg_f0[voiced]))
                    name, octave = self.freq_to_note(freq)
                    detections.append(Detection(
                        time=float(t0), note_name=f"{name}{octave}",
                        frequency=freq, confidence=conf * 100,
                        duration=float(t1 - t0)))
        else:
            chroma, times = self.detect_chroma(y, n_fft, hop_length, transform=transform)
            for (t0, t1), rms in zip(segments, seg_rms):
                i0, i1 = _segment_frames(times, t0, t1)
                energy = chroma[:, i0:i1].mean(axis=1)
                max_idx = int(np.argmax(energy))
                conf = float(energy[max_idx] / (np.sum(energy) + 1e-8))

                if (peak_rms <= 0 or rms < SILENCE_RMS_RATIO * peak_rms
                        or conf < confidence_threshold):
                    quiet = peak_rms <= 0 or rms < SILENCE_RMS_RATIO * peak_rms
                    sil_conf = 90.0 if quiet else float(np.clip(1.0 - conf, 0.0, 1.0)) * 100
                    detections.append(Detection(
                        time=float(t0), note_name='silence', frequency=0.0,
                        confidence=sil_conf, duration=float(t1 - t0)))
                else:
                    detections.append(Detection(
                        time=float(t0), note_name=self.NOTES[max_idx],
                        frequency=0.0, confidence=conf * 100,
                        duration=float(t1 - t0)))

        return detections


# Segments quieter than this fraction of the loudest segment count as silence
SILENCE_RMS_RATIO = 0.02


def _segment_rms(y: np.ndarray, t0: float, t1: float, sr: int) -> float:
    """RMS energy of the audio between t0 and t1 (seconds)."""
    s0, s1 = int(t0 * sr), max(int(t1 * sr), int(t0 * sr) + 1)
    chunk = y[s0:s1]
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))


def _segment_frames(times: np.ndarray, t0: float, t1: float) -> Tuple[int, int]:
    """Frame index range [i0, i1) covering segment [t0, t1); never empty."""
    i0 = int(np.searchsorted(times, t0, side='left'))
    i1 = int(np.searchsorted(times, t1, side='left'))
    i0 = min(i0, len(times) - 1)
    i1 = min(max(i1, i0 + 1), len(times))
    i0 = min(i0, i1 - 1)
    return i0, i1


@dataclass
class AnalysisSettings:
    """All knobs for the segmentation + detection + smoothing pipeline."""
    transform: str = 'cqt'             # 'cqt' | 'stft'
    segmentation: str = 'onsets'       # 'onsets' | 'fixed' | 'adaptive'
    window_size: int = 2048            # samples; fixed mode only
    flux_sensitivity: float = 0.5      # 0.0-1.0; adaptive mode only
    n_fft: int = 2048
    note_method: str = 'pitch'         # 'pitch' | 'chroma'
    note_confidence: float = 0.4
    chord_confidence: float = 0.3
    detect_notes: bool = True
    detect_chords: bool = True
    smoothing: SmoothingSettings = field(default_factory=SmoothingSettings)

    @classmethod
    def from_config(cls, cfg: dict) -> 'AnalysisSettings':
        """Build settings from a user-config dict (see config.Config)."""
        smoothing = SmoothingSettings(
            median_frames=int(cfg.get('median_frames', 3)),
            min_note_duration_ms=int(cfg.get('min_note_duration_ms', 100)),
            majority_frames=int(cfg.get('majority_frames', 0)),
            hmm=bool(cfg.get('hmm_smoothing', False)),
        )
        return cls(
            transform=cfg.get('transform', 'cqt'),
            segmentation=cfg.get('segmentation', 'onsets'),
            window_size=int(cfg.get('window_size', 2048)),
            flux_sensitivity=float(cfg.get('flux_sensitivity', 0.5)),
            n_fft=int(cfg.get('n_fft', 2048)),
            note_method=cfg.get('note_method', 'pitch'),
            note_confidence=float(cfg.get('note_confidence_threshold', 0.4)),
            chord_confidence=float(cfg.get('chord_confidence_threshold', 0.3)),
            detect_notes=bool(cfg.get('detect_notes', True)),
            detect_chords=bool(cfg.get('detect_chords', True)),
            smoothing=smoothing,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def cache_token(self) -> str:
        """Compact string identifying analysis-relevant parameters (cache key)."""
        s = self.smoothing
        return (f"v2_{self.transform}_{self.segmentation}_w{self.window_size}"
                f"_f{self.flux_sensitivity:.2f}_{self.note_method}"
                f"_nc{self.note_confidence:.2f}_cc{self.chord_confidence:.2f}"
                f"_md{s.median_frames}_dur{s.min_note_duration_ms}"
                f"_mj{s.majority_frames}_hmm{int(s.hmm)}")


@dataclass
class AnalysisResult:
    """Raw and smoothed sequences plus the structures used to derive them."""
    raw_notes: List[Detection]
    notes: List[Detection]             # smoothed
    raw_chords: list
    chords: list                       # smoothed
    onset_times: List[float]
    segments: List[Tuple[float, float]]


def run_full_analysis(processor, settings: AnalysisSettings) -> AnalysisResult:
    """
    Full pipeline: segment audio, detect notes/chords per segment, smooth.

    Args:
        processor: a loaded AudioProcessor
        settings: AnalysisSettings (see AnalysisSettings.from_config)
    """
    from chord_detector import ChordDetector

    y, sr = processor.y, processor.sr
    if y is None:
        raise RuntimeError("No audio loaded. Call processor.load() first.")

    onset_times = [float(t) for t in processor.detect_onsets()]

    if settings.segmentation == 'fixed':
        segments = processor.segment_fixed(settings.window_size)
    elif settings.segmentation == 'adaptive':
        segments = processor.segment_adaptive(settings.flux_sensitivity)
    else:
        segments = processor.segment_by_onsets()

    raw_notes: List[Detection] = []
    if settings.detect_notes:
        detector = NoteDetector(sr=sr)
        raw_notes = detector.detect_notes_segmented(
            y, segments, method=settings.note_method,
            transform=settings.transform, n_fft=settings.n_fft,
            confidence_threshold=settings.note_confidence)

    raw_chords: list = []
    if settings.detect_chords:
        chord_detector = ChordDetector(sr=sr)
        raw_chords = chord_detector.detect_chords_segmented(
            y, segments, transform=settings.transform, n_fft=settings.n_fft,
            confidence_threshold=settings.chord_confidence)

    smoother = Smoother(settings.smoothing)
    return AnalysisResult(
        raw_notes=raw_notes,
        notes=smoother.apply(raw_notes, silence_label='silence'),
        raw_chords=raw_chords,
        chords=smoother.apply(raw_chords, silence_label='N.C.'),
        onset_times=onset_times,
        segments=segments,
    )
