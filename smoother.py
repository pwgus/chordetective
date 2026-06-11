"""Post-analysis smoothing for note/chord detection sequences.

Every function here operates on detection dataclasses that expose `time`,
`confidence`, `duration` and one label field (`note_name` or `chord_name`).
Functions return new objects (via dataclasses.replace) and never mutate
their inputs, so raw and smoothed sequences can coexist.
"""

import dataclasses
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_PITCH_RE = re.compile(r'^([A-G]#?)(-?\d+)$')


@dataclass
class SmoothingSettings:
    """Configuration for the smoothing pipeline."""
    median_frames: int = 3            # temporal median filter size; 1 disables
    min_note_duration_ms: int = 100   # events shorter than this are absorbed; 0 disables
    majority_frames: int = 0          # gaussian-weighted sliding vote; 0/1 disables
    hmm: bool = False                 # Viterbi smoothing (computationally expensive)
    hmm_self_prob: float = 0.9        # probability of staying on the same label


def _label_field(det) -> str:
    for attr in ('note_name', 'chord_name'):
        if hasattr(det, attr):
            return attr
    raise TypeError(f"Unsupported detection type: {type(det)!r}")


def get_label(det) -> str:
    return getattr(det, _label_field(det))


def label_to_ordinal(label: str) -> Optional[int]:
    """Map a pitched label like 'C#4' to a semitone index.

    Uses the project convention from NoteDetector (440 Hz <-> index 57).
    Returns None for unpitched labels (chords, 'silence', octave-less notes).
    """
    m = _PITCH_RE.match(label or '')
    if not m:
        return None
    return NOTES.index(m.group(1)) + (int(m.group(2)) + 1) * 12


def ordinal_to_label(ordinal: int) -> str:
    return f"{NOTES[ordinal % 12]}{ordinal // 12 - 1}"


def ordinal_to_freq(ordinal: int) -> float:
    return float(440.0 * 2 ** ((ordinal - 57) / 12))


def with_label(det, label: str):
    """Return a copy of `det` with its label (and matching frequency) replaced."""
    changes = {_label_field(det): label}
    if hasattr(det, 'frequency'):
        ordinal = label_to_ordinal(label)
        if ordinal is not None:
            changes['frequency'] = ordinal_to_freq(ordinal)
        elif label_to_ordinal(get_label(det)) is not None:
            changes['frequency'] = 0.0  # pitched -> unpitched (silence)
    return dataclasses.replace(det, **changes)


def ensure_durations(dets: Sequence) -> List:
    """Fill missing durations from inter-detection gaps (legacy frame data)."""
    out = list(dets)
    if all(getattr(d, 'duration', 0.0) > 0 for d in out):
        return out
    gaps = [out[i + 1].time - out[i].time for i in range(len(out) - 1)]
    positive = [g for g in gaps if g > 0]
    default = float(np.median(positive)) if positive else 0.05
    filled = []
    for i, det in enumerate(out):
        dur = getattr(det, 'duration', 0.0)
        if dur <= 0:
            dur = gaps[i] if i < len(gaps) and gaps[i] > 0 else default
        filled.append(dataclasses.replace(det, duration=float(dur)))
    return filled


def median_filter(dets: Sequence, n_frames: int) -> List:
    """Temporal median filter over the label sequence.

    Duration-weighted and center-anchored: the center element is relabelled
    only when another label occupies a strict majority of the window's total
    duration. On uniform frame sequences this matches a classic categorical
    median (isolated one-frame detections get replaced by their sustained
    neighbours); on event-level segments (onset/adaptive modes, where every
    element may be long and legitimate) it never invents pitches or lets two
    long neighbours outvote a real event.
    """
    if n_frames <= 1 or len(dets) < 3:
        return list(dets)

    half = n_frames // 2
    labels = [get_label(d) for d in dets]
    durations = [max(getattr(d, 'duration', 0.0), 1e-9) for d in dets]

    out = []
    for i, det in enumerate(dets):
        lo, hi = max(0, i - half), min(len(dets), i + half + 1)
        weights = {}
        total = 0.0
        for j in range(lo, hi):
            weights[labels[j]] = weights.get(labels[j], 0.0) + durations[j]
            total += durations[j]

        new_label = labels[i]
        for label, weight in weights.items():
            if label != labels[i] and weight > total / 2:
                new_label = label
                break
        out.append(with_label(det, new_label) if new_label != labels[i] else det)
    return out


def majority_vote(dets: Sequence, n_frames: int) -> List:
    """Sliding-window majority vote with gaussian weights.

    Frames near the window center weigh more than frames at the edges, so
    sustained labels win over brief intrusions without shifting boundaries
    too far.
    """
    if n_frames <= 1 or len(dets) < 3:
        return list(dets)

    half = n_frames // 2
    sigma = max(n_frames / 4.0, 1.0)
    labels = [get_label(d) for d in dets]
    durations = [max(getattr(d, 'duration', 0.0), 1e-9) for d in dets]

    out = []
    for i, det in enumerate(dets):
        scores = {}
        for j in range(max(0, i - half), min(len(dets), i + half + 1)):
            weight = float(np.exp(-0.5 * ((j - i) / sigma) ** 2)) * durations[j]
            scores[labels[j]] = scores.get(labels[j], 0.0) + weight
        best = max(scores.values())
        if scores.get(labels[i]) == best:
            new_label = labels[i]
        else:
            new_label = max(scores, key=scores.get)
        out.append(with_label(det, new_label) if new_label != labels[i] else det)
    return out


def hmm_smooth(dets: Sequence, self_prob: float = 0.9) -> List:
    """Viterbi decoding of the most probable label sequence.

    States are the distinct labels observed in the sequence. Emission
    likelihood comes from each detection's confidence; the transition matrix
    strongly favours staying on the current label, which penalises
    implausibly fast label changes.
    """
    if len(dets) < 3:
        return list(dets)

    labels = [get_label(d) for d in dets]
    states = sorted(set(labels))
    k = len(states)
    if k < 2:
        return list(dets)

    idx = {s: i for i, s in enumerate(states)}
    obs = np.array([idx[l] for l in labels])
    conf = np.clip(np.array([d.confidence for d in dets]) / 100.0, 0.05, 0.95)

    self_prob = float(np.clip(self_prob, 0.5, 0.999))
    log_a = np.full((k, k), np.log((1.0 - self_prob) / (k - 1)))
    np.fill_diagonal(log_a, np.log(self_prob))

    t_len = len(dets)
    emis = np.repeat(np.log((1.0 - conf) / (k - 1))[:, None], k, axis=1)
    emis[np.arange(t_len), obs] = np.log(conf)

    delta = emis[0] - np.log(k)
    psi = np.zeros((t_len, k), dtype=int)
    for t in range(1, t_len):
        scores = delta[:, None] + log_a  # [prev_state, cur_state]
        psi[t] = np.argmax(scores, axis=0)
        delta = scores[psi[t], np.arange(k)] + emis[t]

    path = np.zeros(t_len, dtype=int)
    path[-1] = int(np.argmax(delta))
    for t in range(t_len - 2, -1, -1):
        path[t] = psi[t + 1][path[t + 1]]

    out = []
    for det, s in zip(dets, path):
        new_label = states[s]
        out.append(with_label(det, new_label) if new_label != get_label(det) else det)
    return out


def collapse_runs(dets: Sequence) -> List:
    """Merge consecutive detections sharing a label into single events."""
    if not dets:
        return []
    runs = []
    for det in dets:
        if runs and get_label(runs[-1]) == get_label(det):
            runs[-1] = _merge(runs[-1], det)
        else:
            runs.append(det)
    return runs


def _merge(a, b):
    """Merge detection `b` into `a` (duration-weighted confidence/frequency)."""
    dur_a = max(a.duration, 1e-9)
    dur_b = max(b.duration, 1e-9)
    total = dur_a + dur_b
    changes = {
        'duration': float((b.time + dur_b) - a.time),
        'confidence': float((a.confidence * dur_a + b.confidence * dur_b) / total),
    }
    if hasattr(a, 'frequency'):
        voiced = [(f, w) for f, w in ((a.frequency, dur_a), (b.frequency, dur_b)) if f > 0]
        if voiced:
            changes['frequency'] = float(sum(f * w for f, w in voiced) / sum(w for _, w in voiced))
        else:
            changes['frequency'] = 0.0
    return dataclasses.replace(a, **changes)


def enforce_min_duration(dets: Sequence, min_ms: int, silence_label: str = 'silence') -> List:
    """Absorb events shorter than `min_ms` into the preceding event.

    A short first event (no predecessor) is relabelled as silence instead.
    Adjacent events that end up with the same label are merged afterwards.
    """
    runs = collapse_runs(dets)
    if min_ms <= 0 or len(runs) < 2:
        return runs
    min_s = min_ms / 1000.0

    for _ in range(len(runs)):  # converges far earlier; bound guarantees exit
        out = []
        for run in runs:
            if out and run.duration < min_s:
                end = run.time + max(run.duration, 0.0)
                out[-1] = dataclasses.replace(out[-1], duration=float(end - out[-1].time))
            else:
                out.append(run)
        if (len(out) > 1 and out[0].duration < min_s
                and get_label(out[0]) != silence_label):
            out[0] = with_label(out[0], silence_label)
        out = collapse_runs(out)
        if len(out) == len(runs):
            return out
        runs = out
    return runs


class Smoother:
    """Applies the configured smoothing pipeline to a detection sequence.

    Order: median filter -> majority vote -> HMM -> collapse runs ->
    minimum-duration enforcement.
    """

    def __init__(self, settings: Optional[SmoothingSettings] = None):
        self.settings = settings or SmoothingSettings()

    def apply(self, dets: Sequence, silence_label: str = 'silence') -> List:
        if not dets:
            return []
        s = self.settings
        seq = ensure_durations(dets)
        if s.median_frames > 1:
            seq = median_filter(seq, s.median_frames)
        if s.majority_frames > 1:
            seq = majority_vote(seq, s.majority_frames)
        if s.hmm:
            seq = hmm_smooth(seq, s.hmm_self_prob)
        seq = collapse_runs(seq)
        return enforce_min_duration(seq, s.min_note_duration_ms, silence_label)
