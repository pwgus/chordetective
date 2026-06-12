"""Shared note names and pitch conversions.

Ordinals follow the MIDI convention: A4 (440 Hz) = 69, C4 = 60. Labels use
scientific pitch notation with sharps ('C#4'); 'silence' marks no pitch.
"""

import math
import re
from typing import Optional

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

A4_FREQ = 440.0
A4_ORDINAL = 69  # MIDI note number of A4

_PITCH_RE = re.compile(r'^([A-G]#?)(-?\d+)$')


def freq_to_label(freq: float) -> str:
    """Frequency (Hz) to note label ('A4'); 'silence' for non-positive."""
    if freq <= 0:
        return 'silence'
    ordinal = int(round(12 * math.log2(freq / A4_FREQ))) + A4_ORDINAL
    return ordinal_to_label(ordinal)


def label_to_ordinal(label: str) -> Optional[int]:
    """Pitched label ('C#4') to MIDI ordinal; None for unpitched labels."""
    m = _PITCH_RE.match(label or '')
    if not m:
        return None
    return NOTES.index(m.group(1)) + (int(m.group(2)) + 1) * 12


def ordinal_to_label(ordinal: int) -> str:
    return f"{NOTES[ordinal % 12]}{ordinal // 12 - 1}"


def ordinal_to_freq(ordinal: int) -> float:
    return float(A4_FREQ * 2 ** ((ordinal - A4_ORDINAL) / 12))


def chord_root(chord_name: str) -> Optional[str]:
    """Root note of a chord label ('C#min7' -> 'C#'); None if unparseable."""
    if chord_name[:2] in NOTES:
        return chord_name[:2]
    if chord_name[:1] in NOTES:
        return chord_name[:1]
    return None
