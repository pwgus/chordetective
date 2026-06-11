"""Export analysis results to JSON and CSV formats."""

import json
import csv
from pathlib import Path
from typing import List, Optional
from music_analyzer import Detection
from chord_detector import ChordDetection


class AnalysisExporter:
    """Export note and chord detections to various formats."""

    @staticmethod
    def export_json(output_path: str, audio_file: str, duration: float,
                   notes: List[Detection], chords: Optional[List[ChordDetection]] = None,
                   metadata: Optional[dict] = None):
        """Export analysis to JSON."""
        data = {
            'audio_file': str(Path(audio_file).name),
            'duration': duration,
            'metadata': metadata or {},
            'notes': [
                {
                    'time': n.time,
                    'duration': getattr(n, 'duration', 0.0),
                    'name': n.note_name,
                    'frequency': n.frequency,
                    'confidence': n.confidence
                }
                for n in notes
            ]
        }

        if chords:
            data['chords'] = [
                {
                    'time': c.time,
                    'duration': getattr(c, 'duration', 0.0),
                    'name': c.chord_name,
                    'confidence': c.confidence
                }
                for c in chords
            ]

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def export_csv_notes(output_path: str, notes: List[Detection]):
        """Export notes to CSV."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time (s)', 'Duration (s)', 'Note', 'Frequency (Hz)', 'Confidence (%)'])
            for note in notes:
                writer.writerow([
                    f'{note.time:.3f}',
                    f'{getattr(note, "duration", 0.0):.3f}',
                    note.note_name,
                    f'{note.frequency:.1f}',
                    f'{note.confidence:.1f}'
                ])

    @staticmethod
    def export_csv_chords(output_path: str, chords: List[ChordDetection]):
        """Export chords to CSV."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time (s)', 'Duration (s)', 'Chord', 'Confidence (%)'])
            for chord in chords:
                writer.writerow([
                    f'{chord.time:.3f}',
                    f'{getattr(chord, "duration", 0.0):.3f}',
                    chord.chord_name,
                    f'{chord.confidence:.1f}'
                ])

    @staticmethod
    def export_combined_csv(output_path: str, notes: List[Detection],
                           chords: Optional[List[ChordDetection]] = None):
        """Export notes and chords to single CSV (aligned by time)."""
        # Create time-indexed maps
        note_map = {}
        for note in notes:
            t = round(note.time, 2)
            if t not in note_map:
                note_map[t] = []
            note_map[t].append(note)

        chord_map = {}
        if chords:
            for chord in chords:
                t = round(chord.time, 2)
                if t not in chord_map:
                    chord_map[t] = []
                chord_map[t].append(chord)

        # Merge keys
        all_times = sorted(set(note_map.keys()) | set(chord_map.keys()))

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time (s)', 'Notes', 'Chords'])

            for time_point in all_times:
                notes_str = '; '.join([
                    f"{n.note_name} ({n.confidence:.0f}%)"
                    for n in note_map.get(time_point, [])
                ])
                chords_str = '; '.join([
                    f"{c.chord_name} ({c.confidence:.0f}%)"
                    for c in chord_map.get(time_point, [])
                ]) if chords else ''

                writer.writerow([f'{time_point:.2f}', notes_str, chords_str])
