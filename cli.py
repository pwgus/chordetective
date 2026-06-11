"""Command-line interface for Music Analyzer."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
from audio_processor import AudioProcessor
from music_analyzer import NoteDetector


def analyze_file(filepath: str, n_fft: int = 2048, method: str = 'pitch',
                 output_json: Optional[str] = None, confidence_threshold: float = 0.5):
    """
    Analyze audio file and optionally save results.

    Args:
        filepath: Path to audio file
        n_fft: FFT window size
        method: 'pitch' or 'chroma'
        output_json: Path to save JSON results (optional)
        confidence_threshold: Min confidence (0-1)
    """
    print(f"Loading {filepath}...")
    processor = AudioProcessor()
    processor.load(filepath)
    print(f"Duration: {processor.get_duration():.2f}s")

    print(f"Analyzing notes (method={method}, n_fft={n_fft})...")
    detector = NoteDetector(sr=processor.sr)

    if method == 'pitch':
        detections = detector.detect_notes_from_pitch(
            processor.y, n_fft=n_fft,
            confidence_threshold=confidence_threshold
        )
    elif method == 'chroma':
        detections = detector.detect_notes_from_chroma(
            processor.y, n_fft=n_fft,
            confidence_threshold=confidence_threshold
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Display results
    print(f"\nFound {len(detections)} notes:")
    for det in detections[:20]:  # Show first 20
        print(f"  {det.time:.2f}s: {det.note_name} ({det.confidence:.1f}%)")
    if len(detections) > 20:
        print(f"  ... and {len(detections) - 20} more")

    # Save JSON if requested
    if output_json:
        data = {
            'filepath': filepath,
            'duration': processor.get_duration(),
            'n_fft': n_fft,
            'method': method,
            'detections': [
                {
                    'time': d.time,
                    'note': d.note_name,
                    'frequency': d.frequency,
                    'confidence': d.confidence
                }
                for d in detections
            ]
        }
        with open(output_json, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved to {output_json}")


def main():
    parser = argparse.ArgumentParser(description='Music Analyzer - Analyze audio for notes and chords')
    parser.add_argument('file', help='Audio file path (WAV, MP3, FLAC, OGG, M4A)')
    parser.add_argument('--n-fft', type=int, default=2048, help='FFT window size (default: 2048)')
    parser.add_argument('--method', choices=['pitch', 'chroma'], default='pitch',
                        help='Detection method (default: pitch)')
    parser.add_argument('--output-json', help='Save results to JSON file')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Min confidence threshold 0-1 (default: 0.5)')

    args = parser.parse_args()

    try:
        analyze_file(args.file, n_fft=args.n_fft, method=args.method,
                     output_json=args.output_json, confidence_threshold=args.confidence)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
