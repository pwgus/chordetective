"""Command-line interface for Music Analyzer."""

import argparse
import dataclasses
import json
import sys
from typing import Optional

from audio_processor import AudioProcessor
from music_analyzer import AnalysisSettings, AnalysisResult, run_full_analysis
from config import Config

# Maps CLI argument names to user-config keys (persisted between runs)
CONFIG_ARG_MAP = {
    'n_fft': 'n_fft',
    'method': 'note_method',
    'confidence': 'note_confidence_threshold',
    'segmentation': 'segmentation',
    'window_size': 'window_size',
    'flux_sensitivity': 'flux_sensitivity',
    'min_note_duration': 'min_note_duration_ms',
    'median_frames': 'median_frames',
    'majority_frames': 'majority_frames',
    'hmm': 'hmm_smoothing',
    'transform': 'transform',
}


def analyze_file(filepath: str, settings: AnalysisSettings,
                 output_json: Optional[str] = None) -> AnalysisResult:
    """
    Analyze audio file with the full pipeline and optionally save results.

    Args:
        filepath: Path to audio file
        settings: Analysis pipeline settings
        output_json: Path to save JSON results (optional)
    """
    print(f"Loading {filepath}...")
    processor = AudioProcessor()
    processor.load(filepath)
    print(f"Duration: {processor.get_duration():.2f}s")

    print(f"Analyzing (transform={settings.transform}, "
          f"segmentation={settings.segmentation}, method={settings.note_method})...")
    result = run_full_analysis(processor, settings)

    audible = [d for d in result.notes if d.note_name != 'silence']
    print(f"\nSegments analyzed: {len(result.raw_notes)} "
          f"-> {len(result.notes)} events after smoothing "
          f"({len(audible)} notes, {len(result.notes) - len(audible)} silences)")

    for det in result.notes[:20]:
        print(f"  {det.time:7.2f}s  {det.duration * 1000:5.0f}ms  "
              f"{det.note_name:<8s} ({det.confidence:.1f}%)")
    if len(result.notes) > 20:
        print(f"  ... and {len(result.notes) - 20} more")

    if output_json:
        data = {
            'filepath': filepath,
            'duration': processor.get_duration(),
            'settings': settings.to_dict(),
            'onsets': result.onset_times,
            'detections': [dataclasses.asdict(d) for d in result.notes],
            'raw_detections': [dataclasses.asdict(d) for d in result.raw_notes],
        }
        with open(output_json, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved to {output_json}")

    return result


def main():
    config = Config()

    parser = argparse.ArgumentParser(
        description='Music Analyzer - Analyze audio for notes and chords')
    parser.add_argument('file', help='Audio file path (WAV, MP3, FLAC, OGG, M4A)')
    parser.add_argument('--n-fft', type=int, default=None,
                        help='FFT window size (default: 2048)')
    parser.add_argument('--method', choices=['pitch', 'chroma'], default=None,
                        help='Detection method (default: pitch)')
    parser.add_argument('--output-json', help='Save results to JSON file')
    parser.add_argument('--confidence', type=float, default=None,
                        help='Min confidence threshold 0-1 (default: 0.4)')
    parser.add_argument('--segmentation', choices=['onsets', 'fixed', 'adaptive'],
                        default=None,
                        help='Segmentation mode (default: onsets)')
    parser.add_argument('--window-size', type=int, default=None,
                        help='Window size in samples, fixed mode only (default: 2048)')
    parser.add_argument('--flux-sensitivity', type=float, default=None,
                        help='Transition sensitivity 0.0-1.0, adaptive mode only (default: 0.5)')
    parser.add_argument('--min-note-duration', type=int, default=None,
                        help='Discard notes shorter than this many ms (default: 100)')
    parser.add_argument('--median-frames', type=int, default=None,
                        help='Median filter size in frames, 1-15, 1 disables (default: 3)')
    parser.add_argument('--majority-frames', type=int, default=None,
                        help='Gaussian-weighted majority vote window, 0 disables (default: 0)')
    parser.add_argument('--hmm', action=argparse.BooleanOptionalAction, default=None,
                        help='HMM (Viterbi) smoothing; slower (default: off)')
    parser.add_argument('--transform', choices=['cqt', 'stft'], default=None,
                        help='Analysis transform (default: cqt)')

    args = parser.parse_args()

    # Persist explicitly given parameters; unset ones fall back to saved config
    for arg_name, cfg_key in CONFIG_ARG_MAP.items():
        value = getattr(args, arg_name)
        if value is not None:
            config.set(cfg_key, value)

    settings = AnalysisSettings.from_config(config.to_dict())
    settings.detect_chords = False  # simple CLI analyzes notes only

    try:
        analyze_file(args.file, settings, output_json=args.output_json)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
