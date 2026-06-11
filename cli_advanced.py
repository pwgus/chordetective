"""Advanced CLI with full feature support."""

import argparse
import sys
from pathlib import Path
from audio_processor import AudioProcessor
from music_analyzer import NoteDetector
from chord_detector import ChordDetector
from video_generator import VideoGenerator
from exporters import AnalysisExporter
from cache_manager import CacheManager
from config import Config


def analyze_and_export(audio_path: str, output_dir: str = None, n_fft: int = 2048,
                      note_conf: float = 0.4, chord_conf: float = 0.3,
                      generate_video: bool = False, video_res: str = '720p',
                      use_cache: bool = True):
    """Full analysis pipeline with export and optional video generation."""

    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"✗ File not found: {audio_path}", file=sys.stderr)
        return

    output_dir = Path(output_dir or audio_path.parent / "analysis")
    output_dir.mkdir(exist_ok=True)

    # Cache check
    cache = CacheManager()
    if use_cache:
        cached = cache.load_analysis(str(audio_path), n_fft, 'combined')
        if cached:
            print("✓ Loaded from cache")
            notes = cached.get('notes', [])
            chords = cached.get('chords', [])
            processor = cached.get('processor_data')
            duration = cached.get('duration')
        else:
            print("○ Cache miss, analyzing...")
            processor, notes, chords, duration = _analyze(audio_path, n_fft, note_conf, chord_conf)
            cache.save_analysis(str(audio_path), n_fft, 'combined',
                              {'notes': notes, 'chords': chords, 'duration': duration})
    else:
        processor, notes, chords, duration = _analyze(audio_path, n_fft, note_conf, chord_conf)

    # Export results
    print("Exporting results...")
    json_out = output_dir / f"{audio_path.stem}_analysis.json"
    AnalysisExporter.export_json(str(json_out), str(audio_path), duration, notes, chords)
    print(f"  ✓ JSON: {json_out}")

    csv_notes = output_dir / f"{audio_path.stem}_notes.csv"
    AnalysisExporter.export_csv_notes(str(csv_notes), notes)
    print(f"  ✓ Notes CSV: {csv_notes}")

    csv_chords = output_dir / f"{audio_path.stem}_chords.csv"
    AnalysisExporter.export_csv_chords(str(csv_chords), chords)
    print(f"  ✓ Chords CSV: {csv_chords}")

    csv_combined = output_dir / f"{audio_path.stem}_combined.csv"
    AnalysisExporter.export_combined_csv(str(csv_combined), notes, chords)
    print(f"  ✓ Combined CSV: {csv_combined}")

    # Print summary
    print(f"\n📊 Results:")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Notes detected: {len(notes)}")
    print(f"  Chords detected: {len(chords)}")
    if chords:
        chord_detector = ChordDetector()
        key, key_conf = chord_detector.get_key_estimate(chords)
        if key:
            print(f"  Estimated key: {key} ({key_conf*100:.0f}%)")

    # Generate video if requested
    if generate_video:
        print("\n🎬 Generating video...")
        video_out = output_dir / f"{audio_path.stem}_analysis.mp4"
        try:
            generator = VideoGenerator(fps=30)
            generator.generate_from_analysis(
                str(audio_path), str(video_out),
                notes, chords, resolution=video_res, n_fft=n_fft
            )
            print(f"  ✓ Video: {video_out}")
        except RuntimeError as e:
            print(f"  ✗ Video generation failed: {e}", file=sys.stderr)

    print(f"\n✓ All results saved to: {output_dir}")


def _analyze(audio_path: Path, n_fft: int, note_conf: float, chord_conf: float):
    """Perform analysis, return (processor, notes, chords, duration)."""
    processor = AudioProcessor()
    processor.load(str(audio_path))

    print(f"Detecting notes...")
    note_detector = NoteDetector(sr=processor.sr)
    notes = note_detector.detect_notes_from_pitch(
        processor.y, n_fft=n_fft, confidence_threshold=note_conf
    )
    print(f"  Found {len(notes)} notes")

    print(f"Detecting chords...")
    chord_detector = ChordDetector(sr=processor.sr)
    chords = chord_detector.detect_chords(
        processor.y, n_fft=n_fft, confidence_threshold=chord_conf, smooth=True
    )
    print(f"  Found {len(chords)} chords")

    duration = processor.get_duration()
    return processor, notes, chords, duration


def main():
    parser = argparse.ArgumentParser(
        description='Music Analyzer - Full analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  # Basic analysis
  python cli_advanced.py song.wav

  # High quality with video
  python cli_advanced.py song.wav --video --video-res 1080p

  # Batch processing
  python cli_advanced.py *.wav --output-dir ./results

  # High sensitivity
  python cli_advanced.py song.wav --note-conf 0.3 --chord-conf 0.2
        '''
    )

    parser.add_argument('files', nargs='+', help='Audio file(s) to analyze')
    parser.add_argument('--output-dir', help='Output directory (default: same as input)')
    parser.add_argument('--n-fft', type=int, default=2048,
                       help='FFT window size (default: 2048)')
    parser.add_argument('--note-conf', type=float, default=0.4,
                       help='Note confidence threshold 0-1 (default: 0.4)')
    parser.add_argument('--chord-conf', type=float, default=0.3,
                       help='Chord confidence threshold 0-1 (default: 0.3)')
    parser.add_argument('--video', action='store_true',
                       help='Generate MP4 video with visualization')
    parser.add_argument('--video-res', choices=['720p', '1080p'], default='720p',
                       help='Video resolution (default: 720p)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching')

    args = parser.parse_args()

    # Expand wildcards
    import glob
    all_files = []
    for pattern in args.files:
        matches = glob.glob(pattern)
        if matches:
            all_files.extend(matches)
        else:
            all_files.append(pattern)

    if not all_files:
        parser.print_help()
        return

    print(f"Processing {len(all_files)} file(s)...\n")

    for filepath in all_files:
        print(f"{'='*60}")
        print(f"File: {filepath}")
        print(f"{'='*60}")

        try:
            analyze_and_export(
                filepath,
                output_dir=args.output_dir,
                n_fft=args.n_fft,
                note_conf=args.note_conf,
                chord_conf=args.chord_conf,
                generate_video=args.video,
                video_res=args.video_res,
                use_cache=not args.no_cache
            )
        except Exception as e:
            print(f"✗ Error: {e}", file=sys.stderr)
            continue

        print()

    print("✓ All done!")


if __name__ == '__main__':
    main()
