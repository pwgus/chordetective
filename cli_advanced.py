"""Advanced CLI with full feature support."""

import argparse
import dataclasses
import sys
from pathlib import Path

from audio_processor import AudioProcessor
from music_analyzer import AnalysisSettings, AnalysisResult, run_full_analysis
from chord_detector import ChordDetector
from video_generator import VideoGenerator
from exporters import AnalysisExporter
from cache_manager import CacheManager
from config import Config

# Maps CLI argument names to user-config keys (persisted between runs)
CONFIG_ARG_MAP = {
    'n_fft': 'n_fft',
    'note_conf': 'note_confidence_threshold',
    'chord_conf': 'chord_confidence_threshold',
    'segmentation': 'segmentation',
    'window_size': 'window_size',
    'flux_sensitivity': 'flux_sensitivity',
    'min_note_duration': 'min_note_duration_ms',
    'median_frames': 'median_frames',
    'majority_frames': 'majority_frames',
    'hmm': 'hmm_smoothing',
    'transform': 'transform',
    'video_res': 'video_resolution',
}


def analyze_and_export(audio_path: str, settings: AnalysisSettings,
                       output_dir: str = None, generate_video: bool = False,
                       video_res: str = '720p', use_cache: bool = True,
                       show_raw: bool = False):
    """Full analysis pipeline with export and optional video generation."""

    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"✗ File not found: {audio_path}", file=sys.stderr)
        return

    output_dir = Path(output_dir or audio_path.parent / "analysis")
    output_dir.mkdir(exist_ok=True)

    # Cache check (key includes every analysis-relevant parameter)
    cache = CacheManager()
    cache_token = settings.cache_token()
    result = None
    duration = None

    if use_cache:
        cached = cache.load_analysis(str(audio_path), settings.n_fft, cache_token)
        if cached and isinstance(cached.get('result'), AnalysisResult):
            print("✓ Loaded from cache")
            result = cached['result']
            duration = cached['duration']

    if result is None:
        if use_cache:
            print("○ Cache miss, analyzing...")
        result, duration = _analyze(audio_path, settings)
        if use_cache:
            cache.save_analysis(str(audio_path), settings.n_fft, cache_token,
                                {'result': result, 'duration': duration})

    notes, chords = result.notes, result.chords

    # Export results (smoothed sequences)
    print("Exporting results...")
    metadata = {
        'settings': settings.to_dict(),
        'onsets': result.onset_times,
        'raw_note_count': len(result.raw_notes),
        'raw_chord_count': len(result.raw_chords),
    }
    json_out = output_dir / f"{audio_path.stem}_analysis.json"
    AnalysisExporter.export_json(str(json_out), str(audio_path), duration,
                                 notes, chords, metadata)
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
    audible = [n for n in notes if n.note_name != 'silence']
    real_chords = [c for c in chords if c.chord_name != 'N.C.']
    print(f"\n📊 Results:")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Notes: {len(result.raw_notes)} raw segments -> {len(audible)} smoothed notes")
    print(f"  Chords: {len(result.raw_chords)} raw segments -> {len(real_chords)} smoothed chords")
    print(f"  Onsets detected: {len(result.onset_times)}")
    if real_chords:
        chord_detector = ChordDetector()
        key, key_conf = chord_detector.get_key_estimate(real_chords)
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
                notes, chords,
                raw_note_detections=result.raw_notes if show_raw else None,
                onset_times=result.onset_times,
                resolution=video_res, n_fft=settings.n_fft,
                transform=settings.transform
            )
            print(f"  ✓ Video: {video_out}")
        except RuntimeError as e:
            print(f"  ✗ Video generation failed: {e}", file=sys.stderr)

    print(f"\n✓ All results saved to: {output_dir}")


def _analyze(audio_path: Path, settings: AnalysisSettings):
    """Perform analysis, return (result, duration)."""
    processor = AudioProcessor()
    processor.load(str(audio_path))

    print(f"Analyzing (transform={settings.transform}, "
          f"segmentation={settings.segmentation})...")
    result = run_full_analysis(processor, settings)
    print(f"  {len(result.segments)} segments, "
          f"{len(result.notes)} note events, {len(result.chords)} chord events")

    return result, processor.get_duration()


def _run_one(filepath: str, settings: AnalysisSettings, opts: dict) -> bool:
    """Analyze + export one file with a header. Returns True on success."""
    print('=' * 60)
    print(f"File: {filepath}")
    print('=' * 60)
    try:
        analyze_and_export(filepath, settings, **opts)
        return True
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return False


def _worker(task):
    """Process-pool worker: run one file, capturing its output as a string.

    Output is buffered (not streamed) so concurrent files don't interleave;
    the parent prints each block in order. tqdm bars are disabled in workers.
    """
    import io
    import os
    import contextlib

    os.environ['TQDM_DISABLE'] = '1'
    filepath, settings, opts = task
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        ok = _run_one(filepath, settings, opts)
    return filepath, buf.getvalue(), ok


def _resolve_jobs(requested, n_files: int) -> int:
    """Number of worker processes: auto (<=0/None) = min(CPUs, #files)."""
    import os
    cpu = os.cpu_count() or 1
    if requested is None or requested <= 0:
        return max(1, min(cpu, n_files))
    return max(1, min(requested, n_files))


def main():
    config = Config()

    parser = argparse.ArgumentParser(
        description='Music Analyzer - Full analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  # Basic analysis (onset segmentation + CQT + smoothing defaults)
  python cli_advanced.py song.wav

  # High quality with video showing raw vs smoothed detections
  python cli_advanced.py song.wav --video --video-res 1080p --show-raw

  # Fixed windows like the old behaviour, no smoothing
  python cli_advanced.py song.wav --segmentation fixed --window-size 2048 \\
      --median-frames 1 --min-note-duration 0

  # Adaptive windows, aggressive smoothing with HMM
  python cli_advanced.py song.wav --segmentation adaptive \\
      --flux-sensitivity 0.7 --hmm

  # Batch many files across CPU cores (auto-detect, or force with -j)
  python cli_advanced.py *.wav --video
  python cli_advanced.py *.wav -j 4
        '''
    )

    parser.add_argument('files', nargs='+', help='Audio file(s) to analyze')
    parser.add_argument('--output-dir', help='Output directory (default: same as input)')
    parser.add_argument('--n-fft', type=int, default=None,
                        help='FFT window size (default: 2048)')
    parser.add_argument('--note-conf', type=float, default=None,
                        help='Note confidence threshold 0-1 (default: 0.4)')
    parser.add_argument('--chord-conf', type=float, default=None,
                        help='Chord confidence threshold 0-1 (default: 0.3)')
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
    parser.add_argument('--video', action='store_true',
                        help='Generate MP4 video with visualization')
    parser.add_argument('--video-res', choices=['720p', '1080p'], default=None,
                        help='Video resolution (default: 720p)')
    parser.add_argument('--show-raw', action='store_true',
                        help='Also draw raw (unsmoothed) detections in the video')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching')
    parser.add_argument('-j', '--jobs', type=int, default=None,
                        help='Parallel worker processes (default: auto = '
                             'min(CPUs, #files); 1 = serial with live output)')

    args = parser.parse_args()

    # Persist explicitly given parameters; unset ones fall back to saved config
    for arg_name, cfg_key in CONFIG_ARG_MAP.items():
        value = getattr(args, arg_name)
        if value is not None:
            config.set(cfg_key, value)

    settings = AnalysisSettings.from_config(config.to_dict())
    video_res = config.get('video_resolution', '720p')

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

    opts = {
        'output_dir': args.output_dir,
        'generate_video': args.video,
        'video_res': video_res,
        'use_cache': not args.no_cache,
        'show_raw': args.show_raw,
    }
    jobs = _resolve_jobs(args.jobs, len(all_files))

    if jobs == 1:
        # Serial: stream output live (keeps tqdm video progress bars).
        print(f"Processing {len(all_files)} file(s)...\n")
        for filepath in all_files:
            _run_one(filepath, settings, opts)
            print()
    else:
        # Parallel: one process per file, output buffered and printed in order.
        from concurrent.futures import ProcessPoolExecutor

        print(f"Processing {len(all_files)} file(s) on {jobs} cores...\n")
        tasks = [(fp, settings, opts) for fp in all_files]
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            for _fp, log, _ok in executor.map(_worker, tasks):
                print(log, end='')
                print()

    print("✓ All done!")


if __name__ == '__main__':
    main()
