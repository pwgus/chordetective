"""Implementations of the CLI subcommands: `analyze` and `batch`.

`main.py` parses arguments (see build_parser there) and dispatches to
run_analyze / run_batch with the parsed namespace.
"""

import contextlib
import dataclasses
import glob
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from audio_processor import AudioProcessor
from cache_manager import CacheManager
from chord_detector import ChordDetector
from config import Config
from exporters import AnalysisExporter
from music_analyzer import AnalysisResult, AnalysisSettings, run_full_analysis
from video_generator import VideoGenerator

# Maps CLI argument names to user-config keys (persisted between runs).
# `confidence` (analyze) and `note_conf` (batch) share one config key; each
# subcommand only carries its own attributes, the rest resolve to None.
CONFIG_ARG_MAP = {
    'n_fft': 'n_fft',
    'method': 'note_method',
    'confidence': 'note_confidence_threshold',
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


def _load_settings(args):
    """Persist explicitly given args into config; build settings from it.

    Unset arguments (None) fall back to the last saved value, so preferences
    carry over between runs. Returns (settings, config).
    """
    config = Config()
    config.update({key: getattr(args, name)
                   for name, key in CONFIG_ARG_MAP.items()
                   if getattr(args, name, None) is not None})
    return AnalysisSettings.from_config(config.to_dict()), config


# ---------------------------------------------------------------------------
# analyze: single file, notes only
# ---------------------------------------------------------------------------

def analyze_file(filepath: str, settings: AnalysisSettings,
                 output_json: str = None) -> AnalysisResult:
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


def run_analyze(args):
    """Entry point for the `analyze` subcommand."""
    settings, _ = _load_settings(args)
    settings.detect_chords = False  # analyze reports notes only

    try:
        analyze_file(args.file, settings, output_json=args.output_json)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# batch: full pipeline (notes + chords, export, video, cache, multicore)
# ---------------------------------------------------------------------------

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
        cached = cache.load(str(audio_path), cache_token)
        if cached and isinstance(cached.get('result'), AnalysisResult):
            print("✓ Loaded from cache")
            result = cached['result']
            duration = cached['duration']

    if result is None:
        if use_cache:
            print("○ Cache miss, analyzing...")
        result, duration = _analyze(audio_path, settings)
        if use_cache:
            cache.save(str(audio_path), cache_token,
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
        key, key_conf = ChordDetector().get_key_estimate(real_chords)
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


def _worker_init():
    """Runs once in each worker process: silence tqdm progress bars."""
    os.environ['TQDM_DISABLE'] = '1'


def _worker(task):
    """Process-pool worker: run one file, capturing its output as a string.

    Output is buffered (not streamed) so concurrent files don't interleave;
    the parent prints each block in order.
    """
    filepath, settings, opts = task
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        ok = _run_one(filepath, settings, opts)
    return filepath, buf.getvalue(), ok


def _resolve_jobs(requested, n_files: int) -> int:
    """Number of worker processes: auto (<=0/None) = min(CPUs, #files)."""
    cpu = os.cpu_count() or 1
    if requested is None or requested <= 0:
        return max(1, min(cpu, n_files))
    return max(1, min(requested, n_files))


def _expand_globs(patterns):
    """Expand wildcard patterns; non-matching patterns pass through as-is."""
    files = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        files.extend(matches if matches else [pattern])
    return files


def run_batch(args):
    """Entry point for the `batch` subcommand."""
    settings, config = _load_settings(args)
    video_res = config.get('video_resolution', '720p')

    all_files = _expand_globs(args.files)
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
        print(f"Processing {len(all_files)} file(s) on {jobs} cores...\n")
        tasks = [(fp, settings, opts) for fp in all_files]
        with ProcessPoolExecutor(max_workers=jobs,
                                 initializer=_worker_init) as executor:
            for _fp, log, _ok in executor.map(_worker, tasks):
                print(log, end='')
                print()

    print("✓ All done!")
