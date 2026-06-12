"""Music Analyzer - CLI entry point."""

import argparse

EPILOG = """
Examples:
  # Analyze a single file (notes only)
  python main.py analyze audio.wav
  python main.py analyze audio.wav --output-json results.json

  # Full pipeline (notes + chords, export, video, cache)
  python main.py batch song.wav
  python main.py batch *.wav --video --video-res 1080p
  python main.py batch audio.wav --output-dir ./results

  # Parallel batch across CPU cores (auto-detect, or force with -j)
  python main.py batch *.wav -j 4

  # Advanced analysis options
  python main.py batch song.wav --segmentation adaptive --hmm
  python main.py batch song.wav --video --show-raw
"""


def _common_options() -> argparse.ArgumentParser:
    """Analysis flags shared by both subcommands (None = use saved config)."""
    common = argparse.ArgumentParser(add_help=False)
    group = common.add_argument_group('analysis options')
    group.add_argument('--n-fft', type=int, default=None,
                       help='FFT window size (default: 2048)')
    group.add_argument('--transform', choices=['cqt', 'stft'], default=None,
                       help='Analysis transform (default: cqt)')
    group.add_argument('--segmentation', choices=['onsets', 'fixed', 'adaptive'],
                       default=None, help='Segmentation mode (default: onsets)')
    group.add_argument('--window-size', type=int, default=None,
                       help='Window size in samples, fixed mode only (default: 2048)')
    group.add_argument('--flux-sensitivity', type=float, default=None,
                       help='Transition sensitivity 0.0-1.0, adaptive mode only '
                            '(default: 0.5)')
    group.add_argument('--min-note-duration', type=int, default=None,
                       help='Discard notes shorter than this many ms (default: 100)')
    group.add_argument('--median-frames', type=int, default=None,
                       help='Median filter size in frames, 1 disables (default: 3)')
    group.add_argument('--majority-frames', type=int, default=None,
                       help='Gaussian-weighted majority vote window, 0 disables '
                            '(default: 0)')
    group.add_argument('--hmm', action=argparse.BooleanOptionalAction, default=None,
                       help='HMM (Viterbi) smoothing; slower (default: off)')
    return common


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='music-analyzer',
        description='Music Analyzer - Audio analysis for notes and chords',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG)

    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    common = _common_options()

    analyze = subparsers.add_parser(
        'analyze', parents=[common], help='Analyze a single file (notes only)')
    analyze.add_argument('file', help='Audio file path')
    analyze.add_argument('--method', choices=['pitch', 'chroma'], default=None,
                         help='Note detection method (default: pitch)')
    analyze.add_argument('--confidence', type=float, default=None,
                         help='Note confidence threshold 0-1 (default: 0.4)')
    analyze.add_argument('--output-json', help='Save results to JSON')

    batch = subparsers.add_parser(
        'batch', parents=[common],
        help='Full pipeline: notes + chords, export, video, cache')
    batch.add_argument('files', nargs='+', help='Audio file(s) to analyze')
    batch.add_argument('--output-dir',
                       help='Output directory (default: <input dir>/analysis)')
    batch.add_argument('--note-conf', type=float, default=None,
                       help='Note confidence threshold 0-1 (default: 0.4)')
    batch.add_argument('--chord-conf', type=float, default=None,
                       help='Chord confidence threshold 0-1 (default: 0.3)')
    batch.add_argument('--video', action='store_true',
                       help='Generate MP4 video with visualization')
    batch.add_argument('--video-res', choices=['720p', '1080p'], default=None,
                       help='Video resolution (default: 720p)')
    batch.add_argument('--show-raw', action='store_true',
                       help='Also draw raw (unsmoothed) detections in the video')
    batch.add_argument('--no-cache', action='store_true', help='Disable caching')
    batch.add_argument('-j', '--jobs', type=int, default=None,
                       help='Parallel worker processes (default: auto = '
                            'min(CPUs, #files); 1 = serial with live output)')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == 'analyze':
        from cli import run_analyze
        run_analyze(args)
    elif args.command == 'batch':
        from cli import run_batch
        run_batch(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
