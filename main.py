"""Music Analyzer - CLI entry point."""

import sys
import argparse


def _add_analysis_flags(parser):
    """Shared segmentation/smoothing flags (None = use saved config)."""
    parser.add_argument('--n-fft', type=int, default=None, help='FFT window size')
    parser.add_argument('--segmentation', choices=['onsets', 'fixed', 'adaptive'],
                        default=None, help='Segmentation mode (default: onsets)')
    parser.add_argument('--window-size', type=int, default=None,
                        help='Window size in samples (fixed mode only)')
    parser.add_argument('--flux-sensitivity', type=float, default=None,
                        help='Transition sensitivity 0.0-1.0 (adaptive mode only)')
    parser.add_argument('--min-note-duration', type=int, default=None,
                        help='Discard notes shorter than this many ms (default: 100)')
    parser.add_argument('--median-frames', type=int, default=None,
                        help='Median filter frames, 1 disables (default: 3)')
    parser.add_argument('--majority-frames', type=int, default=None,
                        help='Majority vote window, 0 disables (default: 0)')
    parser.add_argument('--hmm', action=argparse.BooleanOptionalAction, default=None,
                        help='HMM (Viterbi) smoothing; slower')
    parser.add_argument('--transform', choices=['cqt', 'stft'], default=None,
                        help='Analysis transform (default: cqt)')


def _forward_analysis_flags(args, argv):
    """Append shared flags to a rebuilt argv when explicitly given."""
    for flag, attr in [('--n-fft', 'n_fft'), ('--segmentation', 'segmentation'),
                       ('--window-size', 'window_size'),
                       ('--flux-sensitivity', 'flux_sensitivity'),
                       ('--min-note-duration', 'min_note_duration'),
                       ('--median-frames', 'median_frames'),
                       ('--majority-frames', 'majority_frames'),
                       ('--transform', 'transform')]:
        value = getattr(args, attr)
        if value is not None:
            argv.extend([flag, str(value)])
    if args.hmm is not None:
        argv.append('--hmm' if args.hmm else '--no-hmm')


def main():
    parser = argparse.ArgumentParser(
        description='Music Analyzer - Audio analysis for notes and chords',
        prog='music-analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Analyze single file (notes only)
  python main.py analyze audio.wav
  python main.py analyze audio.wav --output-json results.json

  # Full pipeline (notes + chords, batch, video, cache)
  python main.py batch song.wav
  python main.py batch *.wav --video --video-res 1080p
  python main.py batch audio.wav --output-dir ./results

  # Advanced options
  python main.py batch song.wav --segmentation adaptive --hmm
  python main.py batch song.wav --video --show-raw
        '''
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Simple analyze command
    simple_parser = subparsers.add_parser('analyze', help='Analyze single file (notes only)')
    simple_parser.add_argument('file', help='Audio file path')
    simple_parser.add_argument('--method', choices=['pitch', 'chroma'], default=None)
    simple_parser.add_argument('--output-json', help='Save results to JSON')
    simple_parser.add_argument('--confidence', type=float, default=None)
    _add_analysis_flags(simple_parser)

    # Advanced batch command
    batch_parser = subparsers.add_parser('batch', help='Analyze files with all features')
    batch_parser.add_argument('files', nargs='+', help='Audio file(s) to analyze')
    batch_parser.add_argument('--output-dir', help='Output directory')
    batch_parser.add_argument('--note-conf', type=float, default=None)
    batch_parser.add_argument('--chord-conf', type=float, default=None)
    batch_parser.add_argument('--video', action='store_true', help='Generate MP4')
    batch_parser.add_argument('--video-res', choices=['720p', '1080p'], default=None)
    batch_parser.add_argument('--show-raw', action='store_true',
                              help='Draw raw detections in the video too')
    batch_parser.add_argument('--no-cache', action='store_true')
    _add_analysis_flags(batch_parser)

    args = parser.parse_args()

    if args.command == 'analyze':
        from cli import main as cli_main
        argv = [sys.argv[0], args.file]
        if args.method is not None:
            argv.extend(['--method', args.method])
        if args.output_json:
            argv.extend(['--output-json', args.output_json])
        if args.confidence is not None:
            argv.extend(['--confidence', str(args.confidence)])
        _forward_analysis_flags(args, argv)
        sys.argv = argv
        cli_main()

    elif args.command == 'batch':
        from cli_advanced import main as batch_main
        argv = [sys.argv[0]] + args.files
        if args.output_dir:
            argv.extend(['--output-dir', args.output_dir])
        if args.note_conf is not None:
            argv.extend(['--note-conf', str(args.note_conf)])
        if args.chord_conf is not None:
            argv.extend(['--chord-conf', str(args.chord_conf)])
        if args.video:
            argv.append('--video')
        if args.video_res is not None:
            argv.extend(['--video-res', args.video_res])
        if args.show_raw:
            argv.append('--show-raw')
        if args.no_cache:
            argv.append('--no-cache')
        _forward_analysis_flags(args, argv)
        sys.argv = argv
        batch_main()

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
