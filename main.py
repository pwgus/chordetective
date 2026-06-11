"""Music Analyzer - Main entry point with GUI and CLI modes."""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Music Analyzer - Audio analysis for notes and chords',
        prog='music-analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Usage modes:
  GUI (default):
    python main.py gui
    python main.py gui --advanced          # with playback controls

  CLI - Simple:
    python main.py analyze audio.wav
    python main.py analyze audio.wav --output-json results.json

  CLI - Advanced (batch, video, cache):
    python main.py batch *.wav --video --video-res 1080p
    python main.py batch audio.wav --output-dir ./results

Quick start:
  python main.py gui                        # Launch GUI
  python main.py batch song.wav --video    # Analyze + generate video
        '''
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch graphical interface')
    gui_parser.add_argument('--advanced', action='store_true',
                           help='Use advanced GUI with playback and timeline')

    # Simple analyze command
    simple_parser = subparsers.add_parser('analyze', help='Analyze single file (simple)')
    simple_parser.add_argument('file', help='Audio file path')
    simple_parser.add_argument('--n-fft', type=int, default=2048, help='FFT window size')
    simple_parser.add_argument('--method', choices=['pitch', 'chroma'], default='pitch')
    simple_parser.add_argument('--output-json', help='Save results to JSON')
    simple_parser.add_argument('--confidence', type=float, default=0.5)

    # Advanced batch command
    batch_parser = subparsers.add_parser('batch', help='Analyze files with all features')
    batch_parser.add_argument('files', nargs='+', help='Audio file(s) to analyze')
    batch_parser.add_argument('--output-dir', help='Output directory')
    batch_parser.add_argument('--n-fft', type=int, default=2048)
    batch_parser.add_argument('--note-conf', type=float, default=0.4)
    batch_parser.add_argument('--chord-conf', type=float, default=0.3)
    batch_parser.add_argument('--video', action='store_true', help='Generate MP4')
    batch_parser.add_argument('--video-res', choices=['720p', '1080p'], default='720p')
    batch_parser.add_argument('--no-cache', action='store_true')

    args = parser.parse_args()

    if args.command == 'gui' or args.command is None:
        if hasattr(args, 'advanced') and args.advanced:
            try:
                from gui_advanced import main as gui_main
                gui_main()
            except ImportError:
                print("Error: PyQt5 or audio libraries not installed")
                print("Install: pip install -r requirements.txt")
        else:
            try:
                from gui import main as gui_main
                gui_main()
            except ImportError:
                print("Error: PyQt5 not installed")
                print("Install: pip install -r requirements.txt")

    elif args.command == 'analyze':
        from cli import analyze_file
        analyze_file(args.file, n_fft=args.n_fft, method=args.method,
                     output_json=args.output_json, confidence_threshold=args.confidence)

    elif args.command == 'batch':
        from cli_advanced import main as batch_main
        sys.argv = [sys.argv[0]] + args.files
        if args.output_dir:
            sys.argv.extend(['--output-dir', args.output_dir])
        if args.n_fft != 2048:
            sys.argv.extend(['--n-fft', str(args.n_fft)])
        if args.note_conf != 0.4:
            sys.argv.extend(['--note-conf', str(args.note_conf)])
        if args.chord_conf != 0.3:
            sys.argv.extend(['--chord-conf', str(args.chord_conf)])
        if args.video:
            sys.argv.append('--video')
        if args.video_res != '720p':
            sys.argv.extend(['--video-res', args.video_res])
        if args.no_cache:
            sys.argv.append('--no-cache')
        batch_main()

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
