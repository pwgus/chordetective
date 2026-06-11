"""Music Analyzer - Main entry point."""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Music Analyzer - Analyze audio for notes and chords',
        prog='music-analyzer'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch GUI')

    # CLI command
    cli_parser = subparsers.add_parser('analyze', help='Analyze audio file (CLI mode)')
    cli_parser.add_argument('file', help='Audio file path')
    cli_parser.add_argument('--n-fft', type=int, default=2048, help='FFT window size')
    cli_parser.add_argument('--method', choices=['pitch', 'chroma'], default='pitch')
    cli_parser.add_argument('--output-json', help='Save results to JSON')
    cli_parser.add_argument('--confidence', type=float, default=0.5)

    args = parser.parse_args()

    if args.command == 'gui' or args.command is None:
        from gui import main as gui_main
        gui_main()
    elif args.command == 'analyze':
        from cli import analyze_file
        analyze_file(args.file, n_fft=args.n_fft, method=args.method,
                     output_json=args.output_json, confidence_threshold=args.confidence)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
