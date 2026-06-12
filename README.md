# chordetective

CLI tool to analyze audio and extract notes and chords, with sequence smoothing, export (JSON/CSV), video generation and multicore processing.

## Features

✓ Audio file loading (WAV, MP3, FLAC, OGG, M4A)
✓ Note detection via pitch tracking (PYIN) or chroma
✓ Chord detection from chroma features (CQT or STFT)
✓ Onset, fixed-window or adaptive segmentation
✓ Sequence smoothing (median filter, majority vote, HMM/Viterbi)
✓ Key estimation
✓ Result export (JSON, notes/chords/combined CSV)
✓ Video generation with synchronized analysis
✓ Result caching per file and parameters
✓ Multicore batch processing

## Installation

```bash
pip install -r requirements.txt
```

Requires `ffmpeg` installed on the system for video generation:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Arch
sudo pacman -S ffmpeg
```

## Usage

The app exposes two subcommands: `analyze` (fast, notes only) and `batch` (full
pipeline: notes + chords + export + video + cache + multicore).

```bash
python main.py            # show help
python main.py analyze --help
python main.py batch --help
```

### `analyze` — Simple analysis (notes only)

```bash
python main.py analyze audio.wav
python main.py analyze audio.wav \
  --method pitch \
  --confidence 0.5 \
  --output-json results.json
```

- `--method`: `pitch` (pitch tracking) or `chroma` (pitch class features)
- `--confidence`: Minimum confidence threshold (0-1)
- `--output-json`: Save results to JSON
- Also accepts the common analysis options (see below).

### `batch` — Full pipeline (notes + chords)

```bash
# Single file
python main.py batch song.wav

# Multiple files, with 1080p video
python main.py batch *.wav --video --video-res 1080p

# Custom output directory
python main.py batch audio.wav --output-dir ./results
```

**`batch`-specific options:**
- `--output-dir`: Output folder (default: `<audio_folder>/analysis`)
- `--note-conf`: Note confidence threshold 0-1 (default: 0.4)
- `--chord-conf`: Chord confidence threshold 0-1 (default: 0.3)
- `--video`: Generate MP4 with the visualization
- `--video-res`: `720p` or `1080p` (default: 720p)
- `--show-raw`: Also draw the raw (unsmoothed) detections in the video
- `--no-cache`: Disable caching
- `-j`, `--jobs`: Parallel processes (see [Multicore processing](#multicore-processing))

### Common analysis options

Available in both `analyze` and `batch`. If not given, the last value saved in
`.music_analyzer.json` is used.

- `--n-fft`: FFT window size (default: 2048)
- `--transform`: `cqt` or `stft` (default: cqt)
- `--segmentation`: `onsets`, `fixed` or `adaptive` (default: onsets)
- `--window-size`: Window size in samples, `fixed` mode only (default: 2048)
- `--flux-sensitivity`: Transition sensitivity 0.0-1.0, `adaptive` mode only (default: 0.5)
- `--min-note-duration`: Discard notes shorter than this many ms (default: 100)
- `--median-frames`: Median filter in frames, 1 disables it (default: 3)
- `--majority-frames`: Majority vote window, 0 disables it (default: 0)
- `--hmm` / `--no-hmm`: HMM (Viterbi) smoothing; slower (default: off)

**Advanced examples:**
```bash
# Fixed windows (classic behavior), no smoothing
python main.py batch song.wav --segmentation fixed --window-size 2048 \
  --median-frames 1 --min-note-duration 0

# Adaptive windows with HMM smoothing
python main.py batch song.wav --segmentation adaptive \
  --flux-sensitivity 0.7 --hmm
```

## Multicore processing

`batch` processes each file in its own process, spreading the load across the CPU
cores. Parallelism is at the file level (each analysis is CPU-bound: PYIN,
CQT/STFT), so it sidesteps Python's GIL.

```bash
# Auto: uses min(cores, #files)
python main.py batch *.wav

# Force 4 processes
python main.py batch *.wav -j 4

# Serial, with live video progress bars
python main.py batch *.wav -j 1
```

- `-j`/`--jobs` defaults to automatic: `min(available cores, #files)`.
- With a single file, or with `-j 1`, it runs serially and the output (including the
  `tqdm` video bars) is shown live.
- In parallel, each file's output is buffered and printed in order, without
  interleaving.
- `-j` is a runtime option; it is not persisted to the configuration.

## Output

For each input file `<name>`, `batch` generates, in the output directory:

```
<name>_analysis.json     - Full analysis (notes + chords + metadata)
<name>_notes.csv         - Smoothed notes
<name>_chords.csv        - Smoothed chords
<name>_combined.csv      - Notes and chords aligned by time
<name>_analysis.mp4      - Video (only with --video)
```

### JSON
```json
{
  "audio_file": "song.wav",
  "duration": 120.5,
  "metadata": { "settings": { "...": "..." }, "onsets": [0.5, 1.2] },
  "notes": [
    {"time": 0.5, "duration": 0.3, "name": "C4", "frequency": 261.6, "confidence": 85.3}
  ],
  "chords": [
    {"time": 0.0, "duration": 2.0, "name": "Cmaj", "confidence": 78.5}
  ]
}
```

### CSV (Notes)
```
Time (s),Duration (s),Note,Frequency (Hz),Confidence (%)
0.500,0.300,C4,261.6,85.3
```

### CSV (Chords)
```
Time (s),Duration (s),Chord,Confidence (%)
0.000,2.000,Cmaj,78.5
```

### CSV (Combined)
```
Time (s),Notes,Chords
0.00,C4 (85%),Cmaj (79%)
```

## Architecture

```
audio_processor.py      - Audio loading, segmentation and shared signal helpers
notes.py                - Note names and pitch conversions (MIDI convention)
music_analyzer.py       - Note detection and pipeline (run_full_analysis)
chord_detector.py       - Chord detection and key estimation
smoother.py             - Sequence smoothing (median, vote, HMM)
video_generator.py      - MP4 video generation
exporters.py            - JSON/CSV export
cache_manager.py        - Analysis cache per file + parameters
config.py               - Persistent configuration (.music_analyzer.json)
cli.py                  - Subcommand implementations (analyze + batch/multicore)
main.py                 - Entry point (argparse, dispatch)
```

## Parameters Explained

### n_fft (FFT Window Size)
- **Effect:** Larger windows = better frequency resolution but worse temporal
- **Default:** 2048
- **Recommendation:**
  - Low-frequency music: 4096
  - High-frequency music: 1024
  - Balanced: 2048

### transform
- **cqt:** Constant-Q, better for music (logarithmic frequency resolution)
- **stft:** Short-Time Fourier Transform, linear resolution

### segmentation
- **onsets:** Segments at detected attacks (default)
- **fixed:** Fixed-size windows (`--window-size`)
- **adaptive:** Boundaries from spectral flux (`--flux-sensitivity`)

### confidence_threshold
- **Range:** 0.0 - 1.0
- **Effect:** Only shows detections with confidence above this value
- **Default:** 0.4 (notes), 0.3 (chords)

### method (note detection, `analyze` only)
- **pitch:** PYIN for fundamental pitch. Exact frequency and octave; slower.
- **chroma:** Pitch class (relative per-note energy). Faster; no octave.

### Smoothing
- **min-note-duration:** Removes very short spurious notes
- **median-frames:** Median filter over the label sequence
- **majority-frames:** Gaussian-weighted majority vote
- **hmm:** Viterbi decoding for the most likely sequence (slower)

## Performance Notes

- **~3min audio at 22050Hz:** ~10-20s (pitch), ~5-10s (chroma)
- **Video generation:** ~1-2 min per minute of audio
- **Multicore batch:** throughput scales ~linearly with the number of cores up to
  the number of files

## Troubleshooting

### Error: "librosa not found"
```bash
pip install librosa
```

### Error: "FFmpeg not found" (when generating video)
Install ffmpeg (see Installation section)

### Slow analysis
- Reduce `n_fft` (2048 → 1024)
- Use `--method chroma` instead of `pitch` (in `analyze`)
- Raise the confidence thresholds to filter noise
- In batch, leave parallelism on auto or raise `-j`

### Wrong detections
- Try different `n_fft` values
- Raise the confidence thresholds to remove false positives
- Tune smoothing (`--median-frames`, `--hmm`)
- Make sure the audio is mono and clean

## Development

Modular structure allows adding:
- New interfaces (Web, API)
- New detection methods (ML-based)
- Additional analysis (tempo, score)
- Export formats (MIDI, notation)

## License

MIT
