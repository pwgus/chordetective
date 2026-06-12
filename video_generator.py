"""Video generation with spectrogram and note/chord overlay."""

import numpy as np
import librosa
from typing import List, Optional
import tempfile
import os
import subprocess
from tqdm import tqdm

from audio_processor import AudioProcessor
from music_analyzer import Detection
from chord_detector import ChordDetection


def _active_at(detections, current_time: float, fallback_window: float = 0.05):
    """Detections whose span covers `current_time` (legacy ones use a window)."""
    active = []
    for d in detections:
        duration = getattr(d, 'duration', 0.0)
        if duration > 0:
            if d.time <= current_time < d.time + duration:
                active.append(d)
        elif abs(d.time - current_time) < fallback_window:
            active.append(d)
    return active


class VideoGenerator:
    """Generate MP4 video with spectrogram and analysis overlay."""

    def __init__(self, sr: int = 22050, fps: int = 30):
        self.sr = sr
        self.fps = fps

    def generate_from_analysis(self, audio_path: str, output_video: str,
                              note_detections: List[Detection],
                              chord_detections: Optional[List[ChordDetection]] = None,
                              raw_note_detections: Optional[List[Detection]] = None,
                              onset_times: Optional[List[float]] = None,
                              resolution: str = '720p', n_fft: int = 2048,
                              transform: str = 'cqt'):
        """
        Generate video with audio, spectrogram, and detections overlay.

        On-screen annotations always use the smoothed sequences
        (`note_detections` / `chord_detections`). Pass `raw_note_detections`
        to additionally draw the unsmoothed detections in a faint color, and
        `onset_times` to mark detected onsets on the timeline.

        Args:
            audio_path: Input audio file
            output_video: Output MP4 path
            note_detections: Smoothed note sequence
            chord_detections: Smoothed chord sequence (optional)
            raw_note_detections: Raw note detections (optional, faint overlay)
            onset_times: Detected onset times in seconds (optional)
            resolution: '720p' or '1080p'
            n_fft: FFT window size (STFT) / hop source (CQT)
            transform: 'cqt' or 'stft' background spectrogram
        """
        print(f"Loading audio: {audio_path}")
        processor = AudioProcessor(sr=self.sr)
        processor.load(audio_path)
        duration = processor.get_duration()

        print(f"Computing spectrogram ({transform})...")
        S, freqs, times = processor.compute_transform(transform=transform, n_fft=n_fft)
        S_db = librosa.power_to_db(S**2, ref=np.max)

        # Set resolution
        if resolution == '720p':
            width, height = 1280, 720
        else:
            width, height = 1920, 1080

        print(f"Generating {int(duration * self.fps)} frames...")
        with tempfile.TemporaryDirectory() as tmpdir:
            num_frames = int(duration * self.fps)

            for frame_idx in tqdm(range(num_frames), desc="Rendering frames"):
                time_sec = frame_idx / self.fps

                self._render_frame_to_file(
                    S_db, freqs, times, width, height,
                    time_sec, note_detections, chord_detections,
                    raw_note_detections, onset_times,
                    os.path.join(tmpdir, f'frame_{frame_idx:06d}.png'),
                    transform
                )

            print("Combining with audio...")
            self._combine_frames_with_audio(tmpdir, audio_path, output_video, width, height)

        print(f"✓ Video saved: {output_video}")

    def _render_frame_to_file(self, S_db: np.ndarray, freqs: np.ndarray, times: np.ndarray,
                              width: int, height: int, current_time: float,
                              note_detections: List[Detection],
                              chord_detections: Optional[List[ChordDetection]],
                              raw_note_detections: Optional[List[Detection]],
                              onset_times: Optional[List[float]],
                              output_path: str, transform: str):
        """Render frame directly to file."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.transforms as mtransforms
        from matplotlib.figure import Figure

        dpi = 100
        fig = Figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        ax = fig.add_subplot(111)

        # Spectrogram (log frequency axis for CQT)
        if transform == 'cqt':
            ax.pcolormesh(times, freqs, S_db, cmap='viridis', shading='auto')
            ax.set_yscale('log')
            ax.set_ylim(freqs[0], freqs[-1])
        else:
            ax.imshow(S_db, aspect='auto', origin='lower',
                      extent=[times[0], times[-1], freqs[0], freqs[-1]],
                      cmap='viridis', interpolation='nearest')

        # Onset markers on the timeline (small ticks at the bottom)
        if onset_times is not None and len(onset_times) > 0:
            trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.vlines(onset_times, 0.0, 0.035, transform=trans,
                      color='cyan', linewidth=1, alpha=0.8)

        # Raw (unsmoothed) detections in a faint color
        if raw_note_detections:
            raw_pts = [(d.time, d.frequency) for d in raw_note_detections if d.frequency > 0]
            if raw_pts:
                rx, ry = zip(*raw_pts)
                ax.scatter(rx, ry, s=10, color='lightgray', alpha=0.35, zorder=3)

        # Smoothed note spans
        for det in note_detections:
            if det.frequency > 0 and det.duration > 0:
                ax.hlines(det.frequency, det.time, det.time + det.duration,
                          colors='orange', linewidth=3, alpha=0.85, zorder=4)

        # Current time cursor
        ax.axvline(current_time, color='white', linewidth=3, alpha=0.9)

        # Active smoothed notes
        active_notes = [d for d in _active_at(note_detections, current_time)
                        if d.note_name != 'silence']
        for det in active_notes:
            if det.frequency > 0:
                ax.plot(current_time, det.frequency, 'r*', markersize=20,
                        alpha=0.9, zorder=5)
        if active_notes:
            note_str = ' | '.join(f"♪ {d.note_name}" for d in active_notes)
            ax.text(0.02, 0.90, note_str, transform=ax.transAxes,
                    fontsize=13, verticalalignment='top', weight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                    color='orange')

        # Active smoothed chord annotation
        if chord_detections:
            active_chords = _active_at(chord_detections, current_time)
            if active_chords:
                chord_str = ' | '.join([f"{d.chord_name} {d.confidence:.0f}%"
                                       for d in active_chords])
                ax.text(0.02, 0.98, chord_str, transform=ax.transAxes,
                       fontsize=14, verticalalignment='top', weight='bold',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                       color='white')

        # Time indicator
        min_t = int(current_time) // 60
        sec_t = int(current_time) % 60
        ax.text(0.98, 0.02, f"{min_t:02d}:{sec_t:02d}", transform=ax.transAxes,
               fontsize=12, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
               color='white')

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        ax.set_title('Music Analysis Visualization', fontsize=14, weight='bold')

        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def _combine_frames_with_audio(self, frames_dir: str, audio_path: str,
                                  output_video: str, width: int, height: int):
        """Use ffmpeg to combine frames and audio into video."""
        frame_pattern = os.path.join(frames_dir, 'frame_%06d.png')

        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(self.fps),
            '-i', frame_pattern,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '20',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-shortest',
            output_video
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg error: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Install it: brew install ffmpeg (macOS) or sudo apt install ffmpeg (Linux)")
