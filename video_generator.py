"""Video generation with spectrogram and note/chord overlay."""

import numpy as np
import librosa
from pathlib import Path
from typing import List, Optional
import tempfile
import os
import subprocess
from tqdm import tqdm
from music_analyzer import Detection
from chord_detector import ChordDetection


class VideoGenerator:
    """Generate MP4 video with spectrogram and analysis overlay."""

    def __init__(self, sr: int = 22050, fps: int = 30):
        self.sr = sr
        self.fps = fps

    def generate_from_analysis(self, audio_path: str, output_video: str,
                              note_detections: List[Detection],
                              chord_detections: Optional[List[ChordDetection]] = None,
                              resolution: str = '720p', n_fft: int = 2048):
        """
        Generate video with audio, spectrogram, and detections overlay.

        Args:
            audio_path: Input audio file
            output_video: Output MP4 path
            note_detections: List of note detections
            chord_detections: List of chord detections (optional)
            resolution: '720p' or '1080p'
            n_fft: FFT window size
        """
        print(f"Loading audio: {audio_path}")
        y, sr = librosa.load(audio_path, sr=self.sr)
        duration = librosa.get_duration(y=y, sr=sr)

        print(f"Computing spectrogram...")
        S, freqs, times = self._compute_spectrogram(y, n_fft)
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
                    os.path.join(tmpdir, f'frame_{frame_idx:06d}.png'),
                    n_fft
                )

            print("Combining with audio...")
            self._combine_frames_with_audio(tmpdir, audio_path, output_video, width, height)

        print(f"✓ Video saved: {output_video}")

    def _compute_spectrogram(self, y: np.ndarray, n_fft: int):
        """Compute STFT spectrogram."""
        hop_length = n_fft // 4
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        S = np.abs(D)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        times = librosa.frames_to_time(np.arange(S.shape[1]), sr=self.sr, hop_length=hop_length)
        return S, freqs, times

    def _render_frame_to_file(self, S_db: np.ndarray, freqs: np.ndarray, times: np.ndarray,
                              width: int, height: int, current_time: float,
                              note_detections: List[Detection],
                              chord_detections: Optional[List[ChordDetection]],
                              output_path: str, n_fft: int):
        """Render frame directly to file."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        dpi = 100
        fig = Figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        ax = fig.add_subplot(111)

        # Spectrogram
        im = ax.imshow(S_db, aspect='auto', origin='lower',
                      extent=[times[0], times[-1], freqs[0], freqs[-1]],
                      cmap='viridis', interpolation='nearest')

        # Current time cursor
        ax.axvline(current_time, color='white', linewidth=3, alpha=0.9)

        # Nearby notes
        window = 0.05
        nearby_notes = [d for d in note_detections
                       if abs(d.time - current_time) < window]
        for det in nearby_notes:
            if det.frequency > 0:
                ax.plot(det.time, det.frequency, 'r*', markersize=20, alpha=0.9)

        # Chord annotation
        if chord_detections:
            nearby_chords = [d for d in chord_detections
                           if abs(d.time - current_time) < window]
            if nearby_chords:
                chord_str = ' | '.join([f"{d.chord_name} {d.confidence:.0f}%"
                                       for d in nearby_chords])
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
        ax.set_title(f'Music Analysis Visualization', fontsize=14, weight='bold')

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
