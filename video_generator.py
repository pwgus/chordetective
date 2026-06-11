"""Video generation with spectrogram and note/chord overlay."""

import numpy as np
import librosa
import librosa.display
from pathlib import Path
from typing import List, Optional
import tempfile
import os
import subprocess
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
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr)
        duration = librosa.get_duration(y=y, sr=sr)

        # Compute spectrogram
        S, freqs, times = self._compute_spectrogram(y, n_fft)

        # Set resolution
        if resolution == '720p':
            width, height = 1280, 720
        else:  # 1080p
            width, height = 1920, 1080

        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate frames
            num_frames = int(duration * self.fps)
            for frame_idx in range(num_frames):
                time_sec = frame_idx / self.fps
                frame = self._render_frame(S, freqs, times, width, height,
                                          time_sec, note_detections,
                                          chord_detections, n_fft)

                frame_path = os.path.join(tmpdir, f'frame_{frame_idx:06d}.png')
                frame.savefig(frame_path, dpi=100, bbox_inches='tight')
                import matplotlib.pyplot as plt
                plt.close(frame)

                if (frame_idx + 1) % 30 == 0:
                    print(f"Generated {frame_idx + 1}/{num_frames} frames")

            # Combine frames with audio using ffmpeg
            self._combine_frames_with_audio(tmpdir, audio_path, output_video, width, height)

        print(f"Video saved to {output_video}")

    def _compute_spectrogram(self, y: np.ndarray, n_fft: int):
        """Compute STFT spectrogram."""
        hop_length = n_fft // 4
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        S = np.abs(D)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        times = librosa.frames_to_time(np.arange(S.shape[1]), sr=self.sr, hop_length=hop_length)
        return S, freqs, times

    def _render_frame(self, S: np.ndarray, freqs: np.ndarray, times: np.ndarray,
                     width: int, height: int, current_time: float,
                     note_detections: List[Detection],
                     chord_detections: Optional[List[ChordDetection]],
                     n_fft: int):
        """Render single frame with spectrogram and overlays."""
        from matplotlib.figure import Figure

        fig = Figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_subplot(111)

        # Draw spectrogram
        S_db = librosa.power_to_db(S**2, ref=np.max)
        im = ax.imshow(S_db, aspect='auto', origin='lower',
                      extent=[times[0], times[-1], freqs[0], freqs[-1]],
                      cmap='viridis', interpolation='nearest')

        # Vertical line at current time
        ax.axvline(current_time, color='white', linewidth=2, alpha=0.7)

        # Plot notes at current time
        current_notes = [d for d in note_detections
                        if abs(d.time - current_time) < 0.1]
        for det in current_notes:
            if det.frequency > 0:
                ax.plot(det.time, det.frequency, 'r*', markersize=15, alpha=0.8)

        # Plot chords (text annotation)
        if chord_detections:
            current_chords = [d for d in chord_detections
                            if abs(d.time - current_time) < 0.1]
            if current_chords:
                chord_text = ', '.join([f"{d.chord_name} ({d.confidence:.0f}%)"
                                       for d in current_chords])
                ax.text(0.02, 0.95, chord_text, transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                       color='white')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Music Analysis - {current_time:.2f}s')

        return fig

    def _combine_frames_with_audio(self, frames_dir: str, audio_path: str,
                                  output_video: str, width: int, height: int):
        """Use ffmpeg to combine frames and audio into video."""
        frame_pattern = os.path.join(frames_dir, 'frame_%06d.png')

        cmd = [
            'ffmpeg',
            '-framerate', str(self.fps),
            '-i', frame_pattern,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-preset', 'slow',
            '-crf', '18',
            '-c:a', 'aac',
            '-shortest',
            output_video
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except Exception as e:
            raise RuntimeError(f"FFmpeg error: {e}")
