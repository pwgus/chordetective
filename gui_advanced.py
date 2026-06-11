"""Advanced PyQt5 GUI with playback controls and timeline annotations."""

import sys
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog, QProgressBar, QComboBox,
    QSpinBox, QDoubleSpinBox, QTabWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QTime
from PyQt5.QtGui import QFont
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from audio_processor import AudioProcessor
from music_analyzer import NoteDetector
from chord_detector import ChordDetector
from exporters import AnalysisExporter


class AnalysisWorker(QThread):
    """Background thread for audio analysis."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)  # Contains notes, chords, key
    error = pyqtSignal(str)

    def __init__(self, y: np.ndarray, sr: int, n_fft: int, detect_notes: bool = True,
                 detect_chords: bool = True):
        super().__init__()
        self.y = y
        self.sr = sr
        self.n_fft = n_fft
        self.detect_notes = detect_notes
        self.detect_chords = detect_chords

    def run(self):
        try:
            result = {}

            if self.detect_notes:
                detector = NoteDetector(sr=self.sr)
                notes = detector.detect_notes_from_pitch(
                    self.y, n_fft=self.n_fft, confidence_threshold=0.4
                )
                result['notes'] = notes
                self.progress.emit(50)

            if self.detect_chords:
                chord_detector = ChordDetector(sr=self.sr)
                chords = chord_detector.detect_chords(
                    self.y, n_fft=self.n_fft, confidence_threshold=0.3, smooth=True
                )
                result['chords'] = chords
                key, key_conf = chord_detector.get_key_estimate(chords)
                result['key'] = key
                result['key_confidence'] = key_conf
                self.progress.emit(100)

            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MusicAnalyzerAdvancedGUI(QMainWindow):
    """Advanced GUI with playback and annotations."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Music Analyzer - Advanced")
        self.setGeometry(100, 100, 1400, 900)

        self.processor = None
        self.detections = {'notes': [], 'chords': [], 'key': None}
        self.analysis_worker = None
        self.current_time = 0.0
        self.is_playing = False

        # Audio playback
        try:
            import sounddevice
            import soundfile
            self.audio_available = True
        except ImportError:
            self.audio_available = False

        self.init_ui()

    def init_ui(self):
        """Initialize UI components."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # File selection bar
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file loaded")
        self.file_label.setFont(QFont("Arial", 10))
        file_btn = QPushButton("Open Audio File")
        file_btn.clicked.connect(self.open_file)
        file_layout.addWidget(QLabel("File:"))
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(file_btn)
        file_layout.addStretch()
        layout.addLayout(file_layout)

        # Controls bar
        control_layout = QHBoxLayout()

        # FFT size
        control_layout.addWidget(QLabel("Window (n_fft):"))
        self.n_fft_slider = QSlider(Qt.Horizontal)
        self.n_fft_slider.setMinimum(9)
        self.n_fft_slider.setMaximum(12)
        self.n_fft_slider.setValue(11)
        self.n_fft_slider.sliderMoved.connect(self.on_nfft_changed)
        self.n_fft_label = QLabel("2048")
        control_layout.addWidget(self.n_fft_slider)
        control_layout.addWidget(self.n_fft_label)
        control_layout.addSpacing(20)

        # Analysis options
        control_layout.addWidget(QLabel("Analyze:"))
        self.detect_notes_combo = QComboBox()
        self.detect_notes_combo.addItems(["Both", "Notes Only", "Chords Only"])
        control_layout.addWidget(self.detect_notes_combo)
        control_layout.addSpacing(20)

        # Buttons
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.analyze)
        self.analyze_btn.setEnabled(False)
        control_layout.addWidget(self.analyze_btn)

        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        control_layout.addWidget(self.export_btn)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        # Playback controls
        playback_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        playback_layout.addWidget(self.play_btn)

        self.time_label = QLabel("00:00 / 00:00")
        playback_layout.addWidget(self.time_label)

        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.sliderMoved.connect(self.on_timeline_seek)
        playback_layout.addWidget(self.timeline_slider)

        layout.addLayout(playback_layout)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Tabs for different views
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Spectrogram tab
        self.spec_figure = Figure(figsize=(12, 5), dpi=100)
        self.spec_canvas = FigureCanvas(self.spec_figure)
        self.tabs.addTab(self.spec_canvas, "Spectrogram")

        # Timeline tab
        self.timeline_figure = Figure(figsize=(12, 4), dpi=100)
        self.timeline_canvas = FigureCanvas(self.timeline_figure)
        self.tabs.addTab(self.timeline_canvas, "Timeline")

        # Status bar
        self.statusBar().showMessage("Ready")

    def open_file(self):
        """Open audio file dialog."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*)"
        )
        if filepath:
            self.load_file(filepath)

    def load_file(self, filepath: str):
        """Load audio file."""
        try:
            self.processor = AudioProcessor()
            self.processor.load(filepath)
            self.file_label.setText(Path(filepath).name)
            self.analyze_btn.setEnabled(True)
            self.play_btn.setEnabled(self.audio_available)
            self.timeline_slider.setMaximum(int(self.processor.get_duration() * 100))
            self.draw_spectrogram()
            self.statusBar().showMessage(f"Loaded: {Path(filepath).name}")
        except Exception as e:
            self.statusBar().showMessage(f"Error: {e}")

    def on_nfft_changed(self, value):
        """Update n_fft label."""
        n_fft = 2 ** value
        self.n_fft_label.setText(str(n_fft))

    def draw_spectrogram(self):
        """Draw spectrogram."""
        if self.processor is None or self.processor.y is None:
            return

        n_fft = 2 ** self.n_fft_slider.value()

        try:
            S, freqs, times = self.processor.compute_spectrogram(n_fft=n_fft)
            S_db = librosa.power_to_db(S**2, ref=np.max)

            self.spec_figure.clear()
            ax = self.spec_figure.add_subplot(111)

            im = ax.imshow(S_db, aspect='auto', origin='lower',
                          extent=[times[0], times[-1], freqs[0], freqs[-1]],
                          cmap='viridis', interpolation='nearest')

            # Overlay notes
            for note in self.detections.get('notes', []):
                if note.frequency > 0:
                    ax.plot(note.time, note.frequency, 'r.', markersize=6, alpha=0.6)

            # Vertical line for current playback position
            ax.axvline(self.current_time, color='white', linewidth=1, alpha=0.5)

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'Spectrogram (n_fft={n_fft})')
            self.spec_figure.colorbar(im, ax=ax, label='dB')

            self.spec_canvas.draw()
        except Exception as e:
            self.statusBar().showMessage(f"Draw error: {e}")

    def draw_timeline(self):
        """Draw timeline with chord annotations."""
        notes = self.detections.get('notes', [])
        chords = self.detections.get('chords', [])

        if not chords and not notes:
            return

        self.timeline_figure.clear()
        ax = self.timeline_figure.add_subplot(111)

        # Draw chord regions
        if chords:
            y_pos = 1
            for i, chord in enumerate(chords):
                next_time = chords[i+1].time if i+1 < len(chords) else self.processor.get_duration()
                width = next_time - chord.time
                rect = Rectangle((chord.time, y_pos - 0.4), width, 0.8,
                               edgecolor='black', facecolor='skyblue', alpha=0.7)
                ax.add_patch(rect)
                ax.text(chord.time + width/2, y_pos, chord.chord_name,
                       ha='center', va='center', fontsize=10, weight='bold')

            ax.set_ylim(0.5, 1.5)
            ax.set_yticks([1])
            ax.set_yticklabels(['Chords'])

        # Draw note points
        if notes:
            times = [n.time for n in notes]
            confidences = [n.confidence for n in notes]
            ax.scatter(times, [0.5]*len(times), s=confidences, alpha=0.6, color='red', label='Notes')

        ax.set_xlabel('Time (s)')
        ax.set_xlim(0, self.processor.get_duration())
        ax.axvline(self.current_time, color='red', linewidth=2, alpha=0.7, label='Current')

        if notes or chords:
            ax.legend(loc='upper right')

        self.timeline_canvas.draw()

    def analyze(self):
        """Start background analysis."""
        if self.processor is None or self.processor.y is None:
            return

        n_fft = 2 ** self.n_fft_slider.value()
        analysis_mode = self.detect_notes_combo.currentText()

        detect_notes = analysis_mode in ["Both", "Notes Only"]
        detect_chords = analysis_mode in ["Both", "Chords Only"]

        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.analyze_btn.setEnabled(False)

        self.analysis_worker = AnalysisWorker(
            self.processor.y, self.processor.sr, n_fft,
            detect_notes=detect_notes, detect_chords=detect_chords
        )
        self.analysis_worker.progress.connect(self.progress.setValue)
        self.analysis_worker.finished.connect(self.on_analysis_complete)
        self.analysis_worker.error.connect(self.on_analysis_error)
        self.analysis_worker.start()

        self.statusBar().showMessage("Analyzing...")

    def on_analysis_complete(self, result):
        """Handle analysis completion."""
        self.detections = result
        self.progress.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

        notes_count = len(self.detections.get('notes', []))
        chords_count = len(self.detections.get('chords', []))
        key = self.detections.get('key')

        msg = f"Done: {notes_count} notes, {chords_count} chords"
        if key:
            msg += f", Key: {key}"
        self.statusBar().showMessage(msg)

        self.draw_spectrogram()
        self.draw_timeline()

    def on_analysis_error(self, error_msg):
        """Handle analysis error."""
        self.progress.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.statusBar().showMessage(f"Error: {error_msg}")

    def toggle_playback(self):
        """Toggle audio playback."""
        if not self.audio_available:
            self.statusBar().showMessage("Audio playback not available (install sounddevice)")
            return

        if self.is_playing:
            self.is_playing = False
            self.play_btn.setText("Play")
        else:
            self.is_playing = True
            self.play_btn.setText("Pause")
            self.play_audio()

    def play_audio(self):
        """Play audio from current position."""
        try:
            import sounddevice
            start_sample = int(self.current_time * self.processor.sr)
            end_sample = len(self.processor.y)

            chunk = self.processor.y[start_sample:end_sample]

            sounddevice.play(chunk, samplerate=self.processor.sr)
            self.is_playing = True
        except Exception as e:
            self.statusBar().showMessage(f"Playback error: {e}")
            self.is_playing = False

    def on_timeline_seek(self, value):
        """Handle timeline slider movement."""
        if self.processor is None:
            return
        self.current_time = value / 100.0
        self.update_time_label()
        self.draw_spectrogram()
        self.draw_timeline()

    def update_time_label(self):
        """Update current time display."""
        if self.processor is None:
            return

        current_min = int(self.current_time) // 60
        current_sec = int(self.current_time) % 60
        total_min = int(self.processor.get_duration()) // 60
        total_sec = int(self.processor.get_duration()) % 60

        self.time_label.setText(
            f"{current_min:02d}:{current_sec:02d} / {total_min:02d}:{total_sec:02d}"
        )

    def export_results(self):
        """Export analysis results."""
        if self.processor is None:
            return

        filepath, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "JSON (*.json)")
        if filepath:
            try:
                AnalysisExporter.export_json(
                    filepath,
                    str(self.file_label.text()),
                    self.processor.get_duration(),
                    self.detections.get('notes', []),
                    self.detections.get('chords', []),
                    {'key': self.detections.get('key')}
                )
                self.statusBar().showMessage(f"Exported to {filepath}")
            except Exception as e:
                self.statusBar().showMessage(f"Export error: {e}")


def main():
    app = QApplication(sys.argv)
    window = MusicAnalyzerAdvancedGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
