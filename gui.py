"""PyQt5 GUI for Music Analyzer."""

import sys
import numpy as np
import librosa
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from audio_processor import AudioProcessor
from music_analyzer import NoteDetector


class AnalysisWorker(QThread):
    """Background thread for audio analysis."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)  # List of detections
    error = pyqtSignal(str)

    def __init__(self, y: np.ndarray, sr: int, n_fft: int, method: str = 'pitch'):
        super().__init__()
        self.y = y
        self.sr = sr
        self.n_fft = n_fft
        self.method = method

    def run(self):
        try:
            detector = NoteDetector(sr=self.sr)
            if self.method == 'pitch':
                detections = detector.detect_notes_from_pitch(
                    self.y, n_fft=self.n_fft, confidence_threshold=0.4
                )
            else:
                detections = detector.detect_notes_from_chroma(
                    self.y, n_fft=self.n_fft, confidence_threshold=0.3
                )
            self.finished.emit(detections)
        except Exception as e:
            self.error.emit(str(e))


class MusicAnalyzerGUI(QMainWindow):
    """Main GUI window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Music Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        self.processor = None
        self.detections = []
        self.analysis_worker = None

        self.init_ui()

    def init_ui(self):
        """Initialize UI components."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # File selection
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

        # Controls
        control_layout = QHBoxLayout()

        QLabel("Window Size (n_fft):"),
        self.n_fft_slider = QSlider(Qt.Horizontal)
        self.n_fft_slider.setMinimum(9)  # 2^9 = 512
        self.n_fft_slider.setMaximum(12)  # 2^12 = 4096
        self.n_fft_slider.setValue(11)  # 2^11 = 2048
        self.n_fft_slider.sliderMoved.connect(self.on_nfft_changed)
        self.n_fft_label = QLabel("2048")

        control_layout.addWidget(QLabel("Window Size (n_fft):"))
        control_layout.addWidget(self.n_fft_slider)
        control_layout.addWidget(self.n_fft_label)
        control_layout.addSpacing(20)

        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.analyze)
        self.analyze_btn.setEnabled(False)
        control_layout.addWidget(self.analyze_btn)
        control_layout.addStretch()

        layout.addLayout(control_layout)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Spectrogram canvas
        self.figure = Figure(figsize=(12, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

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
        """Load and display audio file."""
        try:
            self.processor = AudioProcessor()
            self.processor.load(filepath)
            self.file_label.setText(Path(filepath).name)
            self.analyze_btn.setEnabled(True)
            self.draw_spectrogram()
            self.statusBar().showMessage(f"Loaded: {Path(filepath).name}")
        except Exception as e:
            self.statusBar().showMessage(f"Error: {e}")

    def on_nfft_changed(self, value):
        """Update n_fft label when slider moves."""
        n_fft = 2 ** value
        self.n_fft_label.setText(str(n_fft))

    def draw_spectrogram(self):
        """Draw spectrogram with current n_fft."""
        if self.processor is None or self.processor.y is None:
            return

        n_fft = 2 ** self.n_fft_slider.value()

        try:
            S, freqs, times = self.processor.compute_spectrogram(n_fft=n_fft)

            self.figure.clear()
            ax = self.figure.add_subplot(111)

            # Convert to dB scale
            S_db = librosa.power_to_db(S**2, ref=np.max)

            im = ax.imshow(S_db, aspect='auto', origin='lower',
                          extent=[times[0], times[-1], freqs[0], freqs[-1]],
                          cmap='viridis', interpolation='nearest')

            # Plot detected notes on top
            if self.detections:
                for det in self.detections:
                    if det.frequency > 0:
                        ax.plot(det.time, det.frequency, 'r.', markersize=8, alpha=0.7)

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'Spectrogram (n_fft={n_fft})')
            self.figure.colorbar(im, ax=ax, label='Magnitude (dB)')

            self.canvas.draw()
        except Exception as e:
            self.statusBar().showMessage(f"Error drawing: {e}")

    def analyze(self):
        """Start background analysis."""
        if self.processor is None or self.processor.y is None:
            return

        n_fft = 2 ** self.n_fft_slider.value()
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.analyze_btn.setEnabled(False)

        self.analysis_worker = AnalysisWorker(
            self.processor.y, self.processor.sr, n_fft, method='pitch'
        )
        self.analysis_worker.finished.connect(self.on_analysis_complete)
        self.analysis_worker.error.connect(self.on_analysis_error)
        self.analysis_worker.start()

        self.statusBar().showMessage("Analyzing...")

    def on_analysis_complete(self, detections):
        """Called when analysis finishes."""
        self.detections = detections
        self.progress.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.statusBar().showMessage(f"Analysis complete: {len(detections)} notes detected")
        self.draw_spectrogram()

    def on_analysis_error(self, error_msg):
        """Called if analysis fails."""
        self.progress.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.statusBar().showMessage(f"Error: {error_msg}")


def main():
    app = QApplication(sys.argv)
    window = MusicAnalyzerGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
