"""PyQt5 GUI for Music Analyzer."""

import re
import sys
import numpy as np
import librosa
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog, QProgressBar,
    QComboBox, QCheckBox, QGroupBox, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import matplotlib.cm as cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from audio_processor import AudioProcessor
from music_analyzer import (AnalysisSettings, AnalysisResult, NoteDetector,
                            run_full_analysis)
from config import Config


def _pitch_class_color(label: str):
    """Color for a note label, keyed by pitch class; gray for silence."""
    m = re.match(r'^([A-G]#?)', label or '')
    if not m:
        return (0.75, 0.75, 0.75, 1.0)
    return cm.hsv(NoteDetector.NOTES.index(m.group(1)) / 12.0)


class AnalysisWorker(QThread):
    """Background thread for audio analysis."""
    finished = pyqtSignal(object)  # AnalysisResult
    error = pyqtSignal(str)

    def __init__(self, processor: AudioProcessor, settings: AnalysisSettings):
        super().__init__()
        self.processor = processor
        self.settings = settings

    def run(self):
        try:
            self.finished.emit(run_full_analysis(self.processor, self.settings))
        except Exception as e:
            self.error.emit(str(e))


class MusicAnalyzerGUI(QMainWindow):
    """Main GUI window."""

    SEGMENTATION_MODES = [
        ('Onsets', 'onsets'),
        ('Ventana fija', 'fixed'),
        ('Adaptativa (flux)', 'adaptive'),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Music Analyzer")
        self.setGeometry(100, 100, 1200, 860)

        self.config = Config()
        self.processor = None
        self.result = None  # AnalysisResult
        self.analysis_worker = None
        # Suppress config writes while widgets are being initialized,
        # otherwise half-initialized widget values clobber the saved config
        self._loading_config = True

        self.init_ui()
        self.apply_config_to_widgets()
        self._loading_config = False

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

        self.n_fft_slider = QSlider(Qt.Horizontal)
        self.n_fft_slider.setMinimum(9)   # 2^9 = 512
        self.n_fft_slider.setMaximum(12)  # 2^12 = 4096
        self.n_fft_slider.setValue(11)    # 2^11 = 2048
        self.n_fft_slider.valueChanged.connect(self.on_nfft_changed)
        self.n_fft_label = QLabel("2048")

        control_layout.addWidget(QLabel("FFT Size (n_fft):"))
        control_layout.addWidget(self.n_fft_slider)
        control_layout.addWidget(self.n_fft_label)
        control_layout.addSpacing(20)

        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.analyze)
        self.analyze_btn.setEnabled(False)
        control_layout.addWidget(self.analyze_btn)
        control_layout.addStretch()

        layout.addLayout(control_layout)

        # Smoothing & windowing settings
        layout.addWidget(self._build_smoothing_group())

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Spectrogram + timeline canvas
        self.figure = Figure(figsize=(12, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Status bar
        self.statusBar().showMessage("Ready")

    def _build_smoothing_group(self) -> QGroupBox:
        """Build the 'Suavizado y ventanas' settings panel."""
        group = QGroupBox("Suavizado y ventanas")
        grid = QGridLayout(group)

        # Segmentation mode + transform
        grid.addWidget(QLabel("Segmentación:"), 0, 0)
        self.seg_combo = QComboBox()
        self.seg_combo.addItems([name for name, _ in self.SEGMENTATION_MODES])
        self.seg_combo.currentIndexChanged.connect(self.on_settings_changed)
        grid.addWidget(self.seg_combo, 0, 1)

        self.cqt_check = QCheckBox("Análisis CQT (en vez de STFT)")
        self.cqt_check.stateChanged.connect(self.on_settings_changed)
        grid.addWidget(self.cqt_check, 0, 2, 1, 2)

        # Fixed window size (only active in fixed mode)
        grid.addWidget(QLabel("Tamaño ventana fija:"), 1, 0)
        self.window_slider = QSlider(Qt.Horizontal)
        self.window_slider.setMinimum(9)   # 2^9 = 512
        self.window_slider.setMaximum(14)  # 2^14 = 16384
        self.window_slider.valueChanged.connect(self.on_settings_changed)
        grid.addWidget(self.window_slider, 1, 1)
        self.window_label = QLabel("2048")
        grid.addWidget(self.window_label, 1, 2)

        # Flux sensitivity (only active in adaptive mode)
        grid.addWidget(QLabel("Sensibilidad de transición:"), 2, 0)
        self.flux_slider = QSlider(Qt.Horizontal)
        self.flux_slider.setMinimum(0)
        self.flux_slider.setMaximum(100)
        self.flux_slider.valueChanged.connect(self.on_settings_changed)
        grid.addWidget(self.flux_slider, 2, 1)
        self.flux_label = QLabel("0.50")
        grid.addWidget(self.flux_label, 2, 2)

        # Minimum note duration
        grid.addWidget(QLabel("Duración mínima (ms):"), 3, 0)
        self.min_dur_slider = QSlider(Qt.Horizontal)
        self.min_dur_slider.setMinimum(0)    # 0 = sin filtro
        self.min_dur_slider.setMaximum(500)
        self.min_dur_slider.setSingleStep(10)
        self.min_dur_slider.valueChanged.connect(self.on_settings_changed)
        grid.addWidget(self.min_dur_slider, 3, 1)
        self.min_dur_label = QLabel("100 ms")
        grid.addWidget(self.min_dur_label, 3, 2)

        # Median filter frames
        grid.addWidget(QLabel("Filtro de mediana (frames):"), 4, 0)
        self.median_slider = QSlider(Qt.Horizontal)
        self.median_slider.setMinimum(1)   # 1 = sin suavizado
        self.median_slider.setMaximum(15)
        self.median_slider.valueChanged.connect(self.on_settings_changed)
        grid.addWidget(self.median_slider, 4, 1)
        self.median_label = QLabel("3")
        grid.addWidget(self.median_label, 4, 2)

        # Checkboxes
        self.majority_check = QCheckBox("Voto por mayoría (ventana gaussiana)")
        self.majority_check.stateChanged.connect(self.on_settings_changed)
        grid.addWidget(self.majority_check, 5, 0, 1, 2)

        self.hmm_check = QCheckBox("Suavizado HMM (mayor tiempo de procesado)")
        self.hmm_check.stateChanged.connect(self.on_settings_changed)
        grid.addWidget(self.hmm_check, 5, 2, 1, 2)

        self.show_raw_check = QCheckBox("Mostrar detección cruda (comparativa)")
        self.show_raw_check.stateChanged.connect(self.on_show_raw_changed)
        grid.addWidget(self.show_raw_check, 6, 0, 1, 2)

        return group

    # ---------------------------------------------------------------- config

    def apply_config_to_widgets(self):
        """Initialize widgets from persisted user configuration."""
        cfg = self.config

        n_fft = int(cfg.get('n_fft', 2048))
        self.n_fft_slider.setValue(int(np.clip(np.log2(max(n_fft, 512)), 9, 12)))

        seg_value = cfg.get('segmentation', 'onsets')
        values = [v for _, v in self.SEGMENTATION_MODES]
        self.seg_combo.setCurrentIndex(values.index(seg_value) if seg_value in values else 0)

        self.cqt_check.setChecked(cfg.get('transform', 'cqt') == 'cqt')

        window = int(cfg.get('window_size', 2048))
        self.window_slider.setValue(int(np.clip(np.log2(max(window, 512)), 9, 14)))

        self.flux_slider.setValue(int(float(cfg.get('flux_sensitivity', 0.5)) * 100))
        self.min_dur_slider.setValue(int(cfg.get('min_note_duration_ms', 100)))
        self.median_slider.setValue(int(cfg.get('median_frames', 3)))
        self.majority_check.setChecked(int(cfg.get('majority_frames', 0)) > 1)
        self.hmm_check.setChecked(bool(cfg.get('hmm_smoothing', False)))
        self.show_raw_check.setChecked(bool(cfg.get('show_raw_comparison', True)))

        self._refresh_setting_labels()

    def sync_config_from_widgets(self):
        """Persist current widget state into the user configuration."""
        values = [v for _, v in self.SEGMENTATION_MODES]
        self.config.set('n_fft', 2 ** self.n_fft_slider.value())
        self.config.set('segmentation', values[self.seg_combo.currentIndex()])
        self.config.set('transform', 'cqt' if self.cqt_check.isChecked() else 'stft')
        self.config.set('window_size', 2 ** self.window_slider.value())
        self.config.set('flux_sensitivity', self.flux_slider.value() / 100.0)
        self.config.set('min_note_duration_ms', self.min_dur_slider.value())
        self.config.set('median_frames', self.median_slider.value())
        self.config.set('majority_frames', 5 if self.majority_check.isChecked() else 0)
        self.config.set('hmm_smoothing', self.hmm_check.isChecked())
        self.config.set('show_raw_comparison', self.show_raw_check.isChecked())

    def _refresh_setting_labels(self):
        """Update value labels and enable/disable mode-dependent sliders."""
        self.window_label.setText(str(2 ** self.window_slider.value()))
        self.flux_label.setText(f"{self.flux_slider.value() / 100.0:.2f}")
        min_dur = self.min_dur_slider.value()
        self.min_dur_label.setText(f"{min_dur} ms" if min_dur > 0 else "off")
        median = self.median_slider.value()
        self.median_label.setText(str(median) if median > 1 else "off")

        seg_value = self.SEGMENTATION_MODES[self.seg_combo.currentIndex()][1]
        self.window_slider.setEnabled(seg_value == 'fixed')
        self.flux_slider.setEnabled(seg_value == 'adaptive')

    # --------------------------------------------------------------- events

    def on_settings_changed(self, *_):
        self._refresh_setting_labels()
        if not self._loading_config:
            self.sync_config_from_widgets()

    def on_show_raw_changed(self, *_):
        if self._loading_config:
            return
        self.sync_config_from_widgets()
        if self.result is not None:
            self.draw_all()

    def on_nfft_changed(self, value):
        self.n_fft_label.setText(str(2 ** value))
        if not self._loading_config:
            self.config.set('n_fft', 2 ** value)

    def open_file(self):
        """Open audio file dialog."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", self.config.get('last_open_dir', ''),
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*)"
        )
        if filepath:
            self.config.set('last_open_dir', str(Path(filepath).parent))
            self.load_file(filepath)

    def load_file(self, filepath: str):
        """Load and display audio file."""
        try:
            self.processor = AudioProcessor()
            self.processor.load(filepath)
            self.result = None
            self.file_label.setText(Path(filepath).name)
            self.analyze_btn.setEnabled(True)
            self.draw_all()
            self.statusBar().showMessage(f"Loaded: {Path(filepath).name}")
        except Exception as e:
            self.statusBar().showMessage(f"Error: {e}")

    def analyze(self):
        """Start background analysis with current settings."""
        if self.processor is None or self.processor.y is None:
            return

        self.sync_config_from_widgets()
        settings = AnalysisSettings.from_config(self.config.to_dict())
        settings.detect_chords = False  # this GUI visualizes notes

        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # busy indicator
        self.analyze_btn.setEnabled(False)

        self.analysis_worker = AnalysisWorker(self.processor, settings)
        self.analysis_worker.finished.connect(self.on_analysis_complete)
        self.analysis_worker.error.connect(self.on_analysis_error)
        self.analysis_worker.start()

        hmm_note = " + HMM" if settings.smoothing.hmm else ""
        self.statusBar().showMessage(f"Analyzing ({settings.segmentation}, "
                                     f"{settings.transform}{hmm_note})...")

    def on_analysis_complete(self, result):
        """Called when analysis finishes."""
        self.result = result
        self.progress.setVisible(False)
        self.progress.setRange(0, 100)
        self.analyze_btn.setEnabled(True)

        audible = [d for d in result.notes if d.note_name != 'silence']
        self.statusBar().showMessage(
            f"Analysis complete: {len(result.raw_notes)} raw segments → "
            f"{len(audible)} notes after smoothing")
        self.draw_all()

    def on_analysis_error(self, error_msg):
        """Called if analysis fails."""
        self.progress.setVisible(False)
        self.progress.setRange(0, 100)
        self.analyze_btn.setEnabled(True)
        self.statusBar().showMessage(f"Error: {error_msg}")

    # -------------------------------------------------------------- drawing

    def draw_all(self):
        """Draw spectrogram plus comparative raw/smoothed timeline."""
        if self.processor is None or self.processor.y is None:
            return

        show_raw = self.show_raw_check.isChecked()
        transform = self.config.get('transform', 'cqt')
        n_fft = 2 ** self.n_fft_slider.value()

        try:
            S, freqs, times = self.processor.compute_transform(transform, n_fft=n_fft)
            S_db = librosa.power_to_db(S**2, ref=np.max)

            self.figure.clear()
            gs = self.figure.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.12)
            ax_spec = self.figure.add_subplot(gs[0])
            ax_tl = self.figure.add_subplot(gs[1], sharex=ax_spec)

            self._draw_spectrogram(ax_spec, S_db, freqs, times, transform, show_raw)
            self._draw_timeline(ax_tl, show_raw)

            ax_spec.set_title(f"Spectrogram ({transform.upper()}, n_fft={n_fft})")
            self.canvas.draw()
        except Exception as e:
            self.statusBar().showMessage(f"Error drawing: {e}")

    def _draw_spectrogram(self, ax, S_db, freqs, times, transform, show_raw):
        if transform == 'cqt':
            im = ax.pcolormesh(times, freqs, S_db, cmap='viridis', shading='auto')
            ax.set_yscale('log')
            ax.set_ylim(freqs[0], freqs[-1])
        else:
            im = ax.imshow(S_db, aspect='auto', origin='lower',
                           extent=[times[0], times[-1], freqs[0], freqs[-1]],
                           cmap='viridis', interpolation='nearest')

        if self.result is not None:
            if show_raw:
                raw_pts = [(d.time + d.duration / 2, d.frequency)
                           for d in self.result.raw_notes if d.frequency > 0]
                if raw_pts:
                    rx, ry = zip(*raw_pts)
                    ax.scatter(rx, ry, s=14, color='lightgray', alpha=0.5, zorder=3)

            for det in self.result.notes:
                if det.frequency > 0 and det.duration > 0:
                    ax.hlines(det.frequency, det.time, det.time + det.duration,
                              colors=[_pitch_class_color(det.note_name)],
                              linewidth=3, alpha=0.9, zorder=4)

        ax.set_ylabel('Frequency (Hz)')

    def _draw_timeline(self, ax, show_raw):
        """Comparative lane view: raw detections (gray) vs smoothed (color)."""
        ax.set_xlabel('Time (s)')
        ax.set_xlim(0, self.processor.get_duration())
        ax.set_ylim(0, 1)

        if self.result is None:
            ax.set_yticks([])
            ax.text(0.5, 0.5, 'Run analysis to see detected notes',
                    transform=ax.transAxes, ha='center', va='center', color='gray')
            return

        # Onset marks
        if self.result.onset_times:
            ax.vlines(self.result.onset_times, 0, 1, color='steelblue',
                      linewidth=0.8, alpha=0.4, linestyles='dashed')

        def draw_lane(detections, y0, y1, colored):
            span = ax.get_xlim()[1] - ax.get_xlim()[0]
            for det in detections:
                if det.note_name == 'silence' or det.duration <= 0:
                    continue
                color = _pitch_class_color(det.note_name) if colored else (0.6, 0.6, 0.6, 1.0)
                ax.add_patch(Rectangle((det.time, y0), det.duration, y1 - y0,
                                       facecolor=color, edgecolor='black',
                                       linewidth=0.3, alpha=0.85 if colored else 0.5))
                if det.duration > span * 0.015:
                    ax.text(det.time + det.duration / 2, (y0 + y1) / 2,
                            det.note_name, ha='center', va='center', fontsize=7)

        if show_raw:
            draw_lane(self.result.raw_notes, 0.05, 0.45, colored=False)
            draw_lane(self.result.notes, 0.55, 0.95, colored=True)
            ax.set_yticks([0.25, 0.75])
            ax.set_yticklabels(['Cruda', 'Suavizada'], fontsize=8)
        else:
            draw_lane(self.result.notes, 0.15, 0.85, colored=True)
            ax.set_yticks([0.5])
            ax.set_yticklabels(['Notas'], fontsize=8)


def main():
    app = QApplication(sys.argv)
    window = MusicAnalyzerGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
