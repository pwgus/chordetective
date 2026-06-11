# Music Analyzer

Análisis en tiempo real de audio para extraer notas y acordes con interfaz gráfica y CLI.

## Features

✓ Carga de archivos de audio (WAV, MP3, FLAC, OGG, M4A)
✓ Detección de notas en tiempo real usando análisis de pitch
✓ Detección de acordes desde características de chroma
✓ GUI interactiva con visualización de espectrograma
✓ CLI para procesamiento en batch
✓ Exportación de resultados (JSON, CSV)
✓ Generación de vídeos con análisis sincronizado

## Installation

```bash
pip install -r requirements.txt
```

Requiere `ffmpeg` instalado en el sistema para generación de vídeos:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Arch
sudo pacman -S ffmpeg
```

## Usage

### GUI

```bash
python main.py gui
# o simplemente
python main.py
```

**Flujo:**
1. Selecciona archivo de audio
2. Ajusta slider de ventana FFT (512-4096 samples)
3. Haz clic en "Analyze"
4. Visualiza notas detectadas sobre el espectrograma

### CLI - Análisis simple

```bash
python main.py analyze audio.wav
```

**Opciones:**
```bash
python main.py analyze audio.wav \
  --n-fft 2048 \
  --method pitch \
  --confidence 0.5 \
  --output-json results.json
```

- `--n-fft`: Tamaño de ventana FFT (512-4096)
- `--method`: 'pitch' (pitch tracking) o 'chroma' (pitch class features)
- `--confidence`: Umbral mínimo de confianza (0-1)
- `--output-json`: Guardar resultados en JSON

### CLI - Análisis avanzado

```python
from audio_processor import AudioProcessor
from music_analyzer import NoteDetector
from chord_detector import ChordDetector
from exporters import AnalysisExporter

# Cargar audio
processor = AudioProcessor()
processor.load('audio.wav')

# Detectar notas
note_detector = NoteDetector(sr=processor.sr)
notes = note_detector.detect_notes_from_pitch(
    processor.y, n_fft=2048, confidence_threshold=0.4
)

# Detectar acordes
chord_detector = ChordDetector(sr=processor.sr)
chords = chord_detector.detect_chords(
    processor.y, n_fft=2048, confidence_threshold=0.3
)

# Exportar
AnalysisExporter.export_json(
    'results.json', 'audio.wav', processor.get_duration(),
    notes, chords
)
AnalysisExporter.export_csv_notes('notes.csv', notes)
AnalysisExporter.export_csv_chords('chords.csv', chords)
```

## Architecture

```
audio_processor.py      - Cargar y procesar audio
music_analyzer.py       - Detección de notas (pitch + chroma)
chord_detector.py       - Detección de acordes desde chroma
video_generator.py      - Generación de vídeos MP4
exporters.py            - Exportación JSON/CSV
gui.py                  - Interfaz gráfica PyQt5
cli.py                  - Interfaz de línea de comandos
main.py                 - Punto de entrada
```

## Parameters Explained

### n_fft (FFT Window Size)
- **Rango:** 512 - 4096 samples
- **Efecto:** Ventanas más grandes = mejor resolución de frecuencia pero peor temporal
- **Default:** 2048
- **Recomendación:** 
  - Música de baja frecuencia: 4096
  - Música de alta frecuencia: 1024
  - Equilibrio: 2048

### confidence_threshold
- **Rango:** 0.0 - 1.0
- **Efecto:** Solo muestra detecciones con confianza superior a este valor
- **Default:** 0.5 (notas), 0.3 (acordes)

### method (Detección de notas)
- **pitch**: Usa PYIN para extracción de pitch fundamental
  - Pros: Información de frecuencia exacta, octava correcta
  - Contras: Más lento
- **chroma**: Usa características de pitch class (energía por nota relativa)
  - Pros: Más rápido, robusto
  - Contras: Sin información de octava

## Output Formats

### JSON
```json
{
  "audio_file": "song.wav",
  "duration": 120.5,
  "notes": [
    {"time": 0.5, "name": "C4", "frequency": 261.6, "confidence": 85.3},
    {"time": 1.2, "name": "D4", "frequency": 293.7, "confidence": 92.1}
  ],
  "chords": [
    {"time": 0.0, "name": "Cmaj", "confidence": 78.5},
    {"time": 2.0, "name": "Amin", "confidence": 82.1}
  ]
}
```

### CSV (Notes)
```
Time (s),Note,Frequency (Hz),Confidence (%)
0.500,C4,261.6,85.3
1.200,D4,293.7,92.1
```

### CSV (Chords)
```
Time (s),Chord,Confidence (%)
0.000,Cmaj,78.5
2.000,Amin,82.1
```

## Performance Notes

- **Audio de ~3min a 22050Hz:** ~10-20s (pitch), ~5-10s (chroma)
- **GUI es responsive:** Análisis en background thread
- **Generación de vídeo:** ~1-2 min por minuto de audio (4 threads)

## Troubleshooting

### Error: "librosa not found"
```bash
pip install librosa
```

### Error: "FFmpeg not found" (al generar vídeo)
Instala ffmpeg (ver Installation section)

### Análisis lento
- Reduce `n_fft` (2048 → 1024)
- Usa method='chroma' en lugar de 'pitch'
- Aumenta `confidence_threshold` para filtrar ruido

### Detecciones incorrectas
- Intenta diferentes valores de `n_fft`
- Sube `confidence_threshold` para eliminar falsos positivos
- Verifica que audio sea monoaural y limpio

## Development

Estructura modular permite agregar:
- Nuevas interfaces (Web, CLI mejorada)
- Nuevos métodos de detección (ML-based)
- Análisis adicional (tempo, key detection)
- Formatos de exportación (MIDI, notation)

## License

MIT
