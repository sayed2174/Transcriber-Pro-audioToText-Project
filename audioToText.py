# Required installations:
# must install ffmpeg tool from web and place file on ENV PATH on your System
# should create python environment and install needed packages:
# pip install openai-whisper PySide6 pydub sounddevice python-dotenv [and needed things]
import sys
import os
import threading
import whisper
import torch
from pydub import AudioSegment
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QComboBox, QTextEdit,
                               QProgressBar, QFileDialog, QMessageBox, QSystemTrayIcon, QMenu, QTextBrowser)
from PySide6.QtCore import Qt, QThread, Signal, QUrl, QTimer, QObject
from PySide6.QtGui import QPalette, QColor, QIcon, QAction, QTextDocument, QTextCursor
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
import soundfile as sf
from datetime import datetime
from PySide6.QtWidgets import QPlainTextEdit, QSplitter
import json
import subprocess
import queue
import time
import requests
from tqdm import tqdm

startupinfo = subprocess.STARTUPINFO()
startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
startupinfo.wShowWindow = subprocess.SW_HIDE  # Hide console window

load_dotenv()

class StreamToLogger:
    """Custom stream that redirects writes to a logger function."""
    def __init__(self, logger, log_level="INFO"):
        self.logger = logger
        self.log_level = log_level
        self.buffer = ''
        self._closed = False

    def write(self, message):
        """Write message to logger, buffering until newline."""
        if self._closed or not message:
            return
            
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='replace')
            
        self.buffer += message
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            for line in lines[:-1]:
                if line.strip():
                    self.logger(line, self.log_level)
            self.buffer = lines[-1]

    def flush(self):
        """Flush any buffered messages to logger."""
        if not self._closed and self.buffer.strip():
            self.logger(self.buffer, self.log_level)
            self.buffer = ''

    def close(self):
        """Close the stream and flush remaining buffer."""
        if not self._closed:
            self.flush()
            self._closed = True

class BackgroundWorker(QObject):
    """Background worker for running tasks in a separate thread."""
    task_completed = Signal(object)
    progress_updated = Signal(int, str)  # Added status message
    error_occurred = Signal(str)
    log_message = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.task_queue = queue.Queue()
        self.running = True
        self.current_task = None

    def add_task(self, task_func, *args, **kwargs):
        """Add a new task to the queue."""
        self.task_queue.put((task_func, args, kwargs))

    def run(self):
        """Main loop for processing tasks from the queue."""
        while self.running:
            try:
                task_func, args, kwargs = self.task_queue.get(timeout=0.1)
                self.current_task = (task_func.__name__, args, kwargs)
                try:
                    result = task_func(*args, **kwargs)
                    self.task_completed.emit(result)
                except Exception as e:
                    self.error_occurred.emit(str(e))
                finally:
                    self.current_task = None
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                self.error_occurred.emit(f"Worker error: {str(e)}")

    def stop(self):
        """Stop the worker loop."""
        self.running = False

class QtStreamLogger(QObject):
    """Qt-based stream logger that emits messages as signals."""
    message_written = Signal(str, str)

    def __init__(self, log_level="INFO"):
        super().__init__()
        self.log_level = log_level
        self.buffer = ''
        self._closed = False  # Add a flag to track state

    def write(self, message):
        """Write message and emit as signal, buffering until newline."""
        if self._closed or not message:
            return
            
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='replace')
            
        self.buffer += message
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            for line in lines[:-1]:
                if line.strip():
                    self.message_written.emit(line, self.log_level)
            self.buffer = lines[-1]

    def flush(self):
        """Flush any buffered messages as signals."""
        if not self._closed and self.buffer.strip():
            self.message_written.emit(self.buffer, self.log_level)
            self.buffer = ''

    def close(self):
        """Flushes the buffer and permanently closes the stream."""
        if not self._closed:
            self.flush()
            self._closed = True

class AudioTranscriberApp(QMainWindow):
    """Main application window for Whisper Audio Transcriber Pro."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Transcriber Pro by Sayed Mohammed")
        self.setGeometry(100, 100, 1300, 800)
        self.dark_mode = False
        self.current_file = ""
        self.recording = False
        self.audio_data = []
        self.word_timestamps = False
        self._cleanup_complete = False
        self.command_history = []
        self.history_index = 0
        
        # Initialize background worker
        self.worker = BackgroundWorker()
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.task_completed.connect(self.handle_task_completed)
        self.worker.error_occurred.connect(self.handle_worker_error)
        self.worker.log_message.connect(self.log)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker_thread.start()

        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        # Setup our stream redirectors
        self._stdout_redirector = QtStreamLogger("INFO")
        self._stderr_redirector = QtStreamLogger("ERROR")

        # Connect the signal to your log and terminal append methods
        self._stdout_redirector.message_written.connect(self.terminal_append)
        self._stderr_redirector.message_written.connect(self.terminal_append)

        # Redirect system streams
        sys.stdout = self._stdout_redirector
        sys.stderr = self._stderr_redirector
                
        
        self.init_ui()
        self.init_tray()
        self.update_device_list()
        self.init_log_panel()
        self.init_terminal()
        
        # Connect cleanup handler
        QApplication.instance().aboutToQuit.connect(self.cleanup)

    def cleanup(self):
        """Cleanup resources and restore system state before exit."""
        if self._cleanup_complete:
            return
            
        try:
            # Stop background worker
            self.worker.stop()
            self.worker_thread.quit()
            self.worker_thread.wait()
            
            # Stop any running operations
            if hasattr(self, 'transcription_thread'):
                self.transcription_thread.stop()
            self.stop_recording()
            
            # Restore original streams
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
            
            # Close our redirectors
            self._stdout_redirector.close()
            self._stderr_redirector.close()
            
            self._cleanup_complete = True
            self.log("Application cleanup completed")
        except Exception as e:
            sys.__stderr__.write(f"Cleanup error: {str(e)}\n")

    def closeEvent(self, event):
        """Handle window close event and trigger cleanup."""
        self.cleanup()
        event.accept()

    def init_ui(self):
        """Initialize the main user interface layout and widgets."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # File Selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("Audio File:")
        self.file_input = QLineEdit()
        self.file_input.setReadOnly(True)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)
        self.batch_button = QPushButton("Batch Process")
        self.batch_button.clicked.connect(self.browse_batch)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_input, 1)
        file_layout.addWidget(self.browse_button)
        file_layout.addWidget(self.batch_button)

        # Model Selection
        model_layout = QHBoxLayout()
        self.model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large", "turbo"])
        self.model_combo.setCurrentIndex(1)  # Default to 'base'
        self.model_combo.currentTextChanged.connect(self.check_model_availability)
        
        self.download_button = QPushButton("Download Model")
        self.download_button.clicked.connect(self.download_model)
        
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_combo, 1)
        model_layout.addWidget(self.download_button)

        # Language and Options
        options_layout = QHBoxLayout()
        self.lang_label = QLabel("Language:")
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["Auto-detect", "English", "Spanish", "French", "German", "Chinese", "Japanese"])
        
        self.timestamps_check = QPushButton("Timestamps: Off")
        self.timestamps_check.setCheckable(True)
        self.timestamps_check.clicked.connect(self.toggle_timestamps)
        
        options_layout.addWidget(self.lang_label)
        options_layout.addWidget(self.lang_combo, 1)
        options_layout.addWidget(self.timestamps_check)

        # Audio Input
        input_layout = QHBoxLayout()
        self.input_label = QLabel("Input Device:")
        self.input_combo = QComboBox()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.update_device_list)
        
        self.record_button = QPushButton("Record from Mic")
        self.record_button.clicked.connect(self.toggle_recording)
        
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_combo, 1)
        input_layout.addWidget(self.refresh_button)
        input_layout.addWidget(self.record_button)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)

        # Buttons
        button_layout = QHBoxLayout()
        self.transcribe_button = QPushButton("Transcribe")
        self.transcribe_button.clicked.connect(self.start_transcription)
        
        self.save_button = QPushButton("Save Transcription")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_transcription)
        
        self.export_format = QComboBox()
        self.export_format.addItems(["TXT", "SRT", "JSON"])
        
        self.theme_button = QPushButton("Toggle Dark Mode")
        self.theme_button.clicked.connect(self.toggle_theme)
        
        button_layout.addWidget(self.transcribe_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.export_format)
        button_layout.addWidget(self.theme_button)

        # Playback Controls
        playback_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_playback)
        
        self.position_slider = QProgressBar()
        self.position_slider.setRange(0, 100)
        self.position_slider.setTextVisible(False)
        
        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(self.stop_button)
        playback_layout.addWidget(self.position_slider, 1)

        # Results
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)

        # Add to main layout
        main_layout.addLayout(file_layout)
        main_layout.addLayout(model_layout)
        main_layout.addLayout(options_layout)
        main_layout.addLayout(input_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_label)
        main_layout.addLayout(button_layout)
        main_layout.addLayout(playback_layout)
        main_layout.addWidget(QLabel("Transcription Results:"))
        main_layout.addWidget(self.results_text, 1)

        # Initialize media player
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self.update_position)
        self.player.playbackStateChanged.connect(self.update_play_button)

        # Initialize theme
        self.toggle_theme()

        # GPU status
        gpu_status = "GPU: " + ("✅" if torch.cuda.is_available() else "❌")
        self.statusBar().addPermanentWidget(QLabel(gpu_status))

        # Enable drag-and-drop
        self.setAcceptDrops(True)

        # Add model management menu
        model_menu = self.menuBar().addMenu("Models")
        
        download_action = QAction("Download Current Model", self)
        download_action.triggered.connect(self.download_model)
        model_menu.addAction(download_action)

        check_action = QAction("Verify All Models", self)
        check_action.triggered.connect(self.check_all_models)
        model_menu.addAction(check_action)

        # Check initial model availability
        self.check_model_availability()

    def init_tray(self):
        """Initialize system tray icon and menu."""
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon.fromTheme("audio-input-microphone"))
        
        tray_menu = QMenu()
        show_action = QAction("Show", self)
        quit_action = QAction("Quit", self)
        
        show_action.triggered.connect(self.show)
        quit_action.triggered.connect(self.close)
        
        tray_menu.addAction(show_action)
        tray_menu.addAction(quit_action)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
        self.tray_icon.activated.connect(self.tray_icon_activated)

    def tray_icon_activated(self, reason):
        """Show main window when tray icon is double-clicked."""
        if reason == QSystemTrayIcon.DoubleClick:
            self.show()

    def update_device_list(self):
        """Update the list of available audio input devices."""
        self.input_combo.clear()
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                self.input_combo.addItem(f"{device['name']} (Ch: {device['max_input_channels']})", i)

    def toggle_timestamps(self):
        """Toggle word-level timestamps for transcription."""
        self.word_timestamps = not self.word_timestamps
        self.timestamps_check.setText(f"Timestamps: {'On' if self.word_timestamps else 'Off'}")

    def dragEnterEvent(self, event):
        """Accept drag event if it contains audio files."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Handle dropped audio files and set as current file."""
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac')):
                self.current_file = file_path
                self.file_input.setText(file_path)
                self.save_button.setEnabled(False)
                break

    def browse_file(self):
        """Open file dialog to select a single audio file."""
        self.log("Opening file dialog...")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.mp3 *.wav *.m4a *.ogg *.flac);;All Files (*)"
        )
        if file_path:
            self.current_file = file_path
            self.file_input.setText(file_path)
            self.save_button.setEnabled(False)
            self.log(f"Selected file: {file_path}")

    def browse_batch(self):
        """Open file dialog to select multiple audio files for batch processing."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Audio Files",
            "",
            "Audio Files (*.mp3 *.wav *.m4a *.ogg *.flac);;All Files (*)"
        )
        if files:
            self.batch_process(files)

    def batch_process(self, files):
        """Start batch processing of multiple audio files."""
        self.batch_files = files
        self.current_batch_index = 0
        self.process_next_batch_item()

    def process_next_batch_item(self):
        """Process the next file in the batch."""
        if self.current_batch_index < len(self.batch_files):
            self.current_file = self.batch_files[self.current_batch_index]
            self.file_input.setText(self.current_file)
            self.start_transcription()
        else:
            QMessageBox.information(self, "Complete", "Batch processing finished!")

    def check_model_availability(self):
        """Check if selected model is available locally and update UI."""
        model_name = self.model_combo.currentText()
        model_path = os.path.join(
            os.path.expanduser("~/.cache/whisper"),
            f"{model_name}.pt"
        )
        
        if os.path.exists(model_path):
            self.status_label.setText(f"{model_name} model available")
            self.download_button.setEnabled(False)
            return True
        else:
            self.status_label.setText(f"{model_name} model not found")
            self.download_button.setEnabled(True)
            return False

    def download_model(self):
        """Prompt user and start download of selected Whisper model."""
        model_name = self.model_combo.currentText()
        self.log(f"Attempting to download model: {model_name}")
        
        try:
            # Show model sizes to user
            model_sizes = {
                "tiny": "~75MB",
                "base": "~150MB",
                "small": "~500MB",
                "medium": "~1.5GB",
                "large": "~3GB",
                "turbo": "~1.5GB"
            }
            
            size = model_sizes.get(model_name, "unknown size")
            confirm = QMessageBox.question(
                self,
                "Confirm Download",
                f"Download {model_name} model ({size})? This may take several minutes.",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if confirm != QMessageBox.Yes:
                return
                
            self.status_label.setText(f"Downloading {model_name} model...")
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            QApplication.processEvents()  # Force UI update
            
            # Create download directory if it doesn't exist
            download_root = os.path.expanduser("~/.cache/whisper")
            os.makedirs(download_root, exist_ok=True)
            
            # Run download in background
            self.worker.add_task(
                self.download_model_task,
                model_name,
                download_root
            )
            
        except Exception as e:
            self.log(f"Download initialization failed: {str(e)}", "ERROR")
            QMessageBox.critical(self, "Error", f"Download failed:\n{str(e)}")
            self.status_label.setText("Download failed")
            self.progress_bar.setVisible(False)

    # def download_model_task(self, model_name, download_root):
    #     """Background task to download a Whisper model file."""
    #     try:
    #         # Get the actual download URL (this is a simplified approach)
    #         model_url = f"https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/{model_name}.pt"
            
    #         # Temporary file path
    #         temp_path = os.path.join(download_root, f"{model_name}.pt.download")
    #         final_path = os.path.join(download_root, f"{model_name}.pt")
            
    #         # Check if file already exists
    #         if os.path.exists(final_path):
    #             self.worker.log_message.emit(f"Model {model_name} already exists", "INFO")
    #             return True
                
    #         # Start download
    #         self.worker.log_message.emit(f"Starting download of {model_name} model...", "INFO")
            
    #         response = requests.get(model_url, stream=True)
    #         response.raise_for_status()
            
    #         total_size = int(response.headers.get('content-length', 0))
    #         block_size = 1024 * 1024  # 1MB chunks
    #         progress = 0
            
    #         with open(temp_path, 'wb') as f:
    #             for data in response.iter_content(block_size):
    #                 f.write(data)
    #                 progress += len(data)
    #                 percent = int((progress / total_size) * 100)
    #                 self.worker.progress_updated.emit(percent, f"Downloading {model_name}: {percent}%")
            
    #         # Rename temp file to final name
    #         os.rename(temp_path, final_path)
            
    #         self.worker.log_message.emit(f"Model {model_name} downloaded successfully", "SUCCESS")
    #         return True
            
    #     except Exception as e:
    #         # Clean up partial download
    #         if os.path.exists(temp_path):
    #             try:
    #                 os.remove(temp_path)
    #             except:
    #                 pass
                    
    #         self.worker.log_message.emit(f"Download failed: {str(e)}", "ERROR")
    #         raise

    def download_model_task(self, model_name, download_root=None):
        """
        Downloads a Whisper model using the official library function.
        The download progress will appear in the terminal.
        """
        try:
            self.worker.log_message.emit(f"Checking for Whisper model: '{model_name}'...", "INFO")
            self.worker.log_message.emit(
                "If the model is not found, it will be downloaded automatically.", "INFO"
            )
            
            # This is the key line: Whisper handles the download and caching.
            # It's the most reliable way to get the model files.
            whisper.load_model(model_name)
            
            self.worker.log_message.emit(f"Model '{model_name}' is downloaded and ready.", "SUCCESS")
            # You can emit a signal to update the UI if needed
            self.worker.task_completed.emit(f"download_success_{model_name}")

        except Exception as e:
            self.worker.log_message.emit(f"Failed to download model '{model_name}': {str(e)}", "ERROR")
            # Let the main error handler know something went wrong
            self.worker.error_occurred.emit(str(e))
            
    def check_all_models(self):
        """Check and display the availability of all Whisper models."""
        model_dir = os.path.expanduser("~/.cache/whisper")
        os.makedirs(model_dir, exist_ok=True)
        
        available = []
        missing = []
        
        for model in ["tiny", "base", "small", "medium", "large", "turbo"]:
            if os.path.exists(os.path.join(model_dir, f"{model}.pt")):
                available.append(model)
            else:
                missing.append(model)
        
        msg = "Model Status:\n\n"
        msg += "✅ Available: " + ", ".join(available) + "\n"
        msg += "❌ Missing: " + ", ".join(missing)
        
        QMessageBox.information(self, "Model Status", msg)
        self.log("Model status checked", "INFO")

    def update_progress(self, value, message=None):
        """Update progress bar and status message."""
        self.progress_bar.setValue(value)
        if message:
            self.status_label.setText(message)
            
        # Show progress bar only when needed
        self.progress_bar.setVisible(value > 0 and value < 100)
        
        # Enable/disable buttons based on progress
        self.transcribe_button.setEnabled(value == 0 or value == 100)
        self.save_button.setEnabled(value == 100 and hasattr(self, 'current_transcription'))

    def transcription_complete(self, result):
        """Handle completion of transcription and update UI."""
        text = result["text"]
        
        if self.word_timestamps and "segments" in result:
            text = ""
            for segment in result["segments"]:
                text += f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}\n"
        
        # Split text into 1000-character chunks
        chunks = self.chunk_text(text)
        
        # Display with chunk separators
        chunked_text = "\n\n" + "="*80 + "\n\n".join(
            f"CHUNK {i+1}/{len(chunks)} (approx. {len(chunk)} chars):\n\n{chunk}"
            for i, chunk in enumerate(chunks)
        )
        
        self.results_text.setText(chunked_text)
        self.status_label.setText("Transcription complete!")
        self.transcribe_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Store both original and chunked text for saving
        self.current_transcription = {
            'original': text,
            'chunked': chunked_text,
            'chunks': chunks
        }
        
        # For batch processing
        if hasattr(self, 'batch_files'):
            self.current_batch_index += 1
            QTimer.singleShot(1000, self.process_next_batch_item)
        self.log("Transcription completed successfully", "SUCCESS")

    def transcription_error(self, error):
        """Handle errors during transcription."""
        self.log(f"Transcription failed: {error}", "ERROR")
        QMessageBox.critical(self, "Error", f"Transcription failed:\n{error}")
        self.status_label.setText("Error occurred")
        self.transcribe_button.setEnabled(True)
        self.progress_bar.setVisible(False)

    def chunk_text(self, text, chunk_size=1000):
        """Split text into chunks of approximately chunk_size characters."""
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first if possible
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
                
                # If a single paragraph is larger than chunk size
                if len(current_chunk) > chunk_size:
                    # Split by sentences
                    sentences = para.split('. ')
                    current_chunk = ""
                    for sent in sentences:
                        if len(current_chunk) + len(sent) <= chunk_size:
                            current_chunk += sent + '. '
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sent + '. '
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    def save_transcription(self):
        """Save the current transcription to a file in the selected format."""
        self.log("Attempting to save transcription...")

        if not hasattr(self, 'current_transcription'):
            QMessageBox.warning(self, "Error", "No transcription to save")
            return

        format = self.export_format.currentText().lower()
        # Suggest original filename with appropriate extension
        base_name = os.path.splitext(os.path.basename(self.current_file))[0]
        default_path = os.path.join(os.path.dirname(self.current_file), f"{base_name}.{format}")

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Transcription",
            default_path,
            f"{format.upper()} Files (*.{format});;All Files (*)"
        )

        if file_path:
            try:
                if format == "srt" and hasattr(self, 'transcription_thread') and hasattr(self.transcription_thread, 'result'):
                    self.save_as_srt(file_path, self.transcription_thread.result)
                elif format == "json" and hasattr(self, 'transcription_thread') and hasattr(self.transcription_thread, 'result'):
                    with open(file_path, 'w') as f:
                        json.dump(self.transcription_thread.result, f, indent=2)
                else:
                    # Save either original or chunked version based on user selection
                    save_text = self.current_transcription['chunked'] if format == 'txt' else self.current_transcription['original']
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(save_text)
                
                QMessageBox.information(self, "Success", "Transcription saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file:\n{str(e)}")
        self.log(f"Transcription saved to: {file_path}", "SUCCESS")

    def save_as_srt(self, file_path, result):
        """Save transcription result as SRT subtitle file."""
        if "segments" not in result:
            raise ValueError("No timestamp data available for SRT format")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result["segments"], 1):
                start = self.format_time(segment['start'])
                end = self.format_time(segment['end'])
                f.write(f"{i}\n{start} --> {end}\n{segment['text']}\n\n")

    def format_time(self, seconds):
        """Format seconds as SRT time string."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

    def toggle_playback(self):
        """Toggle audio playback of the current file."""
        if not self.current_file:
            return
        
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.setSource(QUrl.fromLocalFile(self.current_file))
            self.player.play()

    def stop_playback(self):
        """Stop audio playback."""
        self.player.stop()

    def update_position(self, position):
        """Update playback position slider."""
        if self.player.duration() > 0:
            self.position_slider.setValue(int(position / self.player.duration() * 100))

    def update_play_button(self, state):
        """Update play button text based on playback state."""
        self.play_button.setText("Pause" if state == QMediaPlayer.PlayingState else "Play")

    def toggle_recording(self):
        """Start or stop microphone recording."""
        if self.recording:
            self.stop_recording()
            self.log("Recording stopped")
        else:
            self.start_recording()
            self.log("Recording started")

    def start_recording(self):
        """Start recording audio from the selected input device."""
        self.audio_data = []
        self.recording = True
        device_index = self.input_combo.currentData()
        
        def callback(indata, frames, time, status):
            if self.recording:
                self.audio_data.append(indata.copy())
        
        self.stream = sd.InputStream(
            device=device_index,
            channels=1,
            samplerate=16000,
            dtype='float32',
            callback=callback
        )
        self.stream.start()

    def stop_recording(self):
        """Stop audio recording and clean up resources."""
        self.recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        if hasattr(self, 'temp_recording_file'):
            try:
                os.remove(self.temp_recording_file)
            except:
                pass
            del self.temp_recording_file

    def toggle_theme(self):
        """Toggle between dark and light UI themes."""
        self.dark_mode = not self.dark_mode
        palette = QPalette()
        
        if self.dark_mode:
            # Dark theme
            palette.setColor(QPalette.Window, QColor(53, 53, 53))
            palette.setColor(QPalette.WindowText, Qt.white)
            palette.setColor(QPalette.Base, QColor(25, 25, 25))
            palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            palette.setColor(QPalette.ToolTipBase, Qt.white)
            palette.setColor(QPalette.ToolTipText, Qt.white)
            palette.setColor(QPalette.Text, Qt.white)
            palette.setColor(QPalette.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.ButtonText, Qt.white)
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
            palette.setColor(QPalette.HighlightedText, Qt.black)
            self.theme_button.setText("Toggle Light Mode")
        else:
            # Light theme
            palette.setColor(QPalette.Window, QColor(240, 240, 240))
            palette.setColor(QPalette.WindowText, Qt.black)
            palette.setColor(QPalette.Base, Qt.white)
            palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
            palette.setColor(QPalette.ToolTipBase, Qt.white)
            palette.setColor(QPalette.ToolTipText, Qt.black)
            palette.setColor(QPalette.Text, Qt.black)
            palette.setColor(QPalette.Button, QColor(240, 240, 240))
            palette.setColor(QPalette.ButtonText, Qt.black)
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Highlight, QColor(142, 45, 197))
            palette.setColor(QPalette.HighlightedText, Qt.white)
            self.theme_button.setText("Toggle Dark Mode")
        
        self.setPalette(palette)

    def init_log_panel(self):
        """Initialize the log panel for displaying application logs."""
        self.log_panel = QPlainTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setMinimumHeight(150)
        self.log_panel.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
        
        old_central = self.centralWidget()
        old_layout = old_central.layout()
        
        splitter = QSplitter(Qt.Vertical)
        top_widget = QWidget()
        top_widget.setLayout(old_layout)
        splitter.addWidget(top_widget)
        splitter.addWidget(self.log_panel)
        splitter.setSizes([500, 250])
        
        new_central = QWidget()
        new_central.setLayout(QVBoxLayout())
        new_central.layout().addWidget(splitter)
        self.setCentralWidget(new_central)
        
        self.log("Application started")
        self.log(f"GPU Available: {torch.cuda.is_available()}")

    def log(self, message, level="INFO"):
        """Log a message to the log panel with timestamp and color."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_colors = {
            "INFO": "black",
            "WARNING": "orange",
            "ERROR": "red",
            "SUCCESS": "green"
        }
        color = level_colors.get(level, "black")
        
        self.log_panel.appendHtml(
            f'<font color="yellow">[{timestamp}]</font> '
            f'<font color="{color}">{level}:</font> '
            f'<font color="white">{message}</font>'
        )
        self.log_panel.verticalScrollBar().setValue(
            self.log_panel.verticalScrollBar().maximum()
        )

    def export_logs(self):
        """Export the current log panel contents to a text file."""
        log_text = self.log_panel.toPlainText()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Log File",
            "",
            "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            with open(file_path, 'w') as f:
                f.write(log_text)
            self.log(f"Logs exported to {file_path}", "SUCCESS")

    def handle_task_completed(self, result):
        """Handle completed tasks from background worker."""
        if isinstance(result, dict) and 'text' in result:  # Transcription result
            self.transcription_complete(result)
        elif isinstance(result, bool) and result:  # Model download success
            self.status_label.setText("Model download complete!")
            self.progress_bar.setValue(100)
            QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
            self.check_model_availability()
        else:
            self.log(f"Background task completed: {str(result)}")

    def handle_worker_error(self, error):
        """Handle errors from background worker."""
        self.log(f"Background worker error: {error}", "ERROR")
        QMessageBox.critical(self, "Error", f"Background task failed:\n{error}")
        self.status_label.setText("Operation failed")
        self.progress_bar.setVisible(False)

    def start_transcription(self):
        """Start the transcription process for the selected or recorded audio."""
        if not self.current_file and not self.audio_data:
            self.log("No audio file selected", "ERROR")
            QMessageBox.warning(self, "Error", "Please select an audio file or record from microphone")
            return

        self.log("Starting transcription process...")

        # If we have recorded audio but haven't saved it
        if self.audio_data and not hasattr(self, 'temp_recording_file'):
            try:
                self.temp_recording_file = "recording.wav"
                self.worker.add_task(self.save_recording_task, self.audio_data, self.temp_recording_file)
                self.current_file = self.temp_recording_file
                self.log("Added recording save task to background worker")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save recording: {str(e)}")
                return

        if not os.path.exists(self.current_file):
            QMessageBox.warning(self, "Error", "File does not exist")
            return

        # Clear previous results
        self.results_text.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Transcribing... Please wait")
        self.transcribe_button.setEnabled(False)
        self.save_button.setEnabled(False)

        # Get selected options
        model_name = self.model_combo.currentText()
        language = self.lang_combo.currentText()

        # Start transcription in background
        self.worker.add_task(
            self.transcribe_task,
            self.current_file,
            model_name,
            language,
            self.word_timestamps
        )
        self.log(f"Added transcription task to background worker (model: {model_name}, language: {language})")

    def save_recording_task(self, audio_data, file_path):
        """Task to save recorded audio (runs in background)."""
        self.worker.log_message.emit("Saving recording...", "INFO")
        sf.write(file_path, np.concatenate(audio_data), 16000)
        return file_path

    def transcribe_task(self, file_path, model_name, language, word_timestamps):
        """Task to transcribe audio (runs in background)."""
        try:
            self.worker.log_message.emit(f"Loading model: {model_name}", "INFO")
            
            # First check if model exists
            model_path = os.path.join(
                os.path.expanduser("~/.cache/whisper"),
                f"{model_name}.pt"
            )
            
            if not os.path.exists(model_path):
                self.worker.log_message.emit(f"Model {model_name} not found, downloading...", "WARNING")
                self.download_model()  # This will show progress
                
            model = whisper.load_model(model_name)
            
            self.worker.log_message.emit("Starting transcription...", "INFO")
            result = model.transcribe(
                file_path,
                language=None if language == "Auto-detect" else language.lower(),
                word_timestamps=word_timestamps,
                task="transcribe",
                verbose=False,
                fp16=torch.cuda.is_available()
            )
            
            return result
        except Exception as e:
            self.worker.log_message.emit(f"Transcription failed: {str(e)}", "ERROR")
            raise  # Re-raise to trigger error handling

    def init_terminal(self):
        """Initialize the terminal panel for command execution and output."""
        # Create terminal output widget
        self.terminal_output = QTextBrowser()
        self.terminal_output.setStyleSheet("""
            QTextBrowser {
                background-color: black;
                color: white;
                font-family: Consolas, monospace;
                font-size: 10pt;
            }
        """)
        self.terminal_output.setReadOnly(True)
        
        # Create command input
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Enter command and press Enter...")
        self.command_input.returnPressed.connect(self.execute_command)
        
        # Create terminal panel layout
        terminal_panel = QWidget()
        terminal_layout = QVBoxLayout(terminal_panel)
        terminal_layout.addWidget(self.terminal_output, 1)
        terminal_layout.addWidget(self.command_input)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Create left panel (existing UI)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Move all existing widgets to left panel
        for i in reversed(range(self.centralWidget().layout().count())):
            widget = self.centralWidget().layout().takeAt(i).widget()
            if widget:
                left_layout.addWidget(widget)
        
        # Add panels to splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(terminal_panel)
        main_splitter.setSizes([600, 400])  # Initial sizes
        
        # Set central widget
        central = QWidget()
        central.setLayout(QVBoxLayout())
        central.layout().addWidget(main_splitter)
        self.setCentralWidget(central)
        
        # Redirect stdout/stderr to terminal
        sys.stdout = StreamToLogger(self.terminal_append, "OUT")
        sys.stderr = StreamToLogger(self.terminal_append, "ERR")

    def terminal_append(self, text, level="OUT"):
        """Append text to the terminal output with color based on level."""
        colors = {
            "OUT": "white",
            "ERR": "red",
            "CMD": "lime"
        }
        color = colors.get(level, "white")
        
        self.terminal_output.append(f'<font color="{color}">{text}</font>')
        self.terminal_output.verticalScrollBar().setValue(
            self.terminal_output.verticalScrollBar().maximum()
        )
        
    # def execute_command(self):
    #     """(Legacy) Execute a shell command and display output in terminal."""
    #     ...

    def search_terminal(self):
        """Search for text in the terminal output."""
        search_text = self.terminal_search.text()
        if not search_text:
            return
            
        # Search backwards from current position
        cursor = self.terminal_output.textCursor()
        found = self.terminal_output.document().find(
            search_text, 
            cursor.position(), 
            QTextDocument.FindBackward | QTextDocument.FindCaseSensitively
        )
        
        if not found.isNull():
            self.terminal_output.setTextCursor(found)
        else:
            # Wrap around if not found
            cursor.movePosition(QTextCursor.End)
            found = self.terminal_output.document().find(
                search_text, 
                cursor.position(), 
                QTextDocument.FindBackward | QTextDocument.FindCaseSensitively
            )
            if not found.isNull():
                self.terminal_output.setTextCursor(found)
            else:
                self.log(f"Text not found: {search_text}", "WARNING")

    def execute_command(self):
        """Execute a shell command entered in the terminal input."""
        cmd = self.command_input.text()
        if not cmd:
            return
            
        # Add to history
        self.command_history.append(cmd)
        self.history_index = len(self.command_history)
        
        self.terminal_append(f"$ {cmd}", "CMD")
        self.command_input.clear()
        
        # Run command in background
        self.worker.add_task(self.run_command_task, cmd)

    def run_command_task(self, cmd):
        """Task to run system command (runs in background)."""
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                startupinfo=startupinfo,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW,
                universal_newlines=True
            )
            
            # Read output in real-time
            while True:
                output = process.stdout.readline()
                error = process.stderr.readline()
                
                if output == '' and error == '' and process.poll() is not None:
                    break
                
                if output:
                    self.worker.log_message.emit(output.strip(), "OUT")
                if error:
                    self.worker.log_message.emit(error.strip(), "ERR")
                    
            return_code = process.poll()
            if return_code:
                self.worker.log_message.emit(f"Process exited with code {return_code}", "ERR")
            else:
                self.worker.log_message.emit("Command completed successfully", "OUT")
                
        except Exception as e:
            self.worker.log_message.emit(f"Error executing command: {str(e)}", "ERR")


def excepthook(exc_type, exc_value, exc_traceback):
    """Global exception handler for uncaught exceptions."""
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
    sys.__stderr__.write(f"Unhandled exception: {str(exc_value)}\n")
    QMessageBox.critical(
        None,
        "Unhandled Exception",
        f"An unexpected error occurred:\n{str(exc_value)}"
    )
    
if __name__ == "__main__":
    sys.excepthook = excepthook
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    
    try:
        window = AudioTranscriberApp()
        window.show()
        ret = app.exec()
        app.setWindowIcon(QIcon("ni.png"))
        # Ensure cleanup happens after event loop ends
        window.cleanup()
        sys.exit(ret)
    except Exception as e:
        sys.__stderr__.write(f"Fatal initialization error: {str(e)}\n")
        sys.exit(1)
