# Whisper Audio Transcriber Pro

A user-friendly desktop application for accurate audio transcription powered by OpenAI's Whisper. This tool provides a rich interface for transcribing audio files, recording from a microphone, and managing transcription models, complete with a built-in terminal and theme support.




---

## ## ✨ Features

* **High-Quality Transcription**: Utilizes the power of OpenAI's Whisper models (`tiny`, `base`, `small`, `medium`, `large`) for state-of-the-art accuracy.
* **Multiple Audio Formats**: Supports transcription for `.mp3`, `.wav`, `.m4a`, `.ogg`, and `.flac` files.
* **Batch Processing**: Queue up and transcribe multiple audio files in a single session.
* **Microphone Recording**: Record audio directly from your microphone and transcribe it instantly.
* **Export Options**: Save transcriptions as `.txt`, `.srt` (for subtitles), or `.json` (with detailed data).
* **Timestamp Generation**: Option to generate word-level or segment-level timestamps.
* **Built-in Terminal**: A fully functional terminal to run system commands directly within the application.
* **GPU Acceleration**: Automatically uses your NVIDIA GPU (if CUDA is available) for significantly faster transcriptions.
* **Light & Dark Themes**: Switch between a light and dark interface for your comfort.

---

## ## 🛠️ Prerequisites

Before you begin, you must have the following installed on your system.

### 1. Python
* **Python 3.12**

### 2. FFmpeg (Crucial)
Whisper requires the **FFmpeg** command-line tool to process audio formats.
* **Windows**: Download the latest build from [ffmpeg.org](https://ffmpeg.org/download.html), unzip it, and add the `bin` folder to your system's **PATH** environment variable.
* **macOS**: Install via Homebrew: `brew install ffmpeg`
* **Linux**: Install using your package manager: `sudo apt update && sudo apt install ffmpeg`

---

## ## ⚙️ Installation

Follow these steps to get the project running on your local machine.

**1. Clone the Repository**
```sh
git clone [https://github.com/sayed2174/Transcriber-Pro-audioToText-Project.git](https://github.com/sayed2174/Transcriber-Pro-audioToText-Project.git)
cd Transcriber-Pro-audioToText-Project

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

## copy the .py file into venv folder which is an python virtual machine setup

pip install -r requirements.txt #[for installing the needed libs for support of running app]
```
## Tip:
**you can make python file into .exe**

**If you want pre-mage one download this file by clickin here:**
```
https://drive.google.com/file/d/1cE-qh4ZG1lEhfwo5gD1lWkkBUzfu5N1c/view?usp=sharing
```
