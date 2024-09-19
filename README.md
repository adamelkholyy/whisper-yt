# YouTube ASR using Whisper

## Overview

Framework for using Whisper to transcribe YouTube videos. Includes scripts for evaluating Whisper's transcriptions against YouTube's subtitles, transcribing the audio from a given YouTube URL and creating a HuggingFace dataset (using YouTube's subtitles as 
the ground truth transcription) from a YouTube URL.

## Prerequisites

Ensure you have the following installed:

- [ffmpeg](https://www.ffmpeg.org/download.html) (for audio processing)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) (for downloading YouTube transcripts)

## Installaing ffmpeg

Installing `ffmpeg` using chocolatey:
    ```bash
    choco install ffmpeg
    ```
Other options for [ffmpeg installation](https://avpres.net/FFmpeg/install_Windows)

Installing `yt-dlp`:
    ```bash
    pip install yt-dlp
    ```
Other options for [yt-dlp installation](https://github.com/yt-dlp/yt-dlp/wiki/Installation)

## Usage

### 1. Transcribing Audio Files

You can transcribe audio files by calling the `transcribe_mp3` function.

```python
from utilities import transcribe_mp3

# Transcribe a local mp3 file using Whisper
transcript = transcribe_mp3(mp3_path="downloads/audio.mp3", model_type="base")
print(transcript)
