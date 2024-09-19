# YouTube ASR using Whisper

## Overview

Framework for using Whisper to transcribe YouTube videos. Includes scripts for evaluating Whisper's transcriptions against YouTube's subtitles, transcribing the audio from a given YouTube URL and creating a HuggingFace dataset (using YouTube's subtitles as 
the ground truth transcription) from a YouTube URL.

## Prerequisites: ffmpeg

Ensure you have the [ffmpeg](https://www.ffmpeg.org/download.html) installed for audio processing.
Installing `ffmpeg` using chocolatey:
    ```bash
    choco install ffmpeg
    ```
Other options for [ffmpeg installation](https://avpres.net/FFmpeg/install_Windows)

## Usage

### 1. Transcribing Audio Files

You can transcribe audio files by calling the `transcribe_mp3` function.

```python
from utilities import transcribe_mp3

# video url: 40 second elevator pitch 
URL = "https://www.youtube.com/watch?v=4WEQtgnBu0I"

# download youtube mp3 from url
download_mp3(URL)

# transcribe using whisper 
transcription = transcribe_mp3(model_type="base")

# save transcript to file 
save_transcript(transcript=transcription)
```
