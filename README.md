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
from yt_downloader import download_mp3
from utilities import transcribe_mp3, save_transcript

# video url: 40 second elevator pitch 
URL = "https://www.youtube.com/watch?v=4WEQtgnBu0I"

# download youtube mp3 from url
download_mp3(URL)

# transcribe using whisper 
transcription = transcribe_mp3(model_type="base")

# save transcript to file 
save_transcript(transcript=transcription)
```

### 2. Evaluating Whisper transcriptions against YouTube subtitles
You can transcribe and evaluate the WER of a given YouTube video using Whisper by running the yt_whisper_pipeline.py script. The script will download the mp3, 
split the audio and transcript into segments, generate Whisper's transcript, create a huggingface dataset from the audio and transcript data and finally evaluate the WER against YouTube's subtitles. The following are the core optional arguments.

```python
# config settings
# video url: 40 minute Taylor Swift interview
URL = "https://www.youtube.com/watch?v=m9Nkqm7FFgk"
EXPERIMENT_TITLE = "taylor_swift_interview"

MODEL_TYPE = "openai/whisper-base"
DATASET_DIR = "datasets"
...
```
Outputs are as follows using whisper-base:
```bash
whisper_transcript.txt:
It all happened really fast because I'm very
impatient. Like if I don't have a song
finished I'll obsess over it I won't
sleep that night and I'll just edit
it constantly to the point where I can't
focus on a conversation and everyone around
me is annoyed because they're like
clearly you're working on something just
finish it. So that one was a long
time to kind of get where it needed to be.
...
Audio split into 941 segments
HuggingFace dataset saved as taylor_swift_interview_ds
WER: 29.553%
```

