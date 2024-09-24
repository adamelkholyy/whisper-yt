# whisper-yt

## Overview

Toolkit for using Whisper to transcribe YouTube videos. Allows for the following functionality given only a YouTube video URL:
- Transcribe YouTube videos using Whisper, including optional diarization using pyannote
- Download and split YouTube audio into Whisper compatible segments (audio_1.mp3, audio_2.mp3, ..., audio_n.mp3) 
- Download and preprocess YouTube subtitles into a HuggingFace compatible timestamped transcript.json file
- Create and save a HuggingFace dataset from a YouTube video for use in Whisper fine tuning
- Evaluate the Word Error Rate of Whisper transcriptions against YouTube's subtitles

## Prerequisites: ffmpeg

Ensure you have the [ffmpeg](https://www.ffmpeg.org/download.html) installed for audio processing. Using chocolatey:  

```
choco install ffmpeg
```   
Or see more instructions for [ffmpeg installation](https://avpres.net/FFmpeg/install_Windows)

## Usage

### 1. Downloading and Transcribing YouTube Videos

You can download and transcribe a YouTube video with Whisper as follows

```python
from whisper_yt.yt_downloader import download_mp3
from whisper_yt.whisper_utilities import transcribe_mp3
from whisper_yt.utilities import save_transcript

# YouTube video url: 40 second elevator pitch 
URL = "https://www.youtube.com/watch?v=4WEQtgnBu0I"

# download youtube mp3 from url: default location downloads/audio.mp3 
download_mp3(URL)

# transcribe using whisper 
transcription = transcribe_mp3(mp3_path="downloads/audio.mp3", model_type="openai/whisper-base")
```

Output  
```bash
[youtube] Extracting URL: https://www.youtube.com/watch?v=4WEQtgnBu0I 
[youtube] 4WEQtgnBu0I: Downloading webpage 
[youtube] 4WEQtgnBu0I: Downloading ios player API JSON 
[youtube] 4WEQtgnBu0I: Downloading web creator player API JSON 
[youtube] 4WEQtgnBu0I: Downloading m3u8 information 
[info] 4WEQtgnBu0I: Downloading 1 format(s): 251 
[download] Destination: downloads\audio 
[download] 100% of  526.71KiB in 00:00:00 at 2.98MiB/s
[ExtractAudio] Destination: downloads\audio.mp3 
[yt_downloader] Audio download complete, MP3 saved at downloads\audio
[whisper] Using cuda: NVIDIA GeForce GTX 1650 with Max-Q Design
[whisper] openai/whisper-base loaded, transcribing downloads/audio.mp3. Note: This could take a long time depending on the length of the audio
[whisper] downloads/audio.mp3 successfully transcribed by whisper-openai/whisper-base
[utilities] Transcript successfully written to transcripts\transcript.txt
```
You can then either print ```transcription`` as is or save it to a file as follows

```python
# save transcript to file: defualt location transcripts/transcript.txt
save_transcript(transcription)
```
Output
```plaintext
Hello, my name is Andrea Fitzgera. I am studying marketing at the University of Texas at Dallas.
I am a member of the American Marketing Association and AlphaCapas-Sci, both of which are dedicated to shaping future business leaders.
I hope to incorporate my business knowledge into consumer trend analysis and strengthening relationships among consumers, as well as other companies. I am savvy, social, and principled, and have exquisite interpersonal communication skills.
I know that I can be an asset in any company and or situation,
and I hope that you will consider me for an internship or job opportunity.
Thank you so much.
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
```plaintext
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

