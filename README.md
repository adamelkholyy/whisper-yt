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
<it>see also Transcribing With Diarization</it>  
You can download and transcribe a YouTube video with Whisper (and optionally save the transcript) as follows

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

# save transcript to file: defualt location transcripts/transcript.txt
save_transcript(transcription)
```

Outputs transcripts/transcript.txt:
```plaintext
Hello, my name is Andrea Fitzgera. I am studying marketing at the University of Texas at Dallas.
I am a member of the American Marketing Association and AlphaCapas-Sci, both of which are dedicated to shaping future business leaders.
I hope to incorporate my business knowledge into consumer trend analysis and strengthening relationships among consumers, as well as other companies. I am savvy, social, and principled, and have exquisite interpersonal communication skills.
I know that I can be an asset in any company and or situation,
and I hope that you will consider me for an internship or job opportunity.
Thank you so much.
```

### 2. Creating and Saving a HuggingFace Dataset from a YouTube Video
You can create a HuggingFace dataset from a YouTube video (using its segmented audio as inputs and subtitles as ground truth transcriptions) for use in Whisper fine-tuning and/or evaluation by calling ```download_and_preprocess_yt()```, which performs the following operations:
  1) Downloads audio mp3 and subtitles raw transcript from YouTube video url
  2) Processess the raw transcript timestamps and text data
  3) Splits audio into segments from transcript 
  4) Creates transcript.json containing audio segment and clean timestamped transcript data

```python
from whisper_yt.yt_downloader import download_and_preprocess_yt
from whisper_yt.utilities import make_dataset

# YouTube video url: 40 second elevator pitch 
URL = "https://www.youtube.com/watch?v=4WEQtgnBu0I"

# downloads audio file and raw transcript to 'downloads/' and saves segmented audio to 'data/'
download_and_preprocess_yt(url)

# create huggingface dataset
ds = make_dataset(data_dir="data")

# save dataset to disk
ds.save_to_disk(dataset_path="datasets/elevator_pitch_ds")
```
This creates a directory, 'data/', structured as follows
```plaintext
data/ 
  audio_1.mp3
  audio_2.mp3 
  ...
  transcript.json
```
with transcript.json as follows 
```json
  {
    "start": 480,
    "end": 2629,
    "text": "hello my name is Andrea fitzer I am",
    "audio": "audio_1.mp3"
  },
  {
    "start": 2639,
    "end": 4430,
    "text": "studying marketing at the University of",
    "audio": "audio_2.mp3"
  },
  ...
```
The HuggingFace dataset is then created from this 'data/' directory, which is in the correct format for Whisper evaluation and fine tuning

### 3. Evaluating Whisper WER Against YouTube Subtitles
You can evalute the Word Error Rate of Whisper transcriptions using YouTube subtitles, which is especially useful in cases of manually transcribed YouTube subtitles, as follows 


