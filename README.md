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
NB: The following examples were run on a laptop using cuda with an NVIDIA GeForce GTX 1650 graphics card with the base Whisper model ('openai-whisper/base'). Better transcription results and faster runtimes would be seen using a larger Whisper model with a better graphics card.  

### 1.a. Downloading and Transcribing YouTube Videos (without Diarization)
You can download and transcribe a YouTube video with Whisper (and optionally save the transcript) without diarization as follows

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

Outputs to transcripts/transcript.txt:
```plaintext
Hello, my name is Andrea Fitzgera. I am studying marketing at the University of Texas at Dallas.
I am a member of the American Marketing Association and AlphaCapas-Sci, both of which are dedicated to shaping future business leaders.
I hope to incorporate my business knowledge into consumer trend analysis and strengthening relationships among consumers, as well as other companies. I am savvy, social, and principled, and have exquisite interpersonal communication skills.
I know that I can be an asset in any company and or situation,
and I hope that you will consider me for an internship or job opportunity.
Thank you so much.
```

### 1.b. Downloading and Transcribing YouTube Videos (with Diarization)
You can download and transcribe a YouTube video with Whisper (and optionally save the transcript) with diarization using pyannote as follows. Note that pyannote requires a HuggingFace authorization token and accepted permissions on the pyannote page, see [here](https://github.com/pyannote/pyannote-audio) for more details.

```python
from whisper_yt.yt_downloader import download_mp3
from whisper_yt.whisper_utilities import transcribe_mp3
from whisper_yt.utilities import save_transcript

# YouTube video url: 2 minute two person job interview 
URL = "https://www.youtube.com/watch?v=naIkpQ_cIt0"

# replace with your HuggingFace authorization token
my_auth_token = "hf_my_huggingface_authtoken"

# download youtube mp3 from url: default location downloads/audio.mp3 
download_mp3(URL)

# transcribe and diarize using whisper 
transcription = transcribe_mp3(mp3_path="downloads/audio.mp3",
                               model_type="openai/whisper-base",
                               diarize=True,
                               auth_token=my_auth_token)

# save transcript to file: defualt location transcripts/transcript.txt
save_transcript(transcription)
```
Outputs to transcripts/transcript.txt:
```plaintext
...
SPEAKER_01: Mary, do you have any experience working in the kitchen?
SPEAKER_00: No, but I want to learn. I work hard and I cook a lot at home.
SPEAKER_01: Okay, well tell me about yourself.
SPEAKER_00: Well, I love to learn new things. I'm very organized.
SPEAKER_00: And I follow directions exactly.
SPEAKER_00: That's why my boss at my last job made me a trainer.
SPEAKER_00: And the company actually gave me a special certificate
SPEAKER_00: for coming to work on time every day for a year.
SPEAKER_01: That's great.
SPEAKER_01: Why did you leave your last job?
...
```
Note that the pyannote diarization can sometimes be faulty, double checking the diarized transcript is highly recommended

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
```
The HuggingFace dataset is then created from this 'data/' directory, which is in the correct format for Whisper evaluation and fine tuning

### 3. Evaluating Whisper WER Against YouTube Subtitles
You can evalute the Word Error Rate of Whisper transcriptions using YouTube subtitles, which is especially useful in cases of manually transcribed YouTube subtitles.  
  
We will first quickly make a dataset using a video known to have manual transcriptions
```python
from whisper_yt.yt_downloader import download_and_preprocess_yt
from whisper_yt.utilities import make_dataset

# YouTube video url: 2 minute two person job interview 
URL = "https://www.youtube.com/watch?v=naIkpQ_cIt0"
download_and_preprocess_yt(URL)
ds = make_dataset(data_dir="data")
```
Now we evaluate the WER of Whisper's transcriptions against the manually transcribed subtitles
```python
from whisper_yt.whisper_utilities import get_whisper_transcription
from whisper_yt.utilities import filter_empty_references
from evaluate import load

# transcribed dataset using Whisper
transcribed_ds = get_whisper_transcription(ds)

references = transcribed_ds['reference']    # ground truth text
predictions = transcribed_ds['prediction']  # Whisper transcriptions

# filter out blocks of silence for more accurate WER calculation
references, predictions = filter_empty_references(references, predictions)

# calculate final WER
wer_function = load("wer")
wer_score = 100 * wer_function.compute(references=references, predictions=predictions)
print(f"Word Error Rate (WER) of Whisper transcriptions against youtube subtitles: {wer_score:.3f}%")
```
Which gives us a WER output as follows: ```Word Error Rate (WER) of Whisper transcriptions against youtube subtitles: 10.219%```. For context the full output of the above example is as follows:
```
[youtube] Extracting URL: https://www.youtube.com/watch?v=naIkpQ_cIt0 
[youtube] naIkpQ_cIt0: Downloading webpage 
[youtube] naIkpQ_cIt0: Downloading ios player API JSON 
[youtube] naIkpQ_cIt0: Downloading web creator player API JSON 
[youtube] naIkpQ_cIt0: Downloading m3u8 information 
[info] naIkpQ_cIt0: Downloading 1 format(s): 251 
[download] Destination: downloads\audio 
[download] 100% of 1.70MiB in 00:00:00 at 9.91MiB/s
[ExtractAudio] Destination: downloads\audio.mp3 
[yt_downloader] Audio download complete, MP3 saved at downloads\audio
[youtube] Extracting URL: https://www.youtube.com/watch?v=naIkpQ_cIt0 
[youtube] naIkpQ_cIt0: Downloading webpage 
[youtube] naIkpQ_cIt0: Downloading ios player API JSON 
[youtube] naIkpQ_cIt0: Downloading web creator player API JSON 
[youtube] naIkpQ_cIt0: Downloading m3u8 information 
[info] naIkpQ_cIt0: Downloading subtitles: en 
[info] naIkpQ_cIt0: Downloading 1 format(s): 616+251 
[info] Writing video subtitles to: downloads\raw_transcript.en.vtt
[download] Destination: downloads\raw_transcript.en.vtt 
[download] 100% of 2.71KiB in 00:00:00 at 63.94KiB/s
[yt_downloader] Manual subtitles downloaded, raw transcript saved to raw_transcript.en.vtt
[yt_downloader] Raw manual transcript successfully processed and written to data\transcript.json
Segmenting Audio: 100%|███████████████████████████████████████████████████████████████████████| 45/45 [00:06<00:00,  6.99segment/s, Text: Thank you                                                                  ]
[audio_splitter] Transcriptions have been successfully written to data\transcript.json
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 45/45 [00:00<00:00, 84.76 examples/s]
[pipeline] Using cuda: NVIDIA GeForce GTX 1650 with Max-Q Design
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 45/45 [00:13<00:00,  3.30 examples/s]
Word Error Rate (WER) of Whisper transcriptions against youtube subtitles: 10.219%
```

### See Also
- main.py for more examples
- [pyannote](https://github.com/pyannote/pyannote-audio)
- [ffmpeg](https://www.ffmpeg.org/)
- [Whisper](https://github.com/openai/whisper)
- YouTube: (auto-generated subtitles) [40 second elevator pitch](https://www.youtube.com/watch?v=4WEQtgnBu0I)
- YouTube: (manually transcribed) [2 minute two person job interview](https://www.youtube.com/watch?v=naIkpQ_cIt0)
- YouTube: (manually transcribed) [3 minute tutorial video](https://www.youtube.com/watch?v=VatNBZh66Po)

