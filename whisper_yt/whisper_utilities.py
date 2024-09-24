import torch 
from pyannote.audio import Pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from datasets import Dataset


def transcribe_mp3(mp3_path: str, model_type="openai/whisper-base", diarize=False, auth_token=None):
    """
    Transcribes a given mp3 file using the specified whisper model.
    Args:
        mp3_path (str): path to the mp3 file to be transcribed
        model_type (str, optional): Whisper model used to transcribe. Defaults to 'openai/whisper-base'
        diarize (bool, optional): boolean to toggle diariaztion. Defaults to False 
        auth_token (str, optional): Huggingface authorization token required for diariazation. Defaults to None
    Returns:
        transcription (list): Whisper transcription of the audio file as a list of dicts, where each dict is a line with timestamps and text e.g. {'timestamp': (0, 3000), 
        'text': 'It is a truth universally acknowledged'}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[whisper] Using {device}: {torch.cuda.get_device_name(0)}")

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model_type,
        chunk_length_s=30,
        device=device,
    )

    print(f"[whisper] {model_type} loaded, transcribing {mp3_path}. Note: This could take a long time depending on the length of the audio")
    transcription = asr_pipeline(mp3_path, batch_size=8, return_timestamps=True)["chunks"]
    print(f"[whisper] {mp3_path} successfully transcribed by whisper-{model_type}")

    if diarize:
        transcription = diarize_transcript(transcription, mp3_path, device, auth_token)

    return transcription


def diarize_transcript(transcription: list, mp3_path: str, device: torch.device, auth_token: str):
    """
    Transcribes and diarizes an mp3 file using whisper and pyannote
    Args:
        transcription (list): whisper transcription of mp3 file
        mp3_path (str): path to mp3 audio file 
        device (torch.device): pytorch device used for diariazation
        auth_token (str): Huggingface authorisation token. Note that the terms and conditions of pyannote have to be accepted in order to use the module. 
            More info can be found at https://github.com/pyannote/pyannote-audio.
    Returns:
        transcription (str): diarized transcription of the audio file (e.g. {'text': 'SPEAKER_00: It is a truth universally acknowledged, that a single man in possession of a good fortune, 
        must be in want of a wife', ...})
    """
    # auth token validation
    if auth_token == None: 
        raise TypeError("[whisper]: No value passed for auth_token. Pyannote does not work without a valid HuggingFace auth token, see https://github.com/pyannote/pyannote-audio for more details.")

    # diarize using pyannote
    print(f"[whisper] Diarizing {mp3_path}. Note: This could take a long time depending on the length of the audio")
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=auth_token)
    diarization_pipeline.to(device)
    diarization = diarization_pipeline(mp3_path)
    
    # flatten contiguous lines from the same speaker into blocks
    speaker_blocks = flatten_speakers(diarization)

    # match transcript lines to speaker diarizations
    for line in transcription: 
        start, end = line['timestamp']
        speaker = assign_speaker(start, end, speaker_blocks)
        line["text"] = f" {speaker}:{line["text"]}"

    return transcription


def get_timestamp_difference(start: float, end: float, block_start: float, block_end: float):
    """
    Calculates the difference between 2 given start and end times (of a line of speech and of a speaker block). If the start and end times of the line of speech are within 
    the block then return 0 as we can confidently assign the speaker 
    Returns:
        distance (float): 0 if speech within speaker block, else distance between line of speech and speaker block
    """
    if block_start <= start <= block_end and block_start <= end <= block_end:
        return 0
    return abs(block_start - start) + abs(block_end - end)


def assign_speaker(start: float, end: float, blocks: list):
    """
    Assigns a speaker given the start and end timestamp and a list of flattened speaker blocks by calculating the speaker assignment closest to the start and end timestamps
    Args:
        start (float): start timestamp in ms 
        end (float): end timestamp in ms 
        blocks (list): list of flattened speaker blocks
    Returns:
        speaker (str): the assigned speaker
    """
    speaker = None
    best_dist = 1e8

    # assign smallest distance speaker 
    for (block_start, block_end, block_speaker) in blocks: 
        distance = get_timestamp_difference(start, end, block_start, block_end)
        if distance == 0:
            return block_speaker
        elif distance < best_dist:
            best_dist = distance
            speaker = block_speaker
    return speaker


def flatten_speakers(diarization: Pipeline):
    """
    Flattens contiguous lines of speech from the same speaker into one singular block for speaker assignment e.g.
        SPEAKER_00: (0, 3000) It is a truth universally acknowledged
        SPEAKER_00: (3000, 6000) that a single man in possession of a good fortune
        SPEAKER_00: (6000, 9000) must be in want of a wife
        --> SPEAKER_00: (0, 9000) It is a truth universally acknowledged that a single man in possession of a good fortune must be in want of a wife
    Args:
        diarization (Pipeline): diariazed mp3 file in form of a pyannote Pipeline object
    Returns:
        blocks (list): list of flattened speaker blocks, where each block is a tuple in the form (start time, end time, speaker)
    """
    blocks = []
    blocking = False
    prev_speaker = None
    block_start = 0
    block_end = 0

    # contiguous speaker block flattening algorithm
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if prev_speaker == speaker:
            if blocking: 
                block_end = turn.end
            else:
                blocking = True
        else:
            if blocking:
                blocks.append((block_start, block_end, prev_speaker))
                blocking = False
            block_start = turn.start
        prev_speaker = speaker
    if blocking:
        blocks.append((block_start, block_end, prev_speaker))
    return blocks


def batch_inference(batch, model, processor, device):
    """
    Performs Whisper transcription inference on a batch of data. 
    Returns:
        batch (dict): original batch with added "reference", "transcription" and "prediction" keys.
    """
    # tokenize ground truth text
    batch["reference"] = processor.tokenizer.normalize(batch['text'])

    # make model predictions
    audio = batch["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features

    # no_grad: no gradient calculations i.e. no training backward pass, only inference
    with torch.no_grad(): 
        predicted_ids = model.generate(input_features.to(device))[0]
    
    # tokenize whisper transcription and add to batch
    batch["transcription"] = processor.decode(predicted_ids, skip_special_tokens=True)
    batch["prediction"] = processor.tokenizer.normalize(batch["transcription"])
    return batch


def get_whisper_transcription(ds: Dataset, model_type="openai/whisper-base"):
    """
    Gets Whisper transcriptions from dataset containing audio and ground truth text
    Args:
        ds (Dataset): dataset containing audio and ground truth transcription data
        model_type (str): Whisper model used for transcription. Defaults to "openai/whisper-base"
    Returns:
        transcribed_ds (Dataset): original dataset with added "reference", "prediction" and "tanscription" keys containing Whisper transcriptions
    """
    # load GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pipeline] Using {device}: {torch.cuda.get_device_name(0)}")
   
    # load model and processor
    processor = WhisperProcessor.from_pretrained(model_type)
    model = WhisperForConditionalGeneration.from_pretrained(model_type).to(device)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    model.generation_config.forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(language="en", task="transcribe")

    # make transcription inference
    transcribed_ds = ds.map(lambda batch: batch_inference(batch, model, processor, device))
    return transcribed_ds

if __name__=="__main__":
    from yt_downloader import download_mp3
    from utilities import save_transcript

    url = "https://www.youtube.com/watch?v=JSLhP8i-5U0"
    token = ""

    download_mp3(url)
    transcript = transcribe_mp3("downloads/audio.mp3", diarize=True,auth_token=token )   
    save_transcript(transcript)