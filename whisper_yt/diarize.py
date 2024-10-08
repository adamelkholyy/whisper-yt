import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment

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
        transcription (list): list of dicts containing diarized transcription of the audio file 
        (e.g. {'speaker': 'SPEAKER_00', 'text': ' It is a truth universally acknowledged, that a single man in possession of a good fortune, 
        must be in want of a wife', ...})
    """
    # auth token validation
    if auth_token == None: 
        raise TypeError("[whisper]: No value passed for auth_token. Pyannote does not work without a valid HuggingFace auth token, see https://github.com/pyannote/pyannote-audio for more details.")

    # diarize using pyannote
    print(f"[whisper] Diarizing {mp3_path}. Note: This could take a long time depending on the length of the audio")
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=auth_token)
    diarization_pipeline.to(device)

    # diarize and flatten into contiguous speaker blocks 
    diarization = diarization_pipeline(mp3_path, num_speakers=2)
    diarization = diarization.support(1)

    text_segments = [(Segment(line["timestamp"][0], line["timestamp"][1]), line["text"]) for line in transcription]
    # assign speaker: https://github.com/yinruiqing/pyannote-whisper/blob/main/pyannote_whisper/utils.py
    transcription = [{"speaker": diarization.crop(seg).argmax(), "text": text} for seg, text in text_segments]

    return transcription


