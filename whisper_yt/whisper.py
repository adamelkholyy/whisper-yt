import torch 
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from datasets import Dataset
from whisper_yt.diarize import diarize_transcript

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
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[pipeline] Using {device}: {torch.cuda.get_device_name(0)}")
    except: 
        device = torch.device("cpu")
        print(f"[pipeline] Torch not built with Cuda, using {device}")

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


def batch_inference(batch, model, processor, device):
    """
    Performs Whisper transcription inference on a batch of data. 
    Returns:
        batch (dict): original batch with added "reference", "transcription" and "prediction" keys.
    """
    # tokenize ground truth text
    batch["reference"] = processor.tokenizer._normalize(batch['text'])

    # make model predictions
    audio = batch["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features

    # no_grad: no gradient calculations i.e. no training backward pass, only inference
    with torch.no_grad(): 
        predicted_ids = model.generate(input_features.to(device))[0]
    
    # tokenize whisper transcription and add to batch
    batch["transcription"] = processor.decode(predicted_ids, skip_special_tokens=True)
    batch["prediction"] = processor.tokenizer._normalize(batch["transcription"])
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

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[pipeline] Using {device}: {torch.cuda.get_device_name(0)}")
    except: 
        device = torch.device("cpu")
        print(f"[pipeline] Torch not built with Cuda, using {device}")


    # load model and processor
    processor = WhisperProcessor.from_pretrained(model_type)
    model = WhisperForConditionalGeneration.from_pretrained(model_type).to(device)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    model.generation_config.forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(language="en", task="transcribe")

    # make transcription inference
    transcribed_ds = ds.map(lambda batch: batch_inference(batch, model, processor, device))
    return transcribed_ds
