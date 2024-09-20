import torch 
from whisper import load_model
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset


def transcribe_mp3(mp3_path: str, model_type="base"):
    """
    Transcribes a given mp3 file using the specified whisper model.
    Args:
        mp3_path (str): The path to the mp3 file to be transcribed
        model_type (str, optional): The whisper model used to transcribe. Defaults to 'base'
    Returns:
        text (str): The whisper transcription of the audio file
    """
    whisper_model = load_model(model_type)
    print(f"[utilities] whisper-{model_type} loaded, transcribing {mp3_path}. Note: This could take a long time depending on the length of the audio")
    transcription = whisper_model.transcribe(mp3_path)
    print(f"[utilities] {mp3_path} successfully transcribed by whisper-{model_type}")
    return transcription["text"]


def batch_inference(batch, model, processor, device):
    """
    Performs inference on a batch of data. 
    
    Args:
        batch (dict): A dictionary containing the audio and text data.

    Returns:
        batch (dict): The original batch with added "reference", "transcription", 
        and "prediction" keys.
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
    Gets Whisper transcriptions of audio from dataset.
    Args:
        ds (Dataset): A dataset containing audio and ground truth transcription data
        model_type (str): Whisper model used for transcription. Defaults to "openai/whisper-base"
    Returns:
        transcribed_ds (Dataset): The original dataset with added "reference", "prediction"
        and "tanscription" keys containing Whisper transcriptions
    """
    # load GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pipeline] Using {device}: {torch.cuda.get_device_name(0)}")
   
    # load model and processor
    processor = WhisperProcessor.from_pretrained(model_type)
    model = WhisperForConditionalGeneration.from_pretrained(model_type).to(device)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

    # make transcription inference
    transcribed_ds = ds.map(lambda batch: batch_inference(batch, model, processor, device))
    return transcribed_ds

