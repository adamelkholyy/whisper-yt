import shutil
import os
import torch

from utilities import split_audio_from_transcript, make_dataset, save_transcript
from yt_downloader import download_mp3, download_transcript, process_transcript
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from evaluate import load
from datasets import Dataset

"""
URLs:
    40 second elevator_pitch: https://www.youtube.com/watch?v=4WEQtgnBu0I
    40 minute Taylor Swift interview (manually transcribed): https://www.youtube.com/watch?v=m9Nkqm7FFgk
    50 minute Beck Institute CBT roleplay: https://www.youtube.com/watch?v=ac5Jct33oUU
"""

# configuration settings
URL = "https://www.youtube.com/watch?v=ac5Jct33oUU"
EXPERIMENT_TITLE = "beck_cbt_roleplay_2"

MODEL_TYPE = "openai/whisper-tiny"
DATASET_DIR = "datasets"

TRANSCRIPT_FILENAME = EXPERIMENT_TITLE + ".txt"
DATASET_NAME = EXPERIMENT_TITLE + "_"+ MODEL_TYPE.replace("/", "-") + "_ds"

REMOVE_EMPTY_REFERENCES = True
SAVE_TRANSCRIBED_DS = False

"""
Garbage collector should remain on as transcription will break if previous audio remains in 'data/' output_dir!
Bearing this in mind however it can be turned off in order to inspect audio segmentation.
"""
GARBAGE_COLLECTOR = True 

def download_and_segment_yt_data(url: str, audio_dir="downloads", data_dir="data"):
    """
    Downloads mp3 audio and vtt transcript from youtube URL and splits data into 
    audio segments. Creates a transcriptions.json file containing corresponding 
    (audio segment, transcript segment) info needed for creating a huggingface dataset.
    
    Args:
        url (str): Youtube video URL
        audio_dir (str, optional): Directory to save downloaded audio. Defaults to 'downloads'.
        data_dir (str, optional: Directory to save segmented audio data. Defaults to 'data'.
    
    Returns:
        None
    """

    download_mp3(url, output_dir=audio_dir)
    download_transcript(url, output_dir=audio_dir)
    transcript = process_transcript(input_dir=audio_dir)
    mp3_path = os.path.join(audio_dir, "audio.mp3")
    split_audio_from_transcript(transcript, mp3_path=mp3_path, output_dir=data_dir)


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


def get_whisper_transcription(ds: Dataset):
    """
    Gets Whisper transcriptions of audio from dataset.
    
    Args:
        ds (Dataset): A dataset containing audio and ground truth transcription data

    Returns:
        transcribed_ds (Dataset): The original dataset with added "reference", "prediction"
        and "tanscription" keys containing Whisper transcriptions
    """

    # load GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pipeline] Using {device}: {torch.cuda.get_device_name(0)}")
   
    # load model and processor
    processor = WhisperProcessor.from_pretrained(MODEL_TYPE)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_TYPE).to(device)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

    # make transcription inference
    transcribed_ds = ds.map(lambda batch: batch_inference(batch, model, processor, device))
    return transcribed_ds


def filter_empty_references(references, predictions):
    """
    Filters out empty references (blocks of silence) from transcript.
    
    Args:
        references (list): List of references (i.e. ground truth transcriptions)
        predictions (list): List of Whisper transcriptions

    Returns:
        filtered_references (list): List of references with empty references removed
        filtered_predictions (list): List of Whisper transcriptions with corresponding 
        empty references removed
    """

    filtered_references, filtered_predictions = [], []
    for ref, pred in zip(references, predictions):
        # only include non-empty references
        if ref.strip():  
            filtered_references.append(ref)
            filtered_predictions.append(pred)
    return filtered_references, filtered_predictions


if __name__ == "__main__":
    """
    evaluate_yt_transcription_pipeline.py 
        1. Downloads the audio and transcript from a given youtube video URL
        2. Splits audio and transcript into segments and saves as a huggingface dataset in DATASET_DIR\DATASET_NAME
        3. Uses whisper for english transcription on dataset
        4. Calculates Word Error Rate of whisper transcription against original youtube transcript 
    """

    # download and segment youtube audio and transcript
    download_and_segment_yt_data(URL)

    # save as huggingface dataset
    ds = make_dataset(input_dir="data")
    dataset_path = os.path.join(DATASET_DIR, DATASET_NAME)
    ds.save_to_disk(dataset_path=dataset_path)

    # delete downloads and data
    if GARBAGE_COLLECTOR:
        shutil.rmtree("downloads") 
        shutil.rmtree("data")

    # save whisper generated transcript to file
    transcribed_ds = get_whisper_transcription(ds)
    save_transcript(ds=transcribed_ds, transcript_filename=TRANSCRIPT_FILENAME)

    # save whisper transcriptions to separate dataset
    if SAVE_TRANSCRIBED_DS:
        transcribed_ds_path = os.path.join(dataset_path, "_TRANSCRIBED")
        transcribed_ds.save_to_disk(transcribed_ds_path)

    references = transcribed_ds['reference']
    predictions = transcribed_ds['prediction']

    # filter out blocks of silence
    if REMOVE_EMPTY_REFERENCES:
        references, predictions = filter_empty_references(references, predictions)

    # calculate final WER
    wer_function = load("wer")
    wer_score = wer_function.compute(references=references, predictions=predictions)
    print(f"[pipeline] Word Error Rate (WER) of Whisper transcriptions against youtube subtitles: {100 * wer_score}%")