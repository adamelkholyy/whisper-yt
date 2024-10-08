import shutil
import os

from whisper_yt.utilities import make_dataset, save_transcript, filter_empty_references
from whisper_yt.whisper import get_whisper_transcription
from whisper_yt.yt_downloader import download_and_preprocess_yt
from evaluate import load
from settings import *

if __name__ == "__main__":
    """
    Example pipeline:
        1. Downloads the audio and transcript from a given youtube video URL
        2. Splits audio and transcript into segments and saves as a huggingface dataset in DATASET_DIR\\DATASET_NAME
        3. Uses whisper for English transcription on dataset
        4. Calculates Word Error Rate of whisper transcription against original youtube transcript 
    """
    print(f"[pipeline] Initialised. Preparing to download youtube data.")

    # garbage collection
    if GARBAGE_COLLECTOR: 
        if os.path.isdir('downloads'): shutil.rmtree("downloads") 
        if os.path.isdir('data'): shutil.rmtree("data")
        print(f"[pipeline] Garbage collector: downloads/ and data/ successfully deleted")
    else:
        print(f"[pipeline] WARNING: Garbage Collection is turned off. Errors will be caused if data/ and downloads/ are not empty!")

    # download and segment youtube audio and transcript
    download_and_preprocess_yt(URL)

    # save as huggingface dataset
    ds = make_dataset(data_dir="data")
    dataset_path = os.path.join(DATASET_DIR, TITLE+"_ds")
    ds.save_to_disk(dataset_path=dataset_path)

    # generate whisper transcript and save to to file
    transcribed_ds = get_whisper_transcription(ds)
    save_transcript(ds=transcribed_ds, transcript_filename=TITLE+".txt")

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
    wer_score = 100 * wer_function.compute(references=references, predictions=predictions)
    print(f"[pipeline] Word Error Rate (WER) of Whisper transcriptions against youtube subtitles: {wer_score:.3f}%")