import os 
import json

from torchaudio import load
from torchaudio.transforms import Resample
from datasets import Dataset
from pydub import AudioSegment
from tqdm import tqdm


def save_transcript(transcript=None, ds=None, transcript_filename="transcript.txt", verbose=False):
    """
    Saves a given transcript (from either string or dataset, not both) to file.
    Args:
        transcript (str, optional): Text transcriptions in string form. Defaults to None
        ds (Dataset, optional): Huggingface dataset object containing text transcriptions in batch[transcription] key. Defaults to None
        transcript_filename (str, optional): Output transcript text filename. Defaults to "transcript.txt"
        verbose (bool, optional): Toggles verbose logging. Defaults to False 
    Raises:
        TypeError: If neither 'transcript' nor 'ds' is provided.
    Returns:
        None
    """
    # check transcript passed in args
    if not transcript and not ds: 
        raise TypeError("[utilities]: No value passed for ds or transcript")
        
    os.makedirs("transcripts", exist_ok=True)
    transcript_path = os.path.join("transcripts", transcript_filename)

    with open(transcript_path, "w") as file:
        # save transcript string to file
        if transcript:
            for line in transcript.split("."):
                file.write(line[1:] + ".\n")
                if verbose: print(line[1:])

        # save transcript from dataset to file
        elif ds: 
            for batch in ds:
                text = batch["transcription"][1:]
                file.write(text + "\n")
                if verbose: print(text)

    print(f"[utilities] Transcript successfully written to {transcript_path}")


def resample_audio(batch):
    """
    Converts the audio in the given batch to the correct format for Whisper transcription.
    Args:
        batch (dict): A batch of audio for processing
    Returns:
        batch (dict): Batch of audio resampled to 16khz and flattened to numpy array
    """
    new_sample_rate = 16000
    audio_file = batch["audio"]
    waveform, old_sample_rate = load(audio_file)

    # resample to 16khz
    resampler = Resample(orig_freq=old_sample_rate, new_freq=new_sample_rate)
    waveform = resampler(waveform)

    # convert to numpy array
    batch["audio"] = {
        "array": waveform.squeeze().numpy(),  # convert the waveform to a numpy array
        "sampling_rate": new_sample_rate
    }
    return batch


def make_dataset(data_dir: str):
    """
    Creates a huggingface dataset given an input directory structured as follows:
    data/ 
        audio_1.mp3
        audio_2.mp3 
        ...
        transcript.json
    Args:
        data_dir (str): Path to the input directory
    Returns:
        dataset (Dataset): A huggingface dataset object created from the (audio, text transcription) data in input_dir
    """
    # open transcription data
    transcript_path = os.path.join(data_dir, "transcript.json")
    with open(transcript_path, "r") as f:
        transcription_data = json.load(f)

    dataset_dict = {
        "audio": [],
        "text": []
    }

    # format data into dataset dictionary
    for item in transcription_data:
        audio_file_path = os.path.join("data", item["audio"])
        dataset_dict["audio"].append(audio_file_path)
        dataset_dict["text"].append(item["text"])

    # make huggingface dataset 
    dataset = Dataset.from_dict(dataset_dict)

    # convert the audio to the correct format for Whisper transcription
    dataset = dataset.map(resample_audio)
    return dataset


def segment_audio_from_transcript(transcript_path: str, mp3_path: str, output_dir="data"):
    """
    Splits audio into segments given an MP3 file and a transcript.json file with start and end timestamps.
    Adds an "audio" key to each entry with the path to the mp3 file. 
    Args:
        transcript_path (str): Path to transcript.json file
        mp3_path (str): Path to the MP3 audio file to be split
        output_dir (str, optional): Directory where the audio segments and transcriptions will be saved. Defaults to 'data'
    Returns:
        None
    """
    with open(transcript_path, "r") as file:
        transcript = json.load(file)

    audio = AudioSegment.from_mp3(mp3_path)
    total_segments = len(transcript)
    os.makedirs(output_dir, exist_ok=True)

    # progress bar for audio splitting
    with tqdm(total=total_segments, desc="Segmenting Audio", unit="segment") as pbar:

        # iterate through transcript and split at given timestamps
        for i, segment_data in enumerate(transcript):
            segment_filename = f"audio_{i+1}.mp3"
            segment_path = os.path.join(output_dir, segment_filename)
            
            # split and save audio segment
            segment = audio[segment_data["start"]:segment_data["end"]]
            segment.export(segment_path, format="mp3")
            
            # add audio key
            segment_data["audio"] = segment_filename

            # update progress bar
            pbar.set_postfix_str(f"Text: {segment_data['text'][:75].ljust(75)}")
            pbar.update(1)  

    # save transcript.json
    with open(transcript_path, 'w') as file:
        json.dump(transcript, file, indent=2)

    print(f"[audio_splitter] Transcriptions have been successfully written to {transcript_path}")

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
    segment_audio_from_transcript()
