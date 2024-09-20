import whisper
import os 
import json
import torchaudio

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


def transcribe_mp3(mp3_path: str, model_type="base"):
    """
    Transcribes a given mp3 file using the specified whisper model.
    Args:
        mp3_path (str, optional): The path to the mp3 file to be transcribed. Defaults to 'downloads/audio.mp3'
        model_type (str, optional): The whisper model used to transcribe. Defaults to 'base'
    Returns:
        transcription["text"] (str): The whisper transcription of the audio file
    """
    whisper_model = whisper.load_model(model_type)
    print(f"[utilities] whisper-{model_type} loaded, transcribing {mp3_path}. Note: This could take a long time depending on the length of the audio")
    transcription = whisper_model.transcribe(mp3_path)
    print(f"[utilities] {mp3_path} successfully transcribed by whisper-{model_type}")
    return transcription["text"]


def convert_audio_for_whisper(batch):
    """
    Converts the audio in the given batch to the correct format for Whisper transcription.
    Args:
        batch (dict): A batch of audio for processing
    Returns:
        batch (dict): Batch of audio resampled to 16khz and flattened to numpy array
    """
    new_sample_rate = 16000
    audio_file = batch["audio"]
    waveform, old_sample_rate = torchaudio.load(audio_file)

    # resample to 16khz
    resampler = torchaudio.transforms.Resample(orig_freq=old_sample_rate, new_freq=new_sample_rate)
    waveform = resampler(waveform)

    # convert to numpy array
    batch["audio"] = {
        "array": waveform.squeeze().numpy(),  # Convert the waveform to a numpy array
        "sampling_rate": new_sample_rate
    }
    return batch

def make_dataset(data_dir: str):
    """
    Creates a huggingface dataset given an input directory structred as follows:
    data/ 
        audio_1.mp3
        audio_2.mp3 
        ...
        transcript.json
    Args:
        input_dir (str, optional): Path of the input directory. Defaults to 'data'
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
    dataset = dataset.map(convert_audio_for_whisper)
    return dataset

# CHANGE ARGS DOCS
def segment_audio_from_transcript(transcript_path: str, mp3_path: str, output_dir="data"):
    """
    Splits audio into segments given an MP3 file and a transcript with start and end timestamps.
    Saves a `transcriptions.json` file with data for each segment in the following format:
    {
        "audio": "audio_path.mp3",
        "text": "Ground truth text for audio_path.mp3"
    }
    Args:
        transcript (list): A list of tuples where each tuple contains (start, end, text):
            - start (int): Start timestamp in milliseconds
            - end (int): End timestamp in milliseconds
            - text (str): The ground truth transcription for the audio segment
        mp3_path (str, optional): Path to the MP3 audio file to be split. Defaults to 'downloads/audio.mp3'
        output_dir (str, optional): Directory where the audio segments and transcriptions will be saved. Defaults to 'data'
    Returns:
        None
    """

    with open(transcript_path, "r") as file:
        transcript = json.load(file)

    audio = AudioSegment.from_mp3(mp3_path)
    os.makedirs(output_dir, exist_ok=True)
    total_segments = len(transcript)

    # progress bar for audio splitting
    with tqdm(total=total_segments, desc="Segmenting Audio", unit="segment") as pbar:

        # iterate through transcript and split at given timestamps
        for i, segment_data in enumerate(transcript):
            segment_filename = f"audio_{i+1}.mp3"
            segment_path = os.path.join(output_dir, segment_filename)
            
            # split and save audio segment
            segment = audio[segment_data["start"]:segment_data["end"]]
            segment.export(segment_path, format="mp3")
            
            # append json data
            segment_data["audio"] = segment_filename

            # update progress bar
            pbar.set_postfix_str(f"Text: {segment_data['text'][:75].ljust(75)}")
            pbar.update(1)  

    # write transcriptions.json to file
    with open(transcript_path, 'w') as file:
        json.dump(transcript, file, indent=2)

    print(f"[audio_splitter] Transcriptions have been successfully written to {transcript_path}")

if __name__ == "__main__":
    segment_audio_from_transcript()
