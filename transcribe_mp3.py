from yt_downloader import download_mp3
from utilities import transcribe_mp3, save_transcript

# configuration settings
URL = "https://www.youtube.com/watch?v=4WEQtgnBu0I"
TRANSCRIPT_FILENAME = "base_elevator_pitch_transcript.txt"
MODEL_TYPE = "base"

if __name__ == "__main__":
    """
    transcribe_mp3.py
    Downloads the audio of a youtube video and transcribes it using whisper. Saves the transcription to a file transcriptions/TRANSCRIPTION_FILENAME.
    """

    download_mp3(URL)
    transcription = transcribe_mp3(model_type=MODEL_TYPE)
    save_transcript(transcript=transcription, transcript_filename=TRANSCRIPT_FILENAME)