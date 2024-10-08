from whisper_yt.whisper import transcribe_mp3
from whisper_yt.yt_downloader import download_mp3
from settings import job_interview_url
from whisper_yt.utilities import save_transcript

if __name__ == "__main__":
    download_mp3(job_interview_url)
    diarized_transcript = transcribe_mp3("downloads/audio.mp3", 
                                         "openai/whisper-tiny", 
                                         diarize=True, 
                                         auth_token="your_token")
    save_transcript(diarized_transcript)