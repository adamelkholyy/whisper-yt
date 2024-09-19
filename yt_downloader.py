import os 
import re
import subprocess

def download_mp3(url: str, output_dir='downloads', audio_filename='audio.mp3'):
    os.makedirs(output_dir, exist_ok=True)
    audio_output_path = os.path.join(output_dir, audio_filename)

    audio_command = [
        'yt-dlp',
        '-x',  # extract audio only
        '--audio-format', 'mp3',  
        '-o', f'{audio_output_path}', 
        url
    ]

    try:
        subprocess.run(audio_command, check=True)
        print(f"[yt_downloader] Audio download complete! MP3 saved at {audio_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[yt_downloader] Error downloading audio: {e}")
    

def download_transcript(url: str, output_dir='downloads', transcript_filename='raw_transcript'):
    os.makedirs(output_dir, exist_ok=True)
    transcript_output_path = os.path.join(output_dir, transcript_filename)

    transcript_command = [
        'yt-dlp',
        '--write-sub',  # download manual subtitles (if available)
        '--sub-format', 'vtt',  
        '--skip-download',  # skip the video download, only get subtitles
        '-o', f'{transcript_output_path}',  
        url
    ]

    try:
        subprocess.run(transcript_command, check=True)
        print(f"[yt_downloader] Transcript download complete! Transcript saved at {transcript_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[yt_downloader] Error downloading transcript: {e}")


def timestamp_to_milliseconds(timestamp: str):
    pattern = r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})"
    match = re.match(pattern, timestamp)
    if match:
        hours, minutes, seconds, milliseconds = map(int, match.groups())
        total_milliseconds = (hours * 3600 * 1000) + (minutes * 60 * 1000) + (seconds * 1000) + milliseconds
        return total_milliseconds
    else:
        raise ValueError(f"[yt_downloader] Invalid time format: {timestamp}")
    
   
def process_transcript(input_dir="downloads", transcript_filename="raw_transcript.en.vtt"):
    transcript_path = os.path.join(input_dir, transcript_filename)
    with open(transcript_path, 'r') as file:
        content = file.read()

    # todo: comment about yt transcript files handling start weirdly
    initial_timestamps = content.split("\n")[4]
    content = content.replace("\n ", "")
    content = content.split("\n\n")
    extracted_data = [group.split("\n")[:2] for group in content if len(group.split("\n")) > 2][1:]
    transcript = []

    pattern = r"\d+:\d+:\d*\.\d+"
    start_timestamp, end_timestamp = re.findall(pattern, initial_timestamps)

    for (timestamp_data, text) in extracted_data: 
        transcript.append((timestamp_to_milliseconds(start_timestamp), timestamp_to_milliseconds(end_timestamp), text))
        start_timestamp, end_timestamp = re.findall(pattern, timestamp_data)

    return transcript


if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=4WEQtgnBu0I"
    download_mp3(video_url)
    download_transcript(video_url)
    transcript = process_transcript()
    for t in transcript: print(f"[yt_downloader] {t}")


