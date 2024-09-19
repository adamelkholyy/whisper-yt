import os 
import re
import yt_dlp


def download_mp3(url: str, output_dir='downloads', audio_filename='audio'):
    os.makedirs(output_dir, exist_ok=True)
    audio_output_path = os.path.join(output_dir, audio_filename)

    ydl_opts = {
        'format': 'bestaudio/best',  
        'outtmpl': audio_output_path,  
        'postprocessors': [{  # force mp3 filetype
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'noplaylist': True,  
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"[yt_downloader] Audio download complete! MP3 saved at {audio_output_path}")


def download_transcript(url: str, output_dir='downloads', transcript_filename='raw_transcript'):
    os.makedirs(output_dir, exist_ok=True)
    transcript_output_path = os.path.join(output_dir, transcript_filename)

    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'subtitlesformat': 'vtt',
        'outtmpl': transcript_output_path,
        'noplaylist': True,
    }
   
    # attempt to download manual subtitles
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    # if transcript exists then manual subtitles were downloaded successfully 
    if os.path.exists(transcript_output_path):
        print(f"[yt_downloader] Manual subtitles download complete! Transcript saved at {transcript_output_path}.vtt")

    # otherwise there are no manual subs; download auto-generated subs instead
    else:
        print(f"[yt_downloader] Manual subtitles not found, attempting to download auto-generated subtitles")
        ydl_opts['writesubtitles'] = False
        ydl_opts['writeautomaticsub'] = True  
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"[yt_downloader] Auto-generated subtitles download complete! Transcript saved at {transcript_output_path}.vtt")



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


