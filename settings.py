# useful example URLs
elevator_pitch_url = "https://www.youtube.com/watch?v=4WEQtgnBu0I" # 40 seconds, elevator pitch, auto-generated subtitles
job_interview_url = "https://www.youtube.com/watch?v=naIkpQ_cIt0"  # 2 minutes, two person job interview, manually transcribed
tutorial_video_url = "https://www.youtube.com/watch?v=VatNBZh66Po" # 3 minutes, tutorial video, manually transcribed

# configuration settings
URL = elevator_pitch_url
TITLE = "elevator_pitch"

MODEL_TYPE = "openai/whisper-tiny"
DATASET_DIR = "datasets"
REMOVE_EMPTY_REFERENCES = True
SAVE_TRANSCRIBED_DS = False

"""
Garbage collector should remain on as transcription will break if previous audio remains in 'data/' output_dir!
Bearing this in mind however it can be turned off in order to inspect audio segmentation.
"""
GARBAGE_COLLECTOR = True 
