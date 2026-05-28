import torch
import torchaudio
#from audiocraft.models import MusicGen
#from audiocraft.data.audio import audio_write
import sys
from video2text import main

sys.argv = [
    "video_to_music.py",
    "vid1.mp4"
]

top_prompt = main()
print(top_prompt)