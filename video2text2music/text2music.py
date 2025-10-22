import torch
import torchaudio
#from audiocraft.models import MusicGen
#from audiocraft.data.audio import audio_write
from main import video2text

from moviepy.editor import VideoFileClip, AudioFileClip

def add_music(text_descriptions, video, segment_duration=5):
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    model.set_generation_params(duration=segment_duration)

    audio_segments = []

    for i, text in enumerate(text_descriptions):
        print(f"iterating on {text}")
        generated_audio = model.generate([text])[0]
        audio_segments.append(generated_audio)
        
    final_audio = torch.cat(audio_segments, dim=1)

    audio_write("generated", wav = final_audio, sample_rate=model.sample_rate)
    

def overlay_video_with_audio(video, audio):
    video_clip = VideoFileClip(video)
    audio_clip = AudioFileClip(audio)

    video_clip = video_clip.set_audio(audio_clip)

    output_path = f"{video}_with_audio.mp4"
    video_clip.write_videofile(output_path)


overlay_video_with_audio('vid1.mp4', 'generated.wav')
text_descriptions = video2text.main("vid1.mp4")
#text_descriptions = ["rock music", "energetic EDM", "sad jazz"] # get from video2text
video_file = "vid1.mp4"
add_music(text_descriptions, video_file)


"""
model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.set_generation_params(duration=8)  # generate 8 seconds.
wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples
descriptions = ['happy rock', 'energetic EDM', 'sad jazz']
wav = model.generate(descriptions)  # generates 3 samples.

melody, sr = torchaudio.load('./assets/bach.mp3')
# generates using the melody from the given audio and the provided descriptions.
wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
    """