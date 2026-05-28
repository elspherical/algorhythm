import os
import csv
import random
import requests
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import yt_dlp
import logging

#yt-dlp --cookies-from-browser chrome --cookies cookies.txt "https://www.youtube.com/"

CSV_URL = "https://huggingface.co/datasets/igorriti/ambience-audio/resolve/main/train.csv"
OUTPUT_DIR = "clips"
CLIPS_PER_VIDEO = 20
CLIP_DURATION = 10   # seconds
WINDOW_SIZE = 180    # seconds of segment to download (3 minutes) -- download before split
MAX_WORKERS = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)


LOG_FILE = "download_log_2.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# could use for video too
def load_csv():
    logger.info("Downloading CSV...")
    r = requests.get(CSV_URL)
    r.raise_for_status()

    rows = []
    reader = csv.DictReader(r.text.splitlines())
    for row in reader:
        rows.append(row)

    logger.info(f"Loaded {len(rows)} rows from CSV")
    return rows


def get_video_duration(url, video_id):
    logger.info(f"[{video_id}] Getting video duration...")
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        # using my other account.., had to download cookies separately
        "cookies_from_browser": "chrome: Profile 7",
        "cookiefile": "cookies.txt",

        "sleep_interval": 2,
        "max_sleep_interval": 5,

        # because youtube was rate-limiting
        "socket_timeout": 10,
        "retries": 2,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        dur = info.get("duration", None)
        logger.info(f"[{video_id}] Duration = {dur}s")
        return dur

    except Exception as e:
        logger.error(f"[{video_id}] Failed to get duration: {e}")
        return None

# download segments using download_sections
def download_segment(url, video_id, start_time, end_time, temp_mp4):
    logger.info(f"[{video_id}] Downloading segment {start_time}-{end_time} → {temp_mp4}")

    ydl_opts = {
        "quiet": True,
        "format": "mp4",
        "download_sections": f"*{start_time}-{end_time}",
        "outtmpl": temp_mp4,
        "cookies_from_browser": "chrome: Profile 7",
        "merge_output_format": "mp4",
        "cookiefile": "cookies.txt",

        "sleep_interval": 2,
        "max_sleep_interval": 5,

        "socket_timeout": 10,
        "retries": 2,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if not os.path.exists(temp_mp4):
            raise RuntimeError("yt-dlp produced no output file")

        logger.info(f"[{video_id}] Segment download successful")
        return True

    except Exception as e:
        logger.error(f"[{video_id}] Failed segment download: {e}")
        return False

# cut pre-downloaded clip into 10-second bits
def cut_clip(temp_mp4, out_path, start_t):
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_t),
        "-t", str(CLIP_DURATION),
        "-i", temp_mp4,
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "44100",
        "-b:a", "192k",
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def process_video(row, index):
    video_id = row["id"]
    yt_url = f"https://www.youtube.com/watch?v={video_id}"

    logger.info(f"\n=== Processing video {index}: {video_id} ===")

    existing = 0
    for i in range(1, CLIPS_PER_VIDEO + 1):
        if os.path.exists(os.path.join(OUTPUT_DIR, f"{index}_{i}.mp3")):
            existing += 1

    if existing == CLIPS_PER_VIDEO:
        logger.info(f"[{video_id}] All clips already exist, skipping video.")
        return

    duration = get_video_duration(yt_url, video_id)
    if duration is None or duration < (CLIP_DURATION * CLIPS_PER_VIDEO):
        logger.warning(f"[{video_id}] Video too short or failed, skipping.")
        return

    # random segment for clip download
    max_start = duration - WINDOW_SIZE
    segment_start = random.randint(0, max_start)
    segment_end = segment_start + WINDOW_SIZE

    temp_mp4 = f"__temp_{video_id}.mp4"

    ok = download_segment(yt_url, video_id, segment_start, segment_end, temp_mp4)
    if not ok:
        return

    clip_starts = [random.randint(0, WINDOW_SIZE - CLIP_DURATION)
                   for _ in range(CLIPS_PER_VIDEO)]
    
    for i, cs in enumerate(clip_starts, start=1):
        out_name = f"{index}_{i}.mp3"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        logger.info(f"[{video_id}] Cutting clip {i}: {cs}-{cs+CLIP_DURATION} → {out_name}")
        cut_clip(temp_mp4, out_path, cs)

    if os.path.exists(temp_mp4):
        os.remove(temp_mp4)

    logger.info(f"[{video_id}] Completed {CLIPS_PER_VIDEO} clips.")


rows = load_csv()

futures = []
# parallelize as much as possible without errors
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    for idx, row in enumerate(rows, start=1):
        futures.append(executor.submit(process_video, row, idx))

    for _ in tqdm(as_completed(futures), total=len(futures)):
        pass

logger.info("All videos processed.")

import ddsp
import ddsp.training
import ddsp.spectral_ops
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from google.colab import files

# -----------------------------
# 3️⃣ Upload an audio file
# -----------------------------
uploaded = files.upload()
audio_file = list(uploaded.keys())[0]

# Load audio
y, sr = librosa.load(audio_file, sr=16000)  # DDSP expects 16kHz

# -----------------------------
# 4️⃣ Extract F0 and loudness
# -----------------------------
# F0
f0, voiced_flag, voiced_probs = librosa.pyin(y, 
                                             fmin=librosa.note_to_hz('C2'),
                                             fmax=librosa.note_to_hz('C7'),
                                             sr=sr)
f0 = np.nan_to_num(f0)  # Replace unvoiced frames with 0

# Loudness (RMS)
S = librosa.feature.melspectrogram(y, sr=sr, n_fft=1024, hop_length=256)
loudness = librosa.power_to_db(np.sum(S, axis=0))
# Normalize loudness to [-1,1]
loudness = (loudness - np.min(loudness)) / (np.max(loudness) - np.min(loudness))
loudness = loudness * 2 - 1

print("f0 shape:", f0.shape, "loudness shape:", loudness.shape)

# -----------------------------
# 5️⃣ Prepare timbre embedding
# -----------------------------
# Simple random timbre embedding (16 dims)
timbre_dim = 16
timbre = np.random.randn(timbre_dim)

# Optionally create multiple timbre vectors to experiment
timbre2 = np.random.randn(timbre_dim)

# -----------------------------
# 6️⃣ Define DDSP synthesizer
# -----------------------------
synth = ddsp.synths.Additive(n_harmonics=32, n_samples=len(f0)*256)

# -----------------------------
# 7️⃣ Generate audio from F0, loudness, timbre
# -----------------------------
# Prepare tensors
import torch

f0_tensor = torch.tensor(f0, dtype=torch.float32).unsqueeze(0)   # (1, T)
loudness_tensor = torch.tensor(loudness, dtype=torch.float32).unsqueeze(0)  # (1, T)
timbre_tensor = torch.tensor(timbre, dtype=torch.float32).unsqueeze(0)  # (1, latent_dim)

# Upsample timbre if needed to match f0 frames
# Here we keep it constant for the clip
timbre_tensor = timbre_tensor.repeat(1, f0_tensor.shape[1], 1)

# Generate audio
with torch.no_grad():
    audio_out = synth(f0_tensor, loudness_tensor, timbre_tensor)
    audio_out = audio_out.numpy().squeeze()

# -----------------------------
# 8️⃣ Save output
# -----------------------------
sf.write("ddsp_timbre_output.wav", audio_out, sr)
files.download("ddsp_timbre_output.wav")

# -----------------------------
# 9️⃣ Optional: Visualize F0 and Loudness
# -----------------------------
plt.figure(figsize=(12,4))
plt.subplot(2,1,1)
plt.plot(f0)
plt.title("F0 (Pitch)")
plt.subplot(2,1,2)
plt.plot(loudness)
plt.title("Loudness")
plt.show()