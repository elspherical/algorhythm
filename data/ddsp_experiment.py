import ddsp
#import ddsp.training
#import ddsp.spectral_ops
import librosa
import numpy as np
#import soundfile as sf
import matplotlib.pyplot as plt


# Load audio
y, sr = librosa.load("calm.wav", sr=16000)  # DDSP expects 16kHz
print("hello")
"""# -----------------------------
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
plt.show()"""