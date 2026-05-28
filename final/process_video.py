import torch
from video_to_emotion import VideoToEmotion
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import os
import ffmpeg
import tempfile


def main(video_path, output_file):
    model = VideoToEmotion(input_dim=512, hidden_dim=512, output_dim=13)
    state_dict = torch.load("im_embed_to_emotion.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model_clip.eval()

    def generate_clip_embedding(image: Image.Image) -> np.ndarray:
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt")
            image_features = model_clip.get_image_features(**inputs)
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu().numpy()

    class NearestNeighborSearch:
        def __init__(self, vectors_dir, audio_dir=None, normalize=True):
            self.vectors_dir = vectors_dir
            self.audio_dir = audio_dir
            self.normalize = normalize
            self.index = []
            self.metadata = []
            self._load_data()

        def _load_data(self):
            files = sorted(
                f for f in os.listdir(self.vectors_dir)
                if f.endswith(".npy")
            )

            vectors = []
            metadata = []

            for f in files:
                vec = np.load(os.path.join(self.vectors_dir, f)).astype(np.float32)

                if vec.shape != (13,):
                    continue

                if self.normalize:
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec /= norm

                audio_file = f.replace(".npy", ".mp3")
                if self.audio_dir:
                    audio_file = os.path.join(self.audio_dir, audio_file)

                vectors.append(vec)
                metadata.append({
                    "audio_file": audio_file,
                    "emotion_file": f
                })

            self.index = np.array(vectors)
            self.metadata = metadata

        def find_nearest(self, query_vector, k=10, use_cosine=True, max_candidates=200):
            query_vector = np.array(query_vector, dtype=np.float32)

            if self.normalize:
                norm = np.linalg.norm(query_vector)
                if norm > 0:
                    query_vector /= norm

            if len(self.index) == 0:
                return []

            if use_cosine:
                sims = np.dot(self.index, query_vector) / (
                    np.linalg.norm(self.index, axis=1)
                    * np.linalg.norm(query_vector)
                    + 1e-8
                )
                nearest_indices = np.argsort(-sims)[:max_candidates]
                distances = 1 - sims[nearest_indices]
            else:
                dists = np.linalg.norm(self.index - query_vector, axis=1)
                nearest_indices = np.argsort(dists)[:max_candidates]
                distances = dists[nearest_indices]

            results = []

            for i, idx in enumerate(nearest_indices):
                audio_file = self.metadata[idx]["audio_file"]

                if os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
                    results.append({
                        "distance": float(distances[i]),
                        "audio_file": audio_file,
                        "emotion_file": self.metadata[idx]["emotion_file"],
                        "vector": self.index[idx].tolist()
                    })

                if len(results) == k:
                    break

            return results

    searcher = NearestNeighborSearch("emid_emotion_vectors", "emid_audio")

    total = len(searcher.metadata)
    valid_audio = sum(
        os.path.exists(m["audio_file"]) and os.path.getsize(m["audio_file"]) > 0
        for m in searcher.metadata
    )

    print("\n----------------------")
    print("Audio dataset summary")
    print("----------------------")
    print("Total emotion vectors:", total)
    print("Valid audio files:", valid_audio)
    print("Invalid or empty files:", total - valid_audio)

    if total > 0:
        print(f"Percentage valid: {valid_audio / total * 100:.2f}%")
    else:
        print("Percentage valid: 0.00%")

    print("----------------------\n")

    def sample_frames(video_path, interval_sec=5, tmp_dir=None):
        os.makedirs(tmp_dir, exist_ok=True)

        ffmpeg.input(video_path).filter(
            "fps",
            fps=1 / interval_sec
        ).output(
            os.path.join(tmp_dir, "frame_%04d.jpg")
        ).run(overwrite_output=True)

        return sorted(
            os.path.join(tmp_dir, f)
            for f in os.listdir(tmp_dir)
            if f.endswith(".jpg")
        )

    def generate_audio_sequence(frames):
        audio_clips = []

        for frame_path in frames:
            img = Image.open(frame_path).convert("RGB")
            embedding = generate_clip_embedding(img)

            x = torch.tensor(embedding).float().reshape(1, 1, 512)

            with torch.no_grad():
                pred = model(x).cpu().numpy()[0]

            print(f"Frame: {frame_path}")
            print("Predicted vector for nearest neighbor:")
            print(pred)

            results = searcher.find_nearest(pred, k=10)

            if not results:
                print("No valid nearest audio found for this frame.")
                continue

            chosen = np.random.choice(results)
            audio_clips.append(chosen["audio_file"])

            print("chosen", chosen)

        return audio_clips

    def overlay_audio_on_video(video_path, audio_files, output_path, tmp_dir):
        if not audio_files:
            raise ValueError("No valid audio files to overlay")

        temp_wavs = []

        for i, mp3_file in enumerate(audio_files):
            wav_file = os.path.join(tmp_dir, f"tmp_{i}.wav")

            ffmpeg.input(mp3_file).output(
                wav_file,
                ac=2,
                ar=44100
            ).run(overwrite_output=True)

            temp_wavs.append(wav_file)

        concat_txt = os.path.join(tmp_dir, "concat_wavs.txt")

        with open(concat_txt, "w") as f:
            for wav_file in temp_wavs:
                f.write(f"file '{wav_file}'\n")

        combined_audio = os.path.join(tmp_dir, "combined_audio.m4a")

        ffmpeg.input(
            concat_txt,
            format="concat",
            safe=0
        ).output(
            combined_audio,
            acodec="aac"
        ).run(overwrite_output=True)

        video_input = ffmpeg.input(video_path)
        audio_input = ffmpeg.input(combined_audio)

        ffmpeg.output(
            video_input.video,
            audio_input.audio,
            output_path,
            vcodec="libx264",
            acodec="aac",
            shortest=None,
            map_metadata="-1"
        ).run(overwrite_output=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        frames_dir = os.path.join(tmp_dir, "frames")
        frames = sample_frames(video_path, interval_sec=5, tmp_dir=frames_dir)
        audio_files = generate_audio_sequence(frames)
        overlay_audio_on_video(video_path, audio_files, output_file, tmp_dir)

    print(f"Done! Output saved as {output_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python process_video.py <video_path> <output_path>")
        sys.exit(1)

    video_file = sys.argv[1]
    output_file = sys.argv[2]

    main(video_file, output_file)
