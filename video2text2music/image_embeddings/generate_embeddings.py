import os
import csv
import requests
import logging
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from tqdm import tqdm

# Configuration
CSV_URL = "https://huggingface.co/datasets/igorriti/ambience-audio/resolve/main/train.csv"
OUTPUT_DIR = "embeddings"
MAPPING_CSV = "dataset.csv"
CLIPS_PER_VIDEO = 20
MAX_WORKERS = 8  # Increased workers since we are just downloading images

# Logging setup
LOG_FILE = "embedding_generation_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Model Globally
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading CLIP model on {device}...")
try:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise e

def load_csv():
    logger.info("Downloading CSV...")
    try:
        r = requests.get(CSV_URL)
        r.raise_for_status()
        rows = []
        reader = csv.DictReader(r.text.splitlines())
        for row in reader:
            rows.append(row)
        logger.info(f"Loaded {len(rows)} rows from CSV")
        return rows
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return []

def generate_embedding(image):
    try:
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy()
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None

def process_video(row, index):
    video_id = row["id"]
    thumbnail_url = row.get("thumbnailUrl")
    
    if not thumbnail_url:
        logger.warning(f"[{video_id}] No thumbnail URL found, skipping.")
        return None

    out_name = f"{index}.npy"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    
    # Check if embedding already exists
    if os.path.exists(out_path):
        return index

    try:
        # Download image
        response = requests.get(thumbnail_url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        
        # Generate embedding
        emb = generate_embedding(image)
        if emb is not None:
            np.save(out_path, emb)
            return index
    except Exception as e:
        logger.error(f"[{video_id}] Failed to process thumbnail: {e}")
        return None

def main():
    rows = load_csv()
    if not rows:
        return

    processed_indices = []
    
    logger.info("Starting batch processing...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_video, row, idx): idx for idx, row in enumerate(rows, start=1)}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx = future.result()
            if idx is not None:
                processed_indices.append(idx)

    logger.info(f"Processed {len(processed_indices)} videos.")
    
    # Generate Mapping CSV
    logger.info(f"Generating {MAPPING_CSV}...")
    try:
        with open(MAPPING_CSV, 'w', newline='') as csvfile:
            fieldnames = ['audio_file', 'embedding_file']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for idx in processed_indices:
                embedding_file = f"{idx}.npy"
                # Each video corresponds to 20 audio clips
                for clip_num in range(1, CLIPS_PER_VIDEO + 1):
                    audio_file = f"{idx}_{clip_num}.mp3"
                    writer.writerow({'audio_file': audio_file, 'embedding_file': embedding_file})
                    
        logger.info("Mapping CSV generated successfully.")
        
    except Exception as e:
        logger.error(f"Failed to generate mapping CSV: {e}")

if __name__ == "__main__":
    main()
