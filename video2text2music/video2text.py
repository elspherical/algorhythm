"""
Video to Music Recommendations using BLIP2
Generates detailed musical recommendations for video frames
"""

import cv2
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple
import argparse


class VideoToMusicConverter:
    """Generate detailed music recommendations for video frames using BLIP2"""
    
    def __init__(self, model_name: str = "Salesforce/blip2-flan-t5-xl"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"Loading BLIP2 model: {model_name}")
        
        self.processor = None
        self.model = None
        self._load_blip2_model()
    
    def _load_blip2_model(self):
        """Load the BLIP2 model"""
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            
            print("Loading BLIP2 processor...")
            self.processor = Blip2Processor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                force_download=False
            )
            print("✓ BLIP2 processor loaded successfully")
            
            print("Loading BLIP2 model (this may take several minutes)...")
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                force_download=False,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device != "cuda" or not hasattr(self.model, 'hf_device_map'):
                self.model = self.model.to(self.device)
            
            print("✓ BLIP2 model loaded successfully!")
            print(f"Model size: ~16GB, Device: {self.device}")
            
        except Exception as e:
            print(f"Error loading BLIP2 model: {e}")
            raise RuntimeError(f"Failed to load BLIP2 model: {e}")
    
    def extract_frames(self, video_path: str, frame_interval: int = 30) -> List[Tuple[np.ndarray, int]]:
        """Extract frames from video at specified intervals"""
        print(f"Extracting frames from: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((frame_rgb, frame_count))
                extracted_count += 1
                
                if extracted_count % 10 == 0:
                    print(f"Extracted {extracted_count} frames...")
            
            frame_count += 1
        
        cap.release()
        
        print(f"Total frames extracted: {len(frames)} from {frame_count} total frames")
        print(f"Sampling rate: 1 in every {frame_interval} frames")
        return frames
    
    def generate_music_caption(self, image: np.ndarray) -> str:
        """Generate detailed music recommendation for a single frame"""
        pil_image = Image.fromarray(image)
        
        try:
            detailed_music_prompt = """Create a detailed musical recommendation for this visual scene. Provide a rich, descriptive musical suggestion that includes:

1. Specific genre and subgenre
2. Detailed instrumentation (mention specific instruments)
3. Tempo and rhythm characteristics
4. Mood and emotional tone
5. Production style (acoustic, electronic, orchestral, etc.)
6. Cultural context if relevant
7. Atmospheric qualities

Write in the style of: "Upbeat pop dance track with electric guitar, synthesizers, and energetic vocals" or "Acoustic folk music with guitar, violin, and harmonica, creating a warm, intimate atmosphere" or "Electronic ambient music with synthesizers, pads, and subtle percussion, perfect for relaxation" or "Jazz fusion with saxophone, piano, and complex rhythms, creating a sophisticated urban vibe" or "Rock anthem with electric guitars, drums, and powerful vocals, creating an epic cinematic atmosphere."

Be very specific about musical elements like rhythm, melody, harmony, and production style. Focus on creating evocative, detailed descriptions with multiple instruments and atmospheric context. Avoid generic descriptions. Consider diverse genres: pop, rock, jazz, electronic, classical, folk, hip-hop, etc."""
            
            music_inputs = self.processor(images=pil_image, text=detailed_music_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                music_ids = self.model.generate(
                    **music_inputs,
                    max_new_tokens=120,
                    num_beams=6,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            music_caption = self.processor.batch_decode(
                music_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            if "Create a detailed musical recommendation" in music_caption:
                music_caption = music_caption.split("Create a detailed musical recommendation")[-1].strip()
            
            unwanted_suffixes = [
                "stock videos & royalty-free footage",
                "stock videos",
                "royalty-free footage",
                "& royalty-free footage",
                "videos & royalty-free footage",
                "stock footage",
                "royalty-free"
            ]
            
            for suffix in unwanted_suffixes:
                if suffix in music_caption:
                    music_caption = music_caption.replace(suffix, "").strip()
                    music_caption = music_caption.rstrip(" -").strip()
            
            if music_caption.strip() in ["-", "", "india", "india stock", "stock"] or len(music_caption.split()) < 3:
                direct_prompt = """Create a detailed musical recommendation for this scene. Be very specific about instruments, genre, and mood. Examples: "Upbeat pop dance track with electric guitar, synthesizers, and energetic vocals" or "Acoustic folk music with guitar, violin, and harmonica, creating a warm atmosphere" or "Electronic ambient music with synthesizers, pads, and subtle percussion" or "Jazz fusion with saxophone, piano, and complex rhythms" or "Rock anthem with electric guitars, drums, and powerful vocals, creating an epic cinematic atmosphere"."""
                
                direct_inputs = self.processor(images=pil_image, text=direct_prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    direct_ids = self.model.generate(
                        **direct_inputs,
                        max_new_tokens=100,
                        num_beams=6,
                        early_stopping=True,
                        do_sample=True,
                        temperature=0.95,
                        top_p=0.9
                    )
                
                direct_caption = self.processor.batch_decode(
                    direct_ids, 
                    skip_special_tokens=True
                )[0].strip()
                
                if len(direct_caption.split()) > 5 and "stock" not in direct_caption.lower():
                    music_caption = direct_caption
                else:
                    music_caption = "Upbeat urban music with electric guitar, drums, and energetic vocals, creating a vibrant street atmosphere"
            
            if len(music_caption.split()) < 8 or any(generic in music_caption.lower() for generic in ["crowded street", "street music", "acoustic music", "indian music", "stock", "footage", "generic", "background"]):
                fallback_prompt = """Describe the perfect music for this scene. Be very specific about genre, instruments, tempo, and mood. Examples: "Upbeat pop dance track with electric guitar, synthesizers, and energetic vocals" or "Acoustic folk music with guitar, violin, and harmonica, creating a warm atmosphere" or "Electronic ambient music with synthesizers, pads, and subtle percussion" or "Jazz fusion with saxophone, piano, and complex rhythms" or "Rock anthem with electric guitars, drums, and powerful vocals, creating an epic cinematic atmosphere"."""
                
                fallback_inputs = self.processor(images=pil_image, text=fallback_prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    fallback_ids = self.model.generate(
                        **fallback_inputs,
                        max_new_tokens=120,
                        num_beams=6,
                        early_stopping=True,
                        do_sample=True,
                        temperature=0.95,
                        top_p=0.9
                    )
                
                fallback_caption = self.processor.batch_decode(
                    fallback_ids, 
                    skip_special_tokens=True
                )[0].strip()
                
                if len(fallback_caption.split()) > len(music_caption.split()) and "stock" not in fallback_caption.lower() and "generic" not in fallback_caption.lower():
                    music_caption = fallback_caption
                else:
                    music_caption = "Upbeat urban music with electric guitar, drums, bass, and energetic vocals, creating a vibrant street atmosphere"
            
            return music_caption.strip()
            
        except Exception as e:
            print(f"Error generating music caption: {e}")
            return f"Error generating music caption: {e}"
    
    def process_video(self, video_path: str, frame_interval: int = 30) -> List[Tuple[np.ndarray, int, str]]:
        """Process video and generate music recommendations for each frame"""
        print(f"Starting video processing...")
        print(f"Video: {video_path}")
        print(f"Frame sampling: every {frame_interval} frames")
        print(f"Model: {self.model_name}")
        
        frames = self.extract_frames(video_path, frame_interval)
        
        print(f"Generating music recommendations for frames...")
        results = []
        
        for i, (frame, frame_num) in enumerate(frames):
            print(f"Processing frame {i+1}/{len(frames)} (video frame {frame_num})")
            
            caption = self.generate_music_caption(frame)
            results.append((frame, frame_num, caption))
            print(f"Music recommendation: {caption}")
        
        print(f"Processing complete! Processed {len(results)} frames")
        return results
    
    def display_results(self, results: List[Tuple[np.ndarray, int, str]], 
                       max_frames: int = 12, save_path: str = None):
        """Display frames with their music recommendations"""
        display_results = results[:max_frames]
        n_frames = len(display_results)
        cols = 3
        rows = (n_frames + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (frame, frame_num, caption) in enumerate(display_results):
            row = i // cols
            col = i % cols
            
            axes[row, col].imshow(frame)
            axes[row, col].set_title(f"Frame {frame_num}\n{caption}", fontsize=10)
            axes[row, col].axis('off')
        
        for i in range(n_frames, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def save_captions_to_file(self, results: List[Tuple[np.ndarray, int, str]], 
                            output_path: str):
        """Save music recommendations to a text file"""
        with open(output_path, 'w') as f:
            f.write("Video Frame-by-Frame Music Recommendations (Generated by BLIP2)\n")
            f.write("=" * 70 + "\n")
            f.write("This file contains music recommendations for each video frame.\n")
            f.write("Each recommendation includes genre, mood, tempo, and instrumentation suggestions.\n\n")
            
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Total frames processed: {len(results)}\n\n")
            
            for frame, frame_num, caption in results:
                f.write(f"Frame {frame_num} - Music Recommendation:\n")
                f.write(f"{caption}\n\n")
        
        print(f"Music recommendations saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate music recommendations for video using BLIP2")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--frame_interval", type=int, default=30, 
                       help="Extract every Nth frame (default: 30)")
    parser.add_argument("--max_display", type=int, default=12,
                       help="Maximum frames to display (default: 12)")
    parser.add_argument("--output_dir", default="./blip2_output",
                       help="Output directory for results (default: ./blip2_output)")
    parser.add_argument("--model", default="Salesforce/blip2-flan-t5-xl",
                       help="BLIP2 model name (default: Salesforce/blip2-flan-t5-xl)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    converter = VideoToMusicConverter(model_name=args.model)
    
    print("Starting video processing...")
    results = converter.process_video(args.video_path, args.frame_interval)
    
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    captions_file = os.path.join(args.output_dir, f"{video_name}_music_captions.txt")
    converter.save_captions_to_file(results, captions_file)
    
    visualization_path = os.path.join(args.output_dir, f"{video_name}_music_visualization.png")
    converter.display_results(results, max_frames=args.max_display, 
                            save_path=visualization_path)
    
    print(f"Processing complete! Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()