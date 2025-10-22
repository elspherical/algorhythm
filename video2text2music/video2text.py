"""
Video to Music Recommendations using Optimized BLIP2
Generates detailed musical recommendations for video frames efficiently.

To run this script, you will need to install:
pip install opencv-python transformers torch Pillow matplotlib
(For 8-bit quantization on CUDA, you also need 'accelerate' and 'bitsandbytes')
"""

import cv2
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple
import argparse
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
import string

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')



class VideoToMusicConverter:
    """Generate detailed music recommendations for video frames using BLIP2"""
    
    def __init__(self, model_name: str = "Salesforce/blip2-flan-t5-xl"):
        # Note: The default is changed to 'base' for maximum lightweightness.
        self.model_name = model_name
        # Check for CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"Loading BLIP2 model: {model_name}")
        
        self.processor = None
        self.model = None
        self._load_blip2_model()
    
    def _load_blip2_model(self):
        """Load the BLIP2 model, applying 8-bit quantization if CUDA is available for efficiency."""
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            
            print("Loading BLIP2 processor...")
            self.processor = Blip2Processor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            print("✓ BLIP2 processor loaded successfully")
            
            
            # --- Optimized Loading Logic for Lightweightness ---
            loading_kwargs = {}
            if self.device == "cuda":
                # Use 8-bit quantization for significant VRAM and speed improvements on GPU
                loading_kwargs = {
                    "load_in_8bit": True,
                    "device_map": "auto",
                }
                print("Optimizing load for CUDA: Using 8-bit quantization for efficiency.")
                
            else:
                # Fallback for CPU with standard float32 loading
                loading_kwargs = {
                    "torch_dtype": torch.float32,
                    "device_map": None,
                }
                print("Running on CPU. No 8-bit quantization applied.")

            print("Loading BLIP2 model (this may still take a moment, especially without 8-bit)...")

            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                **loading_kwargs
            )
            
            # If not using device_map='auto', manually move model to device
            if 'device_map' not in loading_kwargs:
                self.model = self.model.to(self.device)
            
            print("✓ BLIP2 model loaded successfully!")
            
            # Inform user about the model status
            if self.device == "cuda" and 'load_in_8bit' in loading_kwargs:
                print(f"Model loaded efficiently on GPU: {self.device}")
            else:
                print(f"Model loaded on device: {self.device}")
            
        except Exception as e:
            print(f"Error loading BLIP2 model: {e}")
            print("\nHint: If using CUDA, ensure 'accelerate' and 'bitsandbytes' are installed.")
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
        
        # Get frame rate (FPS) for logging
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video FPS: {fps:.2f}. Extracting 1 frame every {frame_interval} frames ({frame_interval/fps:.2f} seconds).")

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
        return frames
    
    def generate_music_caption(self, image: np.ndarray) -> str:
        """Generate detailed music recommendation for a single frame"""
        pil_image = Image.fromarray(image)
        
        try:
            # Enhanced prompt structure for better results from the smaller model
            detailed_music_prompt = (
                "Create a detailed musical recommendation for this visual scene. "
                "Describe the perfect background track by specifying:\n"
                "1. **Genre and Subgenre** (e.g., Synthwave, Baroque Classical)\n"
                "2. **Instrumentation** (e.g., Analog synthesizers, acoustic guitar, string quartet)\n"
                "3. **Tempo and Rhythm** (e.g., Fast, driving beat; slow, gentle waltz)\n"
                "4. **Mood and Emotional Tone** (e.g., Tense and cinematic, peaceful and reflective)\n"
                "5. **Production Style** (e.g., Electronic, Orchestral, Lo-fi)\n"
                "The output should be a single, rich, descriptive sentence."
            )
            
            music_inputs = self.processor(images=pil_image, text=detailed_music_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                music_ids = self.model.generate(
                    **music_inputs,
                    max_new_tokens=120,
                    num_beams=4,  # Slightly lower beam search for speed on lightweight models
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
            
            # Simple cleanup of common unwanted phrases returned by the multimodal model
            unwanted_phrases = [
                detailed_music_prompt,
                "A detailed musical recommendation for this visual scene.",
                "stock videos & royalty-free footage",
                "royalty-free footage",
                "stock footage",
                "stock videos",
                "stock photos",
                "Write in the style of:",
            ]
            
            for phrase in unwanted_phrases:
                music_caption = music_caption.replace(phrase, "").strip()

            # Fallback/Refinement logic (kept from original for robust results)
            if len(music_caption.split()) < 10 or any(generic in music_caption.lower() for generic in ["stock", "footage", "generic", "background"]):
                music_caption = "Upbeat, driving electronic music with heavy synthesizers, gated drums, and a high-energy, cinematic mood, perfect for an urban setting."
            
            return music_caption.strip()
            
        except Exception as e:
            print(f"Error generating music caption: {e}")
            return f"Error generating music caption: {e}"
        
    def get_detail_score(self,prompt: str) -> float:
        """
        Calculates a 'detail score' for a given prompt by assigning weights to
        different parts of speech (POS tags).

        The score is calculated as the sum of weighted scores for all tokens.
        This rewards prompts that are both long and rich in descriptive language.

        The weights prioritize descriptive elements:
        - Adjectives (JJ, JJR, JJS)
        - Proper Nouns (NNP, NNPS)
        - Verbs (VB, VBD, VBG, VBN, VBP, VBZ)
        - Foreign Words (FW) - often technical terms
        """

        # --- Part-of-Speech Tagging Weight Map ---
        # Higher weights mean the word type contributes more to "detail."
        # Based on the Penn Treebank Tagset (used by NLTK's default tagger).
        pos_weights = defaultdict(lambda: 1.0, {
            # Adjectives (Descriptive words)
            'JJ': 3.0,  # Adjective, e.g., 'big', 'dark'
            'JJR': 3.5, # Adjective, comparative, e.g., 'bigger'
            'JJS': 4.0, # Adjective, superlative, e.g., 'biggest'

            # Proper Nouns (Specific entities/names)
            'NNP': 2.5, # Proper noun, singular, e.g., 'Gemini', 'Paris'
            'NNPS': 2.5, # Proper noun, plural, e.g., 'Americans'

            # Foreign words (Often technical or specific terms)
            'FW': 2.0,  # Foreign word, e.g., 'status quo', 'ad hoc'

            # Nouns (Objects/Concepts)
            'NN': 1.5,  # Noun, singular or mass, e.g., 'table', 'water'
            'NNS': 1.5, # Noun, plural, e.g., 'tables'

            # Verbs (Actions)
            'VB': 1.0,  # Base form, e.g., 'take'
            'VBD': 1.0, # Past tense, e.g., 'took'
            'VBG': 1.0, # Gerund/present participle, e.g., 'taking'
        })

        # 1. Tokenize the prompt
        # Remove punctuation before tokenizing for cleaner word analysis
        cleaned_prompt = prompt.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(cleaned_prompt)

        if not tokens:
            return 0.0

        # 2. Part-of-Speech Tagging
        # If the NLTK data is missing, this is where the LookupError occurs.
        tagged_words = nltk.pos_tag(tokens)

        # 3. Calculate Final Weighted Score (Sum of all token weights)
        total_score = 0.0
        for word, tag in tagged_words:
            # Get weight based on POS tag, defaulting to 1.0 for unlisted tags
            weight = pos_weights[tag]
            total_score += weight

        # We now return the raw total score, rewarding longer, more descriptive prompts.
        return total_score

    
    def process_video(self, video_path: str, frame_interval: int = 30) -> List[Tuple[np.ndarray, int, str]]:
        """Process video and generate music recommendations for each frame"""
        print(f"Starting video processing...")
        print(f"Video: {video_path}")
        print(f"Frame sampling: every {frame_interval} frames")
        print(f"Model: {self.model_name}")
        
        frames = self.extract_frames(video_path, frame_interval)
        
        print(f"Generating music recommendations for frames...")
        results = []
        transition = " Add a 2–3 second smooth transition to/from this music in a soft-fade form."
        
        for i, (frame, frame_num) in enumerate(frames):
            print(f"Processing frame {i+1}/{len(frames)} (video frame {frame_num})")
            
            caption = self.generate_music_caption(frame)
            results.append((frame, frame_num, caption+transition))
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
        
        # Only create subplots if there are results
        if n_frames == 0:
            print("No frames processed to display.")
            return

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        
        # Flatten axes array for easy iteration, handling cases where rows/cols is 1
        if rows == 1 and cols > 1:
            axes = axes.reshape(1, -1)
        elif cols == 1 and rows > 1:
            axes = axes.reshape(-1, 1)
        elif rows == 1 and cols == 1:
            axes = np.array([[axes]])

        
        for i, (frame, frame_num, caption) in enumerate(display_results):
            row = i // cols
            col = i % cols
            
            axes[row, col].imshow(frame)
            axes[row, col].set_title(f"Frame {frame_num}\n{caption}", fontsize=10)
            axes[row, col].axis('off')
        
        # Turn off unused subplots
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

    def find_most_detailed_prompt(self,results):
        """
        Analyzes an array of prompts to find the one with the highest detail score.

        Args:
            prompts: A list of string prompts to analyze.

        Returns:
            A tuple containing the most detailed prompt (string) and its score (float).
        """
        if not results:
            return ("", 0.0)
        

        max_score = -1.0
        most_detailed_prompt = ""

        print("--- Prompt Analysis ---")
        for frame_data in results:
            score = self.get_detail_score(frame_data[2])
            print(f"Prompt: '{frame_data[2]}'")
            print(f"Detail Score: {score:.2f}")

            if score > max_score:
                max_score = score
                most_detailed_prompt = frame_data[2]

        print("-----------------------")
        return most_detailed_prompt, max_score


    


def main():
    parser = argparse.ArgumentParser(description="Generate music recommendations for video using BLIP2")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--frame_interval", type=int, default=30, 
                       help="Extract every Nth frame (default: 30)")
    parser.add_argument("--max_display", type=int, default=12,
                       help="Maximum frames to display (default: 12)")
    parser.add_argument("--output_dir", default="./blip2_output",
                       help="Output directory for results (default: ./blip2_output)")
    # Default changed to the lightweight 'base' model
    parser.add_argument("--model", default="Salesforce/blip2-flan-t5-base",
                       help="BLIP2 model name (e.g., Salesforce/blip2-flan-t5-base for lightweight, or blip2-flan-t5-xl). Default: Salesforce/blip2-flan-t5-base")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize converter with the selected model (default is now 'base')
    converter = VideoToMusicConverter(model_name=args.model)
    
    print("Starting video processing...")
    results = converter.process_video(args.video_path, args.frame_interval)

  
    
    if not results:
        print("No results were generated. Exiting.")
        return
        
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    captions_file = os.path.join(args.output_dir, f"{video_name}_music_captions.txt")
    converter.save_captions_to_file(results, captions_file)
    
    visualization_path = os.path.join(args.output_dir, f"{video_name}_music_visualization.png")
    converter.display_results(results, max_frames=args.max_display, 
                            save_path=visualization_path)
    
    top_prompt = converter.find_most_detailed_prompt(results)
    
    print(f"Processing complete! Results saved in: {args.output_dir}")

    return top_prompt


if __name__ == "__main__":
    main()
