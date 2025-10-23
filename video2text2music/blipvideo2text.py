"""
Video to Music Recommendations using Optimized BLIP (V1)
Generates detailed musical recommendations for video frames efficiently.

To run this script, you will need to install:
pip install opencv-python transformers torch Pillow matplotlib nltk sentence-transformers
(BLIP V1 is simpler and does not require 'accelerate' or 'bitsandbytes')
"""

import cv2
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import textwrap
from typing import List, Tuple
import argparse
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
import string
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# --- CHANGE 1: Rename Class for Clarity ---
class VideoToMusicConverter:
    """Generate detailed music recommendations for video frames using BLIP"""
    
    # --- CHANGE 2: Use BLIP V1 Model ID as Default ---
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"Loading BLIP model: {model_name}")
        
        self.processor = None
        self.model = None
        # --- Change: Call the new BLIP loading method ---
        self._load_blip_model()

        # --- NEW: Load the Sentence Transformer Model ---
        # 'all-MiniLM-L6-v2' is small, fast, and very effective
        self.st_model_name = 'all-MiniLM-L6-v2'
        print(f"Loading Sentence Transformer model: {self.st_model_name}")
        try:
            self.st_model = SentenceTransformer(self.st_model_name, device=self.device)
            print("✓ Sentence Transformer model loaded successfully!")
        except Exception as e:
            print(f"Could not load Sentence Transformer model: {e}")
            raise RuntimeError(f"Failed to load {self.st_model_name}: {e}")
        
        self.music_profiles = [
            # --- 1. ACTION, TENSION & HORROR ---
            {
                "id": "high_energy_electronic",
                "keywords": "fast action, driving, running, city, car chase, sports, energetic, vibrant, neon, upbeat, party, futuristic, modern, club",
                "template": "A high-energy, driving electronic track with a fast tempo ({tempo} BPM) and layered synth melodies. Perfect for a dynamic scene of {caption}."
            },
            {
                "id": "aggressive_rock_metal",
                "keywords": "intense, action, fight, battle, chase, aggressive, loud, fast, distorted guitar, heavy metal, hard rock, explosion, power, anger",
                "template": "An aggressive, high-octane hard rock track. Features {instrument}, a driving drum beat, and a powerful, distorted bassline for this intense scene of {caption}."
            },
            {
                "id":"tense_suspense_thriller",
                "keywords": "dark, suspense, thriller, tense, nervous, alone, night, eerie, danger, heartbeat, approaching, minimalist, drone, investigation, stealth",
                "template": "A tense, minimalist suspense score. Uses low-droning synths and {instrument} to build a sense of unease and anticipation for this scene of {caption}."
            },
            {
                "id": "eerie_horror_ambient",
                "keywords": "eerie, creepy, horror, dark, scary, alone, footsteps, shadow, ghost, unsettling, atmospheric, dissonant, jump scare, haunted",
                "template": "An unsettling, eerie ambient soundscape. Uses dissonant {instrument} and a cold, low drone to create a growing sense of dread for {caption}."
            },
            {
                "id": "chaotic_industrial",
                "keywords": "chaotic, intense, fast, disorienting, confusing, panic, industrial, noise, running, abstract, experimental, machinery, glitch",
                "template": "A chaotic, fast-paced industrial track. Uses {instrument}, glitchy percussion, and a disorienting structure to match the frenetic energy of {caption}."
            },
            {
                "id": "cyberpunk_techno",
                "keywords": "cyberpunk, techno, futuristic, dark city, neon, industrial, robotic, driving beat, repetitive, edgy, dystopian, underground",
                "template": "A driving, repetitive techno beat with a dark, futuristic synth bassline. Perfectly captures the edgy, cyberpunk aesthetic of {caption}."
            },

            # --- 2. CINEMATIC, ORCHESTRAL & EMOTIONAL ---
            {
                "id": "epic_adventure_orchestral",
                "keywords": "epic, adventure, journey, soaring, heroic, vast landscape, cinematic, dramatic, triumphant, powerful, discovery, fantasy, battle",
                "template": "A soaring, heroic orchestral score. Builds with powerful {instrument} and a triumphant choir, perfect for an epic, adventurous shot of {caption}."
            },
            {
                "id": "cinematic_ambient_nature",
                "keywords": "beautiful landscape, ocean, mountains, sunset, stars, nature, slow, peaceful, serene, wonder, vast, atmospheric, space, drone",
                "template": "A beautiful, sweeping cinematic ambient piece. Features {instrument} to evoke a sense of peace and awe for this breathtaking shot of {caption}."
            },
            {
                "id": "sad_melancholy_piano",
                "keywords": "sad, melancholy, crying, loss, grief, rain, lonely, slow, pensive, emotional, somber, longing, reflective, funeral",
                "template": "A slow, somber, and emotional piece. Features a {instrument} melody over a bed of soft strings to highlight the feeling of {caption}."
            },
            {
                "id": "hopeful_inspirational",
                "keywords": "hopeful, rising sun, new beginning, optimistic, uplifting, soaring strings, piano arpeggios, inspirational, overcoming, dawn",
                "template": "An uplifting and inspirational track. Features {instrument} and a gentle crescendo of strings, building a sense of hope and optimism for {caption}."
            },
            {
                "id": "romantic_warm_ballad",
                "keywords": "romantic, love, couple, holding hands, smiling, warm, gentle, heartfelt, soft, relationship, kiss, tender, sweet, wedding",
                "template": "A gentle and warm {instrument} ballad with a {mood} melody. Creates a heartfelt, romantic atmosphere for this tender moment of {caption}."
            },
            {
                "id": "magical_fantasy_wonder",
                "keywords": "magic, wonder, awe, fantasy, stars, glowing, enchanted, mystery, discovery, fairytale, dreamlike, sparkling, celesta",
                "template": "A wondrous and magical orchestral piece. Features twinkling {instrument} like a glockenspiel and soaring strings to capture the enchanting feeling of {caption}."
            },
            {
                "id": "introspective_minimalist",
                "keywords": "introspective, thoughtful, pensive, alone, thinking, minimalist piano, sparse, delicate, reflective, memories, solitude",
                "template": "A sparse, minimalist piano piece with a {mood}, reflective tone. Ample empty space in the music allows for focus on the scene of {caption}."
            },
            {
                "id": "baroque_classical_elegant",
                "keywords": "baroque, classical, harpsichord, formal, elegant, refined, sophisticated, old painting, museum, mansion, ballroom, 1700s, Bach",
                "template": "An elegant, formal Baroque piece featuring a {instrument}. Perfect for a sophisticated, historical, or highly refined scene like {caption}."
            },

            # --- 3. VIBES, GENRES & WORLD ---
            {
                "id": "calm_acoustic_folk",
                "keywords": "calm, relaxing, coffee shop, gentle, soft, indoor, person talking, interview, morning, slow, peaceful, acoustic guitar, folk, singer-songwriter",
                "template": "A soft, gentle acoustic folk piece with a {instrument}. A {mood} and minimalist background, ideal for this calm moment of {caption}."
            },
            {
                "id": "lo_fi_hip_hop_chill",
                "keywords": "lo-fi, studying, rain, window, chill, relaxing, vinyl crackle, boom bap, mellow, introspective, cozy, headphones, anime",
                "template": "A mellow, chill lo-fi hip-hop beat. Features a {mood} piano loop, vinyl crackle, and a simple {instrument} rhythm, great for a relaxing or introspective scene like {caption}."
            },
            {
                "id": "funky_upbeat_pop",
                "keywords": "fun, happy, dancing, friends, comedy, lighthearted, cheerful, bright, walking, groove, upbeat pop, stylish, retro, 1980s",
                "template": "An upbeat, funky pop groove with a clear bassline and {instrument}. Creates a fun, lighthearted, and cheerful mood for {caption}."
            },
            {
                "id": "pure_funk_groove",
                "keywords": "funk, groove, bassline, horns, 1970s, James Brown, dancing, tight, rhythmic, wah-wah guitar, soulful, energetic",
                "template": "A tight, rhythmic funk track driven by a {instrument} and a powerful horn section. Perfect for a high-energy, confident, or 'cool' scene like {caption}."
            },
            {
                "id": "urban_hip_hop_beat",
                "keywords": "hip-hop, urban, city street, graffiti, cool, confident, swagger, stylish, modern, rhythmic, rap, beat, trap, 90s",
                "template": "A confident, {mood} hip-hop beat with a strong rhythmic groove and a {instrument} sample. Fits the cool, urban energy of {caption}."
            },
            {
                "id": "modern_country",
                "keywords": "country, pickup truck, small town, heartfelt, guitar twang, storytelling, sincere, driving, highway, America, boots, acoustic",
                "template": "An earnest, modern country track. Led by an {instrument} and a clear, storytelling vocal style, fitting for a heartfelt or rustic scene of {caption}."
            },
            {
                "id": "southern_blues_rock",
                "keywords": "blues, southern rock, gritty, soulful, electric guitar solo, harmonica, hardship, authentic, raw, bar, smoky, swamp",
                "template": "A gritty, soulful blues-rock track. Defined by a raw {instrument} solo and a driving, emotional rhythm, perfect for a scene with hardship or raw authenticity like {caption}."
            },
            {
                "id": "vintage_jazz_soul",
                "keywords": "nostalgic, vintage, old photo, sepia, retro, old-fashioned, 1950s, 1960s, jazz, soul, classic, memories, speakeasy, blues, saxophone",
                "template": "A nostalgic, warm {instrument} track with a classic, vintage feel. Perfect for a retro-style scene or a moment of reminiscence like {caption}."
            },
            {
                "id": "latin_salsa_reggaeton",
                "keywords": "latin, salsa, reggaeton, dancing, party, fun, energetic, rhythmic, trumpets, congas, vibrant, festive, tropical, mambo",
                "template": "An energetic, rhythmic Latin track (salsa or reggaeton). Features {instrument}, a strong percussion backbone, and a festive, danceable vibe for {caption}."
            },
            {
                "id": "beach_reggae_chill",
                "keywords": "reggae, beach, sunny, relaxing, chill, island, vacation, laid-back, off-beat guitar, good vibes, tropical, summer, ocean",
                "template": "A laid-back, sunny reggae track. Built on an {instrument} and a relaxed, off-beat rhythm, perfect for a chill, tropical, or beach scene like {caption}."
            },
            {
                "id": "traditional_east_asian",
                "keywords": "east asian, traditional, guzheng, koto, flute, serene, temple, meditation, calm, nature, bamboo, china, japan, anime",
                "template": "A serene, traditional East Asian piece. Features the {instrument} to create a peaceful, meditative, and focused atmosphere for {caption}."
            },
            {
                "id": "bollywood_indian_energetic",
                "keywords": "bollywood, indian, vibrant, dancing, sitar, tabla, energetic, colorful, celebration, dramatic, film, song and dance",
                "template": "A vibrant, high-energy Bollywood track. Combines traditional {instrument} with modern pop production for a dramatic, colorful, and celebratory scene like {caption}."
            },
            {
                "id": "gospel_soul_uplifting",
                "keywords": "gospel, spiritual, uplifting, choir, powerful vocals, organ, hopeful, joyful, soulful, celebratory, church, praise",
                "template": "A powerful, uplifting gospel track. Driven by a {instrument}, a full choir, and soulful lead vocals, creating a joyful and celebratory mood for {caption}."
            },
            {
                "id": "quirky_comedy_pizzicato",
                "keywords": "quirky, comedy, funny, weird, awkward, clumsy, lighthearted, playful, silly, bouncing, pizzicato strings, ukulele, whimsical",
                "template": "A lighthearted, quirky, and playful tune. Uses {instrument} to create a bouncing, comedic rhythm for this {mood} scene of {caption}."
            },
            {
                "id": "peaceful_meditation_ambient",
                "keywords": "peaceful, serene, nature, forest, river, birds, meditation, calm, relaxing, gentle, quiet, zen, yoga, spa, ambient, pads",
                "template": "A serene and peaceful new-age track. Gentle {instrument} and soft pads create a relaxing, meditative atmosphere for this scene of {caption}."
            },
            {
                "id": "video_game_chiptune",
                "keywords": "video game, chiptune, 8-bit, 16-bit, retro, arcade, nostalgic, simple melody, fast, electronic, fun, pixel, 1990s",
                "template": "A fast, nostalgic 8-bit chiptune track. Uses simple {instrument} waves to create a fun, retro, video-game feel for {caption}."
            },
            {
                "id": "documentary_neutral_background",
                "keywords": "documentary, interview, talking, neutral, informative, background, subtle, minimalist, focus, corporate, serious, explainer",
                "template": "A neutral, subtle, and informative minimalist track. Uses a simple {instrument} to provide background without being distracting, suitable for {caption}."
            }
        ]

        print("Computing music profile embeddings...")
        profile_keywords = [p['keywords'] for p in self.music_profiles]
        self.profile_embeddings = self.st_model.encode(profile_keywords, convert_to_tensor=True, device=self.device)
        print(f"✓ {len(self.music_profiles)} music profiles embedded.")
        

    
    def _get_semantic_recommendation(self, factual_caption: str) -> str:
        """
        Finds the best music profile by comparing semantic embeddings.
        """
        # print(factual_caption) # <-- This is too verbose for every frame
        # 1. Encode the factual caption from BLIP
        caption_embedding = self.st_model.encode(factual_caption, convert_to_tensor=True, device=self.device)
        
        # 2. Compute Cosine Similarity
        # This compares the caption's vector to all profile vectors at once
        cosine_scores = util.cos_sim(caption_embedding, self.profile_embeddings)[0]
        
        # 3. Find the best match
        best_match_index = torch.argmax(cosine_scores).item()
        best_profile = self.music_profiles[best_match_index]
        best_score = cosine_scores[best_match_index].item()

        # print(f"  [Semantic Match: '{best_profile['id']}' (Score: {best_score:.2f})]") # <-- Too verbose
        
        # 4. Populate the template
        # This is where we can add simple dynamic elements back in
        
        # Simple dynamic placeholders
        instrument = "soaring strings"
        if "electronic" in best_profile['id']:
            instrument = "bright synths"
        elif "acoustic" in best_profile['id']:
            instrument = "a gentle piano"
        elif "tense" in best_profile['id']:
            instrument = "dissonant strings"
        
        tempo = "120-130"
        if "cinematic" in best_profile['id'] or "tense" in best_profile['id']:
            tempo = "60-80"
        
        mood = "thoughtful"
        if "funky" in best_profile['id']:
            mood = "playful"
            
        # 5. Fill the template
        music_caption = best_profile['template'].format(
            caption=factual_caption,
            instrument=instrument,
            tempo=tempo,
            mood=mood
        )
        
        return music_caption
    # --- CHANGE 3: Rewrite Model Loading for BLIP V1 ---
    def _load_blip_model(self):
        """Load the BLIP model (V1)"""
        try:
            # Note: We now import BlipProcessor and BlipForConditionalGeneration (without the '2')
            from transformers import BlipProcessor, BlipForConditionalGeneration 
            
            print("Loading BLIP processor...")
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            print("✓ BLIP processor loaded successfully")
            
            print("Loading BLIP model...")
            # BLIP V1 is simpler and can be loaded directly to the device
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            
            print("✓ BLIP model loaded successfully!")
            
        except Exception as e:
            # Original error handling adjusted for BLIP V1 simplicity
            print(f"Error loading BLIP model: {e}")
            raise RuntimeError(f"Failed to load BLIP model: {e}")
    
    # --- METHOD REMOVED ---
    # def extract_frames(self, video_path: str, frame_interval: int = 30) -> List[Tuple[np.ndarray, int]]:
    # This logic is now merged into process_video
    
    def generate_music_caption(self, image: np.ndarray) -> str:
        """
        Generates a music recommendation by first creating a factual caption
        with BLIP-1 and then semantically matching it to a music profile.
        """
        pil_image = Image.fromarray(image)
        
        try:
            # --- Step 1: Generate a simple, factual caption (Unchanged) ---
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                caption_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=5,
                    early_stopping=True,
                    repetition_penalty=1.2
                )
            
            factual_caption = self.processor.batch_decode(
                caption_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            prefixes_to_remove = ["a photo of ", "a picture of ", "an image of ", "a screenshot of ", "a close up of "]
            factual_caption_lower = factual_caption.lower()
            
            for prefix in prefixes_to_remove:
                if factual_caption_lower.startswith(prefix):
                    factual_caption = factual_caption[len(prefix):].strip()
                    break

            # --- Step 2: Get Semantic Recommendation ---
            # This REPLACES the 'if/elif/else' block
            
            if len(factual_caption.split()) < 3:
                # Fallback if BLIP-1 fails and gives a useless caption
                music_caption = "Upbeat, driving electronic music with a high-energy, cinematic mood."
            else:
                music_caption = self._get_semantic_recommendation(factual_caption)

            return music_caption.strip()
            
        except Exception as e:
            print(f"Error generating music caption: {e}")
            return f"Error generating music caption: {e}"   
             
    def get_detail_score(self,prompt: str) -> float:
        """
        Calculates a 'detail score' for a given prompt by assigning weights to
        different parts of speech (POS tags).
        [REMAINDER OF FUNCTION UNCHANGED]
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

    
    # --- MAJOR CHANGE: process_video now handles frame extraction and chunked analysis ---
    def process_video(self, video_path: str, selector_interval_seconds: int = 5) -> Tuple[List[Tuple[np.ndarray, int, str]], List[Tuple[int, str, float]]]:
        """
        Process video frame-by-frame, generate music recommendations, and
        run the 'best prompt selector' every N seconds.
        
        Returns a tuple:
        1. all_results: List of (frame_image, frame_num, caption) for all frames.
        2. best_prompts_list: List of (frame_num, prompt, score) for the best
           prompt from each time interval.
        """
        print(f"Starting video processing...")
        print(f"Video: {video_path}")
        print(f"Processing every frame. Running best prompt selector every {selector_interval_seconds} seconds.")
        print(f"Model: {self.model_name}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps == 0:
            print("Warning: Could not get video FPS. Defaulting to 30.")
            fps = 30
        
        frames_per_chunk = int(fps * selector_interval_seconds)
        print(f"Video FPS: {fps:.2f}. Analyzing chunks of {frames_per_chunk} frames.")

        all_results = []
        best_prompts_list = []
        current_chunk_results = []
        
        frame_count = 0
        transition = " Add a 0.5 second smooth transition to/from this music in a soft-fade form."

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % (int(fps) or 30) == 0: # Print status every second
                 print(f"Processing video frame {frame_count} (Time: {frame_count/fps:.2f}s)")

            # 1. Process the frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            caption = self.generate_music_caption(frame_rgb)
            caption_with_transition = caption + transition
            
            result_data = (frame_rgb, frame_count, caption_with_transition)
            
            # 2. Store results
            all_results.append(result_data)
            current_chunk_results.append(result_data)
            
            frame_count += 1

            # 3. Check if chunk is complete
            if frame_count % frames_per_chunk == 0 and current_chunk_results:
                print(f"\n--- Analyzing chunk ending at frame {frame_count} ({frame_count/fps:.2f}s) ---")
                best_prompt, best_score = self.find_most_detailed_prompt(current_chunk_results)
                
                if best_prompt:
                    # Store (end_frame_num, prompt, score)
                    best_prompts_list.append((frame_count, best_prompt, best_score))
                    print(f"✓ Best of chunk: {best_prompt} (Score: {best_score:.2f})\n")
                
                # Reset chunk
                current_chunk_results = []
        
        # 4. Process the final remaining chunk
        if current_chunk_results:
            print(f"\n--- Analyzing final chunk (frames {frame_count - len(current_chunk_results)} to {frame_count}) ---")
            best_prompt, best_score = self.find_most_detailed_prompt(current_chunk_results)
            
            if best_prompt:
                best_prompts_list.append((frame_count, best_prompt, best_score))
                print(f"✓ Best of final chunk: {best_prompt} (Score: {best_score:.2f})\n")

        cap.release()
        
        print(f"Processing complete! Processed {len(all_results)} total frames.")
        return all_results, best_prompts_list
    
    def display_results(self, results: List[Tuple[np.ndarray, int, str]], 
                       max_frames: int = 6, save_path: str = None): # <-- Changed default to 6
        """Display frames with their music recommendations and wrapped titles"""
        
        # --- CHANGE 1: Select frames and set grid ---
        # We'll use a 3-column grid by default.
        
        # --- NEW: Select frames more intelligently for visualization ---
        # If we have many results, space them out instead of just taking the first 6
        if len(results) > max_frames:
            indices = np.linspace(0, len(results) - 1, max_frames, dtype=int)
            display_results = [results[i] for i in indices]
        else:
            display_results = results
        
        n_frames = len(display_results)
        cols = 3
        
        # Handle case where we have fewer frames than columns
        if n_frames == 0:
            print("No frames processed to display.")
            return
        elif n_frames < cols:
            cols = n_frames
            
        # Calculate rows, ensuring at least 1 row
        rows = (n_frames + cols - 1) // cols
        
        # --- CHANGE 2: Adjust figsize to give titles more vertical space ---
        # Increased from 5 to 7 per row to make room for wrapped text
        fig, axes = plt.subplots(rows, cols, figsize=(15, 7 * rows))
        
        # --- CHANGE 3: Simplify subplot handling with .flatten() ---
        # This works for all cases (1 plot, 1 row, 1 col, or grid)
        if n_frames == 1:
            axes_flat = [axes] # Make it iterable if it's a single plot
        else:
            axes_flat = axes.flatten()

        for i, (frame, frame_num, caption) in enumerate(display_results):
            
            # --- CHANGE 4: Wrap the caption text ---
            # We use textwrap.fill() to automatically add newlines
            # 60 characters is a good width for a 5-inch wide plot
            wrapped_caption = textwrap.fill(caption, width=60)
            
            ax = axes_flat[i]
            ax.imshow(frame)
            
            # Use the new wrapped_caption and slightly smaller font
            ax.set_title(f"Frame {frame_num}\n{wrapped_caption}", fontsize=9)
            ax.axis('off')
        
        # Turn off unused subplots
        for i in range(n_frames, len(axes_flat)):
            axes_flat[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def save_captions_to_file(self, results: List[Tuple[np.ndarray, int, str]], 
                            output_path: str):
        """Save music recommendations to a text file"""
        with open(output_path, 'w') as f:
            f.write("Video Frame-by-Frame Music Recommendations (Generated by BLIP)\n") # Changed BLIP2 to BLIP
            f.write("=" * 70 + "\n")
            f.write("This file contains music recommendations for each video frame.\n")
            f.write("Each recommendation includes genre, mood, tempo, and instrumentation suggestions.\n\n")
            
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Total frames processed: {len(results)}\n\n")
            
            for frame, frame_num, caption in results:
                f.write(f"Frame {frame_num} - Music Recommendation:\n")
                f.write(f"{caption}\n\n")
        
        print(f"Music recommendations saved to: {output_path}")

    def find_most_detailed_prompt(self,results: List[Tuple[np.ndarray, int, str]]):
        """
        Analyzes a list of result tuples to find the one with the highest detail score.

        Args:
            results: A list of (frame, frame_num, prompt) tuples.

        Returns:
            A tuple containing the most detailed prompt (string) and its score (float).
        """
        if not results:
            return ("", 0.0)
        
        max_score = -1.0
        most_detailed_prompt = ""

        # print("--- Prompt Analysis (Chunk) ---") # <-- Too verbose
        for frame_data in results:
            prompt_text = frame_data[2]
            score = self.get_detail_score(prompt_text)
            # print(f"Prompt: '{prompt_text}'") # <-- Too verbose
            # print(f"Detail Score: {score:.2f}") # <-- Too verbose

            if score > max_score:
                max_score = score
                most_detailed_prompt = prompt_text

        # print("-----------------------")
        return most_detailed_prompt, max_score


    


def main():
    parser = argparse.ArgumentParser(description="Generate music recommendations for video using BLIP") # Changed BLIP2 to BLIP
    parser.add_argument("video_path", help="Path to the input video file")
    
    # --- CHANGE: Removed frame_interval, added selector_interval ---
    parser.add_argument("--selector_interval", type=int, default=5, 
                       help="Run the best prompt selector every N seconds (default: 5)")
    
    parser.add_argument("--max_display", type=int, default=12,
                       help="Maximum frames to display in visualization (default: 12)")
    parser.add_argument("--output_dir", default="./blip_output", # Changed blip2_output to blip_output
                       help="Output directory for results (default: ./blip_output)")
    
    # --- CHANGE 6: Update Default Model to BLIP V1 Base ---
    parser.add_argument("--model", default="Salesforce/blip-image-captioning-base",
                       help="BLIP model name (e.g., Salesforce/blip-image-captioning-base).")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize converter with the selected model
    converter = VideoToMusicConverter(model_name=args.model)
    
    print("Starting video processing...")
    
    # --- CHANGE: Updated process_video call to new signature ---
    all_results, best_prompts_list = converter.process_video(
        args.video_path, 
        args.selector_interval
    )
  
    
    if not all_results:
        print("No results were generated. Exiting.")
        return
        
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    captions_file = os.path.join(args.output_dir, f"{video_name}_music_captions.txt")
    
    # --- CHANGE: Use all_results for saving and display ---
    converter.save_captions_to_file(all_results, captions_file)
    
    visualization_path = os.path.join(args.output_dir, f"{video_name}_music_visualization.png")
    converter.display_results(all_results, max_frames=args.max_display, 
                            save_path=visualization_path)
    
    # --- CHANGE: Removed old top_prompt analysis, print list instead ---
    print("\n" + "="*30)
    print(f" BEST PROMPT FROM EACH {args.selector_interval}-SECOND CHUNK ")
    print("="*30)
    if not best_prompts_list:
        print("No prompts were generated.")
    else:
        for (frame_num, prompt, score) in best_prompts_list:
            print(f"[Chunk ending at frame ~{frame_num}] (Score: {score:.2f}):\n{prompt}\n")
    
    print(f"Processing complete! Results saved in: {args.output_dir}")

    # Return the list of best prompts
    return best_prompts_list


if __name__ == "__main__":
    main()