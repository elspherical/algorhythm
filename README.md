# algorhythm 🎶
**Members**: Anaisha Das, Dominic Bankovitch, Haasini Paparaju, Xuan Zhang, Erica Liu (PL)

This is **Algorhythm**, a Fa'25 Launchpad project! This project aims to overlay music over inputted videos with minimal processing time. The chosen music fits the vibe/emotion/content of the video by running screenshots at 5-second intervals through CLIP (Contrastive Language-Image Pre-training), a multilayer perceptron (MLP), and a nearest-neighbor search to EMID (Emotionally paired Music and Image Dataset): https://huggingface.co/datasets/ecnu-aigc/EMID.

## Instructions
Run ```conda env create -f environment.yml``` to create conda environment for this project. Then, use ```conda activate algorhythm``` to activate the environment. 
Navigate to ```final/``` and run ```python3 app.py``` and open provided localhost URL for interactive UI. In this UI, you can either upload or record a video for processing. Because processing is every 5 seconds, ensure that the video is more than 5 seconds long! You can monitor video processing progress in terminal.

To process a video locally, run ```python3 process_video.py <video_path> <output_path>```.

## Known limitations
- There are no working transitions implemented, so music shifts abruptly every 5 seconds.
- Some generated outputs can be inconsistent because of limitations of MLP and data.
- Because model was trained only on EMID, only videos that bear resemblance to the media in the dataset will receive accurate results.
- Long videos may suffer from long processing times.

## Examples
Input: https://github.com/user-attachments/assets/4592e214-0377-4efd-8195-f2e4a6cd33be \
Output: https://github.com/user-attachments/assets/2c25e364-3066-4cfb-9b71-e2674049fda3 
________________

Input: 
https://github.com/user-attachments/assets/00a26931-414a-4d3e-a4ab-f18a394a9358 \
Output (yes, that is the pen-pineapple-apple-pen song!):
https://github.com/user-attachments/assets/92c98051-5404-4332-b356-4339bf95aa71


Note: folders that are not "final" reflect intermediate work on the project; everything needed for demo/usage is in ```final/```. Furthermore, dependencies in ```environment.yml``` are not all necessary; they were accumulated over the course of the project.
