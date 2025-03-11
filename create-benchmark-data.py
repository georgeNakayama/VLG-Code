import json
from pathlib import Path
import random

import cv2
import numpy as np
import torch
import tqdm

def apply_random_zoom(image):
    # Apply random zoom to the image if it's not a zero tensor (TEXT or REASON types)
    if not isinstance(image, torch.Tensor):
            
            # Random zoom factor between 0.8 (zoom out) and 1.2 (zoom in)
            zoom_factor = random.uniform(0.5, 2)
            
            # Calculate new size while maintaining aspect ratio
            h, w = image.shape[:2] 
            new_h = int(h * zoom_factor)
            new_w = int(w * zoom_factor)
            
            # Resize image using cv2
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Center crop or pad to original size
            if zoom_factor > 1:  # Zoomed in - need to crop
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                image = image[start_h:start_h+h, start_w:start_w+w]
            else:  # Zoomed out - need to pad
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                pad_h_top = pad_h
                pad_h_bottom = pad_h if (h-new_h)%2==0 else pad_h+1
                pad_w_left = pad_w  
                pad_w_right = pad_w if (w-new_w)%2==0 else pad_w+1
                
                image = np.pad(
                    image,
                    ((pad_h_top, pad_h_bottom), 
                     (pad_w_left, pad_w_right),
                     (0, 0)),
                    mode='constant',
                    constant_values=0
            )

    return image


def apply_random_translation(image):
    # Apply random translation to the image
    if not isinstance(image, torch.Tensor):
        h, w = image.shape[:2]
        
        # Random translation in x and y direction (-20% to +20% of image size)
        tx = int(random.uniform(-0.4, 0.4) * w)
        ty = int(random.uniform(-0.4, 0.4) * h)
        
        # Create translation matrix
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # Apply translation
        translated = cv2.warpAffine(image, translation_matrix, (w, h))
        
        # Create white background
        white_background = np.ones_like(image) * 255
        
        # Copy valid parts of translated image onto white background
        image = white_background.copy()
        
        # Calculate valid regions after translation
        y_start = max(0, -ty)
        y_end = min(h, h - ty)
        x_start = max(0, -tx) 
        x_end = min(w, w - tx)
        
        # Source regions in translated image
        src_y_start = max(0, ty)
        src_y_end = min(h, h + ty)
        src_x_start = max(0, tx)
        src_x_end = min(w, w + tx)
        
        # Copy valid region
        image[y_start:y_end, x_start:x_end] = translated[src_y_start:src_y_end, src_x_start:src_x_end]
        return image


def apply_random_background(image):
        # Create a random bright background color (ensuring brightness > 128)
        if not isinstance(image, torch.Tensor):
            random_color = [
                random.randint(128, 255),  # R
                random.randint(128, 255),  # G 
                random.randint(128, 255)   # B
            ]
            # Only change white pixels to random color
            white_mask = np.all(image == 255, axis=-1)
            image[white_mask] = random_color
            return image


def main():
    
    # Load the validation split from the dataset
    split_file = Path("./assets/garmentcodedatav2_datasplit.json")
    assert split_file.exists(), f"Split file not found at {split_file}"
    
    with open(split_file, "r") as f:
        splits = json.load(f)
    
    val_datapoints = splits["val"]
    
    # Define paths similar to those in FullDataset
    root_dir = "/scratch/m000051/garment_gang/data/garmentcodedatav2"
    
    # Create output directory for transformed images
    output_dir = Path("benchmark_data")
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for each transformation
    translation_dir = output_dir / "translation"
    background_dir = output_dir / "background"
    zoom_dir = output_dir / "zoom"
    translation_dir.mkdir(exist_ok=True)
    background_dir.mkdir(exist_ok=True)
    zoom_dir.mkdir(exist_ok=True)
    
    # Randomly select 40 datapoints
    # Select 40 datapoints for each transformation
    translation_datapoints = random.sample(val_datapoints, 40)
    background_datapoints = random.sample(val_datapoints, 40)
    zoom_datapoints = random.sample(val_datapoints, 40)
    
    # Process translation datapoints
    for datapoint_name in tqdm.tqdm(translation_datapoints, desc="Processing translations"):
        # Get the data name (similar to FullDataset)
        data_name = datapoint_name.split("/")[-1]
        
        image_path = Path(root_dir) / datapoint_name / f"{data_name}_render_back.png"
        
        if not image_path.exists():
            print(f"Warning: Image not found at {image_path}, skipping")
            continue
        
        # Read the image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply translation
        translated_image = apply_random_translation(image)
        
        # Save the transformed image
        translated_image_bgr = cv2.cvtColor(translated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(translation_dir / f"{data_name}.png"), translated_image_bgr)
    
    # Process background datapoints
    for datapoint_name in tqdm.tqdm(background_datapoints, desc="Processing backgrounds"):
        # Get the data name (similar to FullDataset)
        data_name = datapoint_name.split("/")[-1]
        
        image_path = Path(root_dir) / datapoint_name / f"{data_name}_render_back.png"
        
        if not image_path.exists():
            print(f"Warning: Image not found at {image_path}, skipping")
            continue
        
        # Read the image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply background change
        background_image = apply_random_background(image.copy())
        
        # Save the transformed image
        background_image_bgr = cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(background_dir / f"{data_name}.png"), background_image_bgr)
    
    # Process zoom datapoints (placeholder for future implementation)
    for datapoint_name in tqdm.tqdm(zoom_datapoints, desc="Processing zooms"):
        # Get the data name (similar to FullDataset)
        data_name = datapoint_name.split("/")[-1]
        
        image_path = Path(root_dir) / datapoint_name / f"{data_name}_render_back.png"
        
        if not image_path.exists():
            print(f"Warning: Image not found at {image_path}, skipping")
            continue
        
        # Read the image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply zoom transformation
        zoomed_image = apply_random_zoom(image.copy())
        
        # Save the transformed image
        zoomed_image_bgr = cv2.cvtColor(zoomed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(zoom_dir / f"{data_name}.png"), zoomed_image_bgr)
    
    print(f"Saved {len(translation_datapoints)} translation images, {len(background_datapoints)} background images, and {len(zoom_datapoints)} zoom images to {output_dir}")


if __name__ == "__main__":
    main()
