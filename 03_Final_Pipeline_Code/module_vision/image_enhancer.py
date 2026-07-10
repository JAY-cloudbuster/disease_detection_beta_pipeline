import cv2
import numpy as np
import os
import torch
import subprocess
from PIL import Image

def assess_quality_simple(img_path):
    """
    Fallback quality assessment using basic metrics if piq/brisque is unavailable.
    Returns (quality_score, needs_enhancement).
    """
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return 0, False
    
    # Calculate simple Shannon Entropy
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    
    # Needs enhancement if entropy is low (e.g. < 7.0 for 8-bit images)
    needs_enhancement = entropy < 7.0
    return entropy, needs_enhancement

def apply_clahe(img_path, clip_limit=2.0, tile_grid=(8, 8)):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) as a fast classical enhancement.
    """
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Could not read image for CLAHE: {img_path}")
        
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    enhanced = clahe.apply(img_gray)
    
    # Convert grayscale -> RGB for downstream models
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return enhanced_rgb

class ImageEnhancer:
    def __init__(self, use_unimie=False, unimie_ckpt_path=None):
        self.use_unimie = use_unimie
        self.unimie_ckpt_path = unimie_ckpt_path

    def enhance(self, img_path):
        """
        Evaluates image quality and applies enhancement if necessary.
        Defaults to CLAHE as it is fast and requires no GPU memory.
        If use_unimie=True, it will attempt diffusion enhancement.
        Returns:
            enhanced_rgb (numpy array RGB)
            action_taken (str: 'Original', 'CLAHE', 'UniMIE')
        """
        entropy, needs_enhancement = assess_quality_simple(img_path)
        
        if not needs_enhancement:
            # High quality, no enhancement needed
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img_rgb, "Original (High Quality)"
            
        if self.use_unimie and self.unimie_ckpt_path and os.path.exists(self.unimie_ckpt_path):
            # Advanced Diffusion Enhancement (Pseudo-code wrapper)
            print("Applying UniMIE Diffusion Enhancement (this may take a while)...")
            # In a real environment, you would call the UniMIE sub-process here as done in the notebook.
            # For robustness in demo, if UniMIE fails or is too slow, we fall back to CLAHE.
            try:
                # TODO: Execute sample_x0_enhancement_MI.py via subprocess here
                pass 
            except Exception as e:
                print(f"UniMIE failed: {e}. Falling back to CLAHE.")
                return apply_clahe(img_path), "CLAHE (Fallback)"
            
        # Default classical enhancement
        return apply_clahe(img_path), "CLAHE"
