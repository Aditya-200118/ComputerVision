import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 12, "figure.dpi": 600,
})

def load_and_preprocess(path):
    """Loads a flat 512x512 .img file, skipping the 512-byte header."""
    try:
        with open(path, "rb") as f:
            _ = f.read(512)
            img_data = np.fromfile(f, dtype=np.uint8)
        return img_data.reshape((512, 512))
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()

import numpy as np

def extract_dynamic_heuristics(image):
    """
    Automatically derives heuristics from the image's statistical profile.
    This fulfills the requirement to choose threshold values automatically.
    """
    mean_val = np.mean(image)
    median_val = np.median(image)
    std_val = np.std(image)
    
    # Calculate higher-order statistical moments
    if std_val > 0:
        skewness = np.mean(((image - mean_val) / std_val) ** 3)
        # Fisher's definition of excess kurtosis (subtract 3)
        kurtosis = np.mean(((image - mean_val) / std_val) ** 4) - 3
    else:
        skewness = 0
        kurtosis = 0

    # Heuristic 1: Iterative Seed (Combats skewness) 
    t0_estimate = (mean_val + median_val) / 2.0

    # Heuristic 2: Peakiness Minimum Distance
    # JUSTIFICATION for 40: In an 8-bit image, the total dynamic range is 256. 
    # A distance of 40 represents ~15% of the total intensity spectrum. Peaks closer 
    # than 15% are statistically highly likely to be intra-class variations (e.g., two 
    # slightly different shadows on the same object) rather than distinct inter-class objects.
    min_distance = int(max(40, std_val * 0.6))
    
    # Heuristic 3: Dynamic Window Size (based on Kurtosis)
    # High kurtosis (sharp peaks) -> smaller window to preserve peaks.
    # Low/Negative kurtosis (flat/noisy) -> larger window to smooth noise aggressively.
    raw_window = 5 - (kurtosis * 2)
    window_size = int(round(raw_window))
    if window_size % 2 == 0:
        window_size += 1  # Force to an odd integer for symmetrical convolution
    window_size = max(3, min(15, window_size)) # Clamp valid kernel sizes

    # Heuristic 4: Dynamic Region Growing Multiplier (based on Skewness)
    # JUSTIFICATION for 0.3: Dropping below 0.3 standard deviations practically eliminates 
    # the hysteresis band. If the multiplier drops to 0, T_L equals T_H, which degenerates 
    # the Dual-Threshold algorithm back into a Single Global Threshold algorithm, entirely 
    # defeating the purpose of region growing. 0.3 ensures a minimal functional hysteresis band.
    dynamic_multiplier = max(0.3, 1.0 - abs(skewness))

    # Calculate Foreground Standard Deviation (sigma_fg) for Region Growing
    T_temp = t0_estimate
    while True:
        G1 = image[image > T_temp]
        G2 = image[image <= T_temp]
        # Safety check if one group is completely empty
        mu1 = np.mean(G1) if len(G1) > 0 else 0
        mu2 = np.mean(G2) if len(G2) > 0 else 0
        
        T_new = (mu1 + mu2) / 2.0
        if abs(T_new - T_temp) < 1.0: break
        T_temp = T_new
    
    fg_pixels = image[image >= T_temp]
    sigma_fg = np.std(fg_pixels) if len(fg_pixels) > 0 else 0
    print(f"window size: {window_size}\n")
    # window_size = 5
    return min_distance, t0_estimate, sigma_fg, window_size, dynamic_multiplier

# def threshold_peakiness(image, min_distance, win_size, prominence_ratio=0.05):
#     """
#     Task 1: Thresholding using peakiness detection.
#     Evaluates peak pairs based on prominence and distance, selecting the 
#     valley that maximizes the peak-to-valley ratio.
#     """
    
#     kernel_size = (win_size * 2) + 1
#     print(f"Inside the function: {kernel_size}")
#     hist = np.bincount(image.ravel(), minlength=256)
#     smoothed_hist = np.convolve(hist, np.ones(kernel_size)/kernel_size, mode='same')
    
#     prominence_threshold = np.max(smoothed_hist) * prominence_ratio
#     peaks = [i for i in range(1, 255) 
#              if smoothed_hist[i] > smoothed_hist[i-1] 
#              and smoothed_hist[i] > smoothed_hist[i+1] 
#              and smoothed_hist[i] > prominence_threshold]
                
#     best_valley = 0
#     max_peakiness = -1
    
#     for i in range(len(peaks)):
#         for j in range(i + 1, len(peaks)):
#             p1, p2 = peaks[i], peaks[j]
            
#             if abs(p1 - p2) < min_distance:
#                 continue
                
#             valley_idx = p1 + np.argmin(smoothed_hist[p1:p2+1])
#             valley_height = smoothed_hist[valley_idx]
            
#             peakiness = min(smoothed_hist[p1], smoothed_hist[p2]) / (valley_height + 1e-6)
                
#             if peakiness > max_peakiness:
#                 max_peakiness = peakiness
#                 best_valley = valley_idx
                
#     T = best_valley if best_valley > 0 else int(np.mean(image))
#     return (image >= T).astype(np.uint8) * 255, T

def threshold_peakiness(image, min_distance, win_size, prominence_ratio=0.05):
    """
    Task 1: Thresholding using peakiness detection.
    Relies on dynamically calculated 'min_distance', 'win_size', and 'prominence_ratio' 
    to filter out local clustering and structural noise based on the image's statistical profile.
    """
    hist = np.zeros(256, dtype=int)
    for val in image.ravel():
        hist[val] += 1
        
    # Replaced the hardcoded 5 with the dynamic Kurtosis-driven win_size
    window = win_size
    smoothed_hist = np.zeros(256, dtype=float)
    for i in range(256):
        start = max(0, i - window)
        end = min(255, i + window)
        smoothed_hist[i] = np.mean(hist[start:end+1])
        
    max_hist_val = np.max(smoothed_hist)
    prominence_threshold = max_hist_val * prominence_ratio
    
    peaks = []
    for i in range(1, 255):
        if smoothed_hist[i] > smoothed_hist[i-1] and smoothed_hist[i] > smoothed_hist[i+1]:
            if smoothed_hist[i] > prominence_threshold:
                peaks.append(i)
                
    best_peak_pair = None
    best_valley = 0
    max_peakiness = -1
    
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            p1 = peaks[i]
            p2 = peaks[j]
            
            # Apply custom profile-driven minimum distance
            if abs(p1 - p2) < min_distance:
                continue
                
            valley_idx = p1 + np.argmin(smoothed_hist[p1:p2+1])
            valley_height = smoothed_hist[valley_idx]
            
            h_p1 = smoothed_hist[p1]
            h_p2 = smoothed_hist[p2]
            
            if valley_height == 0:
                peakiness = float('inf')
            else:
                peakiness = min(h_p1, h_p2) / valley_height
                
            if peakiness > max_peakiness:
                max_peakiness = peakiness
                best_peak_pair = (p1, p2)
                best_valley = valley_idx
                
    T = best_valley if best_valley > 0 else int(np.mean(image))
    binary_image = (image >= T).astype(np.uint8) * 255
    return binary_image, T

def threshold_iterative(image, t0_estimate, epsilon=1.0):
    """
    Task 2: Iterative thresholding.
    Converges on the threshold that minimizes intra-class variance.
    """
    T_old = t0_estimate
    while True:
        G1 = image[image > T_old]
        G2 = image[image <= T_old]
        mu1 = np.mean(G1) if len(G1) > 0 else 0
        mu2 = np.mean(G2) if len(G2) > 0 else 0
        T_new = (mu1 + mu2) / 2.0
        
        if abs(T_new - T_old) < epsilon: break
        T_old = T_new
        
    T_final = int(T_new)
    return (image >= T_final).astype(np.uint8) * 255, T_final

def threshold_dual_region_growing(image, sigma_fg, multiplier=0.75):
    """
    Task 3: Dual thresholding with Breadth-First region growing.
    T_H is found iteratively. T_L is derived from foreground variance.
    """
    # 1. Automatic High Threshold (T_H)
    T_old = np.mean(image)
    while True:
        G1 = image[image > T_old]
        G2 = image[image <= T_old]
        T_new = (np.mean(G1) + np.mean(G2)) / 2.0
        if abs(T_new - T_old) < 1.0: break
        T_old = T_new
    T_H = int(T_new)
    
    # 2. Heuristic Low Threshold (T_L)
    T_L = max(0, int(T_H - (sigma_fg * multiplier)))
            
    # 3. BFS Region Growing
    rows, cols = image.shape
    binary_image = np.zeros((rows, cols), dtype=np.uint8)
    
    seed_r, seed_c = np.where(image >= T_H)
    seeds = deque(zip(seed_r, seed_c))
    
    for r, c in seeds: binary_image[r, c] = 255
            
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    while seeds:
        r, c = seeds.popleft() 
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if binary_image[nr, nc] == 0 and image[nr, nc] >= T_L:
                    binary_image[nr, nc] = 255 
                    seeds.append((nr, nc)) 
                    
    return binary_image, T_H, T_L

if __name__ == "__main__":
    output_dir = "assignment3_ver7_figures"
    os.makedirs(output_dir, exist_ok=True)
    files = ["test1.img", "test2.img", "test3.img"]
    
    for file in files:
        if not os.path.exists(file): continue
            
        print(f"\nProcessing {file}...")
        img = load_and_preprocess(file)
        min_dist, t0_seed, sigma_fg, win_size, dyn_mult = extract_dynamic_heuristics(img)
        base_name = file.split('.')[0]
        
        res_peak, t_peak = threshold_peakiness(img, min_dist, win_size)
        plt.imsave(f"{output_dir}/{base_name}_peakiness.png", res_peak, cmap='gray')
        
        res_iter, t_iter = threshold_iterative(img, t0_seed)
        plt.imsave(f"{output_dir}/{base_name}_iterative.png", res_iter, cmap='gray')
        
        res_dual, th, tl = threshold_dual_region_growing(img, sigma_fg, dyn_mult)
        plt.imsave(f"{output_dir}/{base_name}_dual_region.png", res_dual, cmap='gray')