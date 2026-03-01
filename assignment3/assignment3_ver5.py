import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "figure.dpi": 600,
    }
)

def load_and_preprocess(path):
    try:
        with open(path, "rb") as f:
            # Skip the 512-byte header typical of these .img files
            _ = f.read(512)
            img_data = np.fromfile(f, dtype=np.uint8)
        return img_data.reshape((512, 512))
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()


def threshold_peakiness(image, min_distance, prominence_ratio):
    """
    Task 1: Thresholding using peakiness detection.
    Relies on profile-driven 'min_distance' and 'prominence_ratio' 
    to filter out local clustering and structural noise.
    """
    hist = np.zeros(256, dtype=int)
    for val in image.ravel():
        hist[val] += 1
        
    window = 5
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
    Accepts a profile-driven initial estimate (T0) to bypass distribution skewness.
    """
    T_old = t0_estimate
    
    while True:
        G1 = image[image > T_old]
        G2 = image[image <= T_old]
        
        mu1 = np.mean(G1) if len(G1) > 0 else 0
        mu2 = np.mean(G2) if len(G2) > 0 else 0
        
        T_new = (mu1 + mu2) / 2.0
        
        if abs(T_new - T_old) < epsilon:
            break
        T_old = T_new
        
    T_final = int(T_new)
    binary_image = (image >= T_final).astype(np.uint8) * 255
    return binary_image, T_final


def threshold_dual_region_growing(image, sigma_fg, sigma_multiplier=0.5):
    """
    Task 3: Dual thresholding with region growing.
    T_H is found automatically via iterative variance separation. 
    T_L is calculated dynamically using a strictly weighted intra-class variance to prevent leakage.
    """
    # 1. Automatic High Threshold (T_H) Selection
    T_old = np.mean(image)
    while True:
        G1 = image[image > T_old]
        G2 = image[image <= T_old]
        mu1 = np.mean(G1) if len(G1) > 0 else 0
        mu2 = np.mean(G2) if len(G2) > 0 else 0
        T_new = (mu1 + mu2) / 2.0
        if abs(T_new - T_old) < 1.0:
            break
        T_old = T_new
        
    T_H = int(T_new)
    
    # 2. STRICTER Low Threshold (T_L) using Profile Data
    # We multiply sigma_fg by a fraction (e.g., 0.5) to tighten the tolerance.
    T_L = max(0, int(T_H - (sigma_fg * sigma_multiplier)))
            
    # 3. Region Growing (8-Connected)
    rows, cols = image.shape
    binary_image = np.zeros((rows, cols), dtype=np.uint8)
    
    # Optimization: Use numpy to find seeds instantly instead of a double for-loop
    seed_r, seed_c = np.where(image >= T_H)
    seeds = list(zip(seed_r, seed_c))
    
    # Pre-mark all seeds as foreground (255)
    for r, c in seeds:
        binary_image[r, c] = 255
            
    # The 8 neighboring pixel directions
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    # Grow the regions
    while seeds:
        r, c = seeds.pop()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            # Check boundaries
            if 0 <= nr < rows and 0 <= nc < cols:
                # If pixel is unvisited AND passes the relaxed threshold
                if binary_image[nr, nc] == 0 and image[nr, nc] >= T_L:
                    binary_image[nr, nc] = 255  # Mark as foreground
                    seeds.append((nr, nc))      # Add to queue to check its neighbors
                    
    return binary_image, T_H, T_L


if __name__ == "__main__":
    output_dir = "assignment3_ver_figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Data parsed directly from our deep profiling script
    image_configs = {
    "test1.img": {
            # std_dev = 68.76 -> max(40, 68.76 * 0.6)
            "min_distance": 41,        
            "prominence_ratio": 0.05,
            # (Mean 93.89 + Median 81.00) / 2 to combat positive skew [cite: 47]
            "t0_estimate": 87.45,       
            "sigma_fg": 42.38          
        },
        "test2.img": {
            # std_dev = 68.15 -> max(40, 68.15 * 0.6)
            "min_distance": 41,        
            "prominence_ratio": 0.05,
            # (Mean 68.63 + Median 52.00) / 2 
            "t0_estimate": 60.32,       
            "sigma_fg": 48.09          
        },
        "test3.img": {
            # std_dev = 75.47 -> max(40, 75.47 * 0.6)
            "min_distance": 45,        
            "prominence_ratio": 0.05,
            # (Mean 116.63 + Median 104.00) / 2 [cite: 148]
            "t0_estimate": 110.32,      
            "sigma_fg": 31.63          
        }
    }
    
    for file, config in image_configs.items():
        if not os.path.exists(file):
            print(f"Skipping {file}: File not found.")
            continue
            
        print(f"\nProcessing {file} using extracted profile parameters...")
        img = load_and_preprocess(file)
        
        base_name = file.split('.')[0]
        
        # Task 1
        res_peakiness, t_peak = threshold_peakiness(
            img, 
            min_distance=config["min_distance"], 
            prominence_ratio=config["prominence_ratio"]
        )
        print(f"  Peakiness Threshold: {t_peak} (Distance cutoff: {config['min_distance']})")
        plt.imsave(f"{output_dir}/{base_name}_peakiness.png", res_peakiness, cmap='gray')
        
        # Task 2
        res_iterative, t_iter = threshold_iterative(
            img, 
            t0_estimate=config["t0_estimate"]
        )
        print(f"  Iterative Threshold: {t_iter} (Seeded at {config['t0_estimate']})")
        plt.imsave(f"{output_dir}/{base_name}_iterative.png", res_iterative, cmap='gray')
        
        # Task 3
        res_dual, th, tl = threshold_dual_region_growing(
            img, 
            sigma_fg=config["sigma_fg"]
        )
        print(f"  Dual Region Growing: High={th}, Low={tl} (sigma_fg={config['sigma_fg']})")
        plt.imsave(f"{output_dir}/{base_name}_dual_region.png", res_dual, cmap='gray')

    print(f"\nProcessing complete. Images generated using pure parameter injection.")