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
            _ = f.read(512)
            img_data = np.fromfile(f, dtype=np.uint8)
        return img_data.reshape((512, 512))
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()

def threshold_peakiness(image, min_distance, prominence_ratio):
    """
    Task 1: Thresholding using peakiness detection.
    Evaluation criteria parameters are passed based on image profiling.
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
    peaks = []
    
    # Apply custom prominence ratio from profiling
    for i in range(1, 255):
        if smoothed_hist[i] > smoothed_hist[i-1] and smoothed_hist[i] > smoothed_hist[i+1]:
            if smoothed_hist[i] > (max_hist_val * prominence_ratio):
                peaks.append(i)
                
    best_peak_pair = None
    best_valley = 0
    max_peakiness = -1
    
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            p1 = peaks[i]
            p2 = peaks[j]
            
            # Apply custom minimum distance from profiling
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

def threshold_iterative(image, epsilon=1.0):
    """
    Task 2: Iterative thresholding.
    Self-contained; dynamically starts at the image mean extracted from profiling.
    """
    T_old = np.mean(image)
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

def threshold_dual_region_growing(image, drop_ratio):
    """
    Task 3: Dual thresholding with region growing.
    Uses iterative logic to automatically find T_H from the histogram, 
    then applies the customized drop_ratio heuristic for T_L.
    """
    # Automatically find optimal T_H boundary
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
    
    # Apply custom heuristic derived from profiling
    T_L = int(T_H * drop_ratio)
            
    rows, cols = image.shape
    binary_image = np.zeros((rows, cols), dtype=np.uint8)
    seeds = []
    
    for r in range(rows):
        for c in range(cols):
            if image[r, c] >= T_H:
                binary_image[r, c] = 255
                seeds.append((r, c))
                
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    while seeds:
        r, c = seeds.pop()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if binary_image[nr, nc] == 0 and image[nr, nc] >= T_L:
                    binary_image[nr, nc] = 255
                    seeds.append((nr, nc))
                    
    return binary_image, T_H, T_L

if __name__ == "__main__":
    output_dir = "assignment3_figures_ver3"
    os.makedirs(output_dir, exist_ok=True)
    
    # Our Custom Profiling Configuration Data
    image_configs = {
        "test1.img": {
            "min_distance": 80,       # Peaks 51/53/55 need large distance to find true foreground
            "prominence_ratio": 0.05, 
            "drop_ratio": 0.60        # Found via OpenCV Otsu comparison
        },
        "test2.img": {
            "min_distance": 50,       # Tighter spread, peaks at 17, 19, 77
            "prominence_ratio": 0.05,
            "drop_ratio": 0.55        # Slightly sharper drop due to higher standard deviation
        },
        "test3.img": {
            "min_distance": 60,       # High mean (116.6), peaks cluster at 230+
            "prominence_ratio": 0.05,
            "drop_ratio": 0.65        # Flatter edges (Mean grad 5.59), need tighter region bounds
        }
    }
    
    for file, config in image_configs.items():
        if not os.path.exists(file):
            print(f"Skipping {file}: File not found.")
            continue
            
        print(f"\nProcessing {file} using custom profile data...")
        img = load_and_preprocess(file)
        
        base_name = file.split('.')[0]
        
        # Task 1
        res_peakiness, t_peak = threshold_peakiness(img, min_distance=config["min_distance"], prominence_ratio=config["prominence_ratio"])
        print(f"  Peakiness Threshold: {t_peak} (Dist >= {config['min_distance']})")
        plt.imsave(f"{output_dir}/{base_name}_peakiness.png", res_peakiness, cmap='gray')
        
        # Task 2
        res_iterative, t_iter = threshold_iterative(img)
        print(f"  Iterative Threshold: {t_iter}")
        plt.imsave(f"{output_dir}/{base_name}_iterative.png", res_iterative, cmap='gray')
        
        # Task 3
        res_dual, th, tl = threshold_dual_region_growing(img, drop_ratio=config["drop_ratio"])
        print(f"  Dual Region Growing: High={th}, Low={tl} (Drop Ratio={config['drop_ratio']})")
        plt.imsave(f"{output_dir}/{base_name}_dual_region.png", res_dual, cmap='gray')

    print(f"\nProcessing complete. Images generated from custom profiling data saved.")