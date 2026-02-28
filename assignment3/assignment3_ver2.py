import numpy as np
import matplotlib.pyplot as plt
import hashlib
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

        original_2d = img_data.reshape((512, 512))
        binary = (original_2d <= 128).astype(np.uint8)
        return original_2d, binary
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()


def threshold_peakiness(image):
    """
    Task 1: Thresholding using peakiness detection.
    Evaluates peaks based on distance, prominence, and valley depth.
    """
    # 1. Compute Histogram manually
    hist = np.zeros(256, dtype=int)
    for val in image.ravel():
        hist[val] += 1
        
    # 2. Smooth Histogram (Moving Average) to reduce noise spikes
    window = 5
    smoothed_hist = np.zeros(256, dtype=float)
    for i in range(256):
        start = max(0, i - window)
        end = min(255, i + window)
        smoothed_hist[i] = np.mean(hist[start:end+1])
        
    # 3. Find Candidate Peaks (Local Maxima)
    peaks = []
    for i in range(1, 255):
        if smoothed_hist[i] > smoothed_hist[i-1] and smoothed_hist[i] > smoothed_hist[i+1]:
            peaks.append(i)
            
    # 4. Evaluate Distance, Prominence, and Peakiness
    best_peak_pair = None
    best_valley = 0
    max_peakiness = -1
    
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            p1 = peaks[i]
            p2 = peaks[j]
            
            # Distance Criterion: Peaks must be separated by distinct grayscale levels
            if abs(p1 - p2) < 30:
                continue
                
            # Find the deepest valley between the two peaks
            valley_idx = p1 + np.argmin(smoothed_hist[p1:p2+1])
            valley_height = smoothed_hist[valley_idx]
            
            # Prominence / Peakiness Criterion: min(H(p1), H(p2)) / H(valley)
            h_p1 = smoothed_hist[p1]
            h_p2 = smoothed_hist[p2]
            
            # Prevent division by zero
            if valley_height == 0:
                peakiness = float('inf')
            else:
                peakiness = min(h_p1, h_p2) / valley_height
                
            # Maximize the peakiness to find the clearest bimodal separation
            if peakiness > max_peakiness:
                max_peakiness = peakiness
                best_peak_pair = (p1, p2)
                best_valley = valley_idx
                
    # 5. Apply the discovered threshold (Default to 128 if purely unimodal)
    T = best_valley if best_valley > 0 else 128
    binary_image = (image >= T).astype(np.uint8) * 255
    
    return binary_image, T


def threshold_iterative(image, epsilon=1.0):
    """
    Task 2: Iterative thresholding.
    Converges by repeatedly averaging the means of the partitioned foreground and background.
    """
    # Initial estimate T0 using the global mean
    T_old = np.mean(image)
    
    while True:
        # Partition into two groups based on the current threshold
        G1 = image[image > T_old]
        G2 = image[image <= T_old]
        
        # Calculate the means of both groups
        mu1 = np.mean(G1) if len(G1) > 0 else 0
        mu2 = np.mean(G2) if len(G2) > 0 else 0
        
        # Compute the new threshold candidate
        T_new = (mu1 + mu2) / 2.0
        
        # Check for convergence against the epsilon tolerance
        if abs(T_new - T_old) < epsilon:
            break
            
        T_old = T_new
        
    binary_image = (image >= T_new).astype(np.uint8) * 255
    
    return binary_image, T_new


def threshold_dual_region_growing(image):
    """
    Task 3: Dual thresholding with region growing.
    Uses Cumulative Distribution Function (CDF) heuristics for threshold selection.
    """
    # 1. Automatic Threshold Selection Logic via Histogram CDF
    hist = np.zeros(256, dtype=int)
    for val in image.ravel():
        hist[val] += 1
        
    cdf = np.cumsum(hist) / image.size
    
    T_H = 0
    T_L = 0
    
    # Heuristic Logic: 
    # High threshold targets the top 10% brightest pixels (confident seeds).
    # Low threshold targets the top 25% brightest pixels (relaxed connectivity).
    for i in range(256):
        if cdf[i] >= 0.75 and T_L == 0:
            T_L = i
        if cdf[i] >= 0.90 and T_H == 0:
            T_H = i
            break
            
    # 2. Region Growing (8-connected flood-fill via stack)
    rows, cols = image.shape
    binary_image = np.zeros((rows, cols), dtype=np.uint8)
    seeds = []
    
    # Seed phase: Flag all highly confident pixels
    for r in range(rows):
        for c in range(cols):
            if image[r, c] >= T_H:
                binary_image[r, c] = 255
                seeds.append((r, c))
                
    # Directional array for 8-connected neighbors
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    # Growth phase: Iteratively attach neighbors that meet the relaxed threshold
    while seeds:
        r, c = seeds.pop()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            # Boundary checks
            if 0 <= nr < rows and 0 <= nc < cols:
                # If neighbor is unassigned and passes the low threshold, add it
                if binary_image[nr, nc] == 0 and image[nr, nc] >= T_L:
                    binary_image[nr, nc] = 255
                    seeds.append((nr, nc))
                    
    return binary_image, T_H, T_L


if __name__ == "__main__":
    # Create output directory for figures
    output_dir = "assignment3_figures_ver2"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test images specified in the assignment
    files = ["test1.img", "test2.img", "test3.img"]
    
    for file in files:
        if not os.path.exists(file):
            print(f"Skipping {file}: File not found in the current directory.")
            continue
            
        print(f"Processing {file}...")
        img, _ = load_and_preprocess(file)
        
        # Save original test image directly
        base_name = file.split('.')[0]
        plt.imsave(f"{output_dir}/{base_name}_original.png", img, cmap='gray')
        
        # Execute Task 1: Peakiness
        res_peakiness, t_peak = threshold_peakiness(img)
        print(f"  Peakiness Threshold selected: {t_peak}")
        plt.imsave(f"{output_dir}/{base_name}_peakiness.png", res_peakiness, cmap='gray')
        
        # Execute Task 2: Iterative
        res_iterative, t_iter = threshold_iterative(img)
        print(f"  Iterative Threshold converged at: {t_iter:.2f}")
        plt.imsave(f"{output_dir}/{base_name}_iterative.png", res_iterative, cmap='gray')
        
        # Execute Task 3: Dual Thresholding with Region Growing
        res_dual, th, tl = threshold_dual_region_growing(img)
        print(f"  Dual Thresholds selected -> High: {th}, Low: {tl}")
        plt.imsave(f"{output_dir}/{base_name}_dual_region.png", res_dual, cmap='gray')

    print(f"Processing complete. Images saved to the '{output_dir}' directory.")