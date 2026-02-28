import numpy as np
import cv2
import os

# Configuration for output
OUTPUT_FOLDER = "assignment3_figures"

def load_img(path):
    try:
        with open(path, "rb") as f:
            f.seek(512) # Skip header
            img_data = np.fromfile(f, dtype=np.uint8)
        return img_data.reshape((512, 512))
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_histogram_manual(img):
    hist = np.zeros(256, dtype=int)
    for pixel in img.ravel():
        hist[pixel] += 1
    return hist

# --- 1. Peakiness Detection ---
def peakiness_threshold(img):
    # Use OpenCV for smoothing to handle noise in the histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    smoothed = cv2.GaussianBlur(hist, (1, 5), 0).flatten()
    
    best_score = -1
    best_t = 128
    min_peak_dist = 40
    
    # Evaluation Criterion: Maximize (H[i] * H[j]) / H[k]
    # where i, j are peaks and k is the deepest valley between them.
    for i in range(len(smoothed)):
        for j in range(i + min_peak_dist, len(smoothed)):
            valley_segment = smoothed[i:j]
            if len(valley_segment) == 0: continue
            
            k_idx = i + np.argmin(valley_segment)
            h_k = smoothed[k_idx] if smoothed[k_idx] > 0 else 1
            
            score = (smoothed[i] * smoothed[j]) / h_k
            if score > best_score:
                best_score = score
                best_t = k_idx
                
    binary = (img > best_t).astype(np.uint8) * 255
    return binary, int(best_t)

# --- 2. Iterative Thresholding ---
def iterative_threshold(img):
    t = np.mean(img)
    old_t = 0
    
    while abs(t - old_t) > 0.5:
        old_t = t
        m1 = np.mean(img[img > t]) if np.any(img > t) else 0
        m2 = np.mean(img[img <= t]) if np.any(img <= t) else 0
        t = (m1 + m2) / 2
        
    binary = (img > t).astype(np.uint8) * 255
    return binary, int(t)

# --- 3. Dual Thresholding with Region Growing ---
# Heuristic: T1 (Strong) is the highest peak. 
# T2 (Weak) is the point where the histogram drops to 20% of the peak height 
# or a fixed 20-unit distance, allowing the region to expand into edges.


def dual_threshold_region_growing(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    t1 = np.argmax(hist)
    
    # Heuristic for T2: Find first intensity below T1 where hist < 0.2 * max
    t2 = t1
    threshold_limit = 0.2 * hist[t1]
    for val in range(int(t1), 0, -1):
        if hist[val] < threshold_limit:
            t2 = val
            break
    if t2 == t1: t2 = max(0, t1 - 30)

    rows, cols = img.shape
    output = np.zeros_like(img, dtype=np.uint8)
    visited = np.zeros_like(img, dtype=bool)
    
    # Seeds: strong threshold
    seeds = np.argwhere(img >= t1)
    stack = [tuple(s) for s in seeds]
    
    for r, c in stack:
        output[r, c] = 255
        visited[r, c] = True

    # 8-connectivity search
    while stack:
        r, c = stack.pop()
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                    if img[nr, nc] >= t2:
                        visited[nr, nc] = True
                        output[nr, nc] = 255
                        stack.append((nr, nc))
                        
    return output, (int(t1), int(t2))

def run_assignment():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, OUTPUT_FOLDER)
    os.makedirs(output_dir, exist_ok=True)
    
    image_names = ["test1.img", "test2.img", "test3.img"]
    
    for name in image_names:
        path = os.path.join(script_dir, name)
        img = load_img(path)
        if img is None: continue
        
        base_name = name.split('.')[0]
        
        # 1. Peakiness
        res1, val1 = peakiness_threshold(img)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_peakiness_T{val1}.png"), res1)
        
        # 2. Iterative
        res2, val2 = iterative_threshold(img)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_iterative_T{val2}.png"), res2)
        
        # 3. Dual/Region Growing
        res3, val3 = dual_threshold_region_growing(img)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_dualRG_T{val3[0]}_{val3[1]}.png"), res3)
        
        print(f"Processed {name}: Peakiness T={val1}, Iterative T={val2}, Dual T1,T2={val3}")

if __name__ == "__main__":
    run_assignment()