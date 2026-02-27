import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Maintain the exact requested styling
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "figure.dpi": 600,
    }
)

def load_raw_img(path):
    try:
        with open(path, "rb") as f:
            f.seek(512) # Skip header
            img_data = np.fromfile(f, dtype=np.uint8)
        return img_data.reshape((512, 512)).astype(float)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def calculate_gradients(img):
    """Calculates spatial pixel-to-pixel differences (edges)."""
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    # Central difference for interior pixels
    gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
    gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0
    
    magnitude = np.sqrt(gx**2 + gy**2)
    return magnitude

def extract_image_profile(img, filename, output_dir):
    flat_img = img.ravel()
    
    # 1. Core Statistics
    mean_val = np.mean(flat_img)
    median_val = np.median(flat_img)
    std_val = np.std(flat_img)
    min_val, max_val = np.min(flat_img), np.max(flat_img)
    
    # Percentiles for CDF heuristics
    p10, p25, p75, p90, p95 = np.percentile(flat_img, [10, 25, 75, 90, 95])
    
    # 2. Histogram & Entropy
    hist = np.zeros(256, dtype=int)
    for val in flat_img.astype(int):
        hist[val] += 1
        
    # Calculate Entropy (Information Content)
    probabilities = hist / flat_img.size
    entropy = -np.sum([p * math.log2(p) for p in probabilities if p > 0])
    
    # Calculate CDF
    cdf = np.cumsum(probabilities)
    
    # Smooth histogram for peak detection
    window = 5
    smoothed_hist = np.zeros(256, dtype=float)
    for i in range(256):
        start = max(0, i - window)
        end = min(255, i + window)
        smoothed_hist[i] = np.mean(hist[start:end+1])
        
    # Find Peaks
    peaks = []
    for i in range(1, 255):
        if smoothed_hist[i] > smoothed_hist[i-1] and smoothed_hist[i] > smoothed_hist[i+1]:
            if smoothed_hist[i] > (np.max(smoothed_hist) * 0.05): # Filter tiny noise spikes
                peaks.append((i, smoothed_hist[i]))
    peaks.sort(key=lambda x: x[1], reverse=True) # Sort by height
    
    # 3. Spatial Gradients
    grad_mag = calculate_gradients(img)
    mean_grad = np.mean(grad_mag)

    # 4. Plotting
    fig = plt.figure(figsize=(14, 10))
    grid = plt.GridSpec(2, 2, hspace=0.3, wspace=0.2)
    
    # Plot A: Original Image
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax1.set_title(f"Original: {filename}")
    ax1.axis('off')
    
    # Plot B: Histogram and CDF
    
    ax2 = fig.add_subplot(grid[0, 1])
    ax2.plot(smoothed_hist, color='black', label='Smoothed Hist')
    ax2_cdf = ax2.twinx()
    ax2_cdf.plot(cdf, color='red', linestyle='--', alpha=0.7, label='CDF')
    
    ax2.axvline(p25, color='blue', linestyle=':', label=f'25th % ({int(p25)})')
    ax2.axvline(p75, color='purple', linestyle=':', label=f'75th % ({int(p75)})')
    ax2.axvline(p95, color='orange', linestyle=':', label=f'95th % ({int(p95)})')
    
    ax2.set_title(f"Intensity Distribution & CDF")
    ax2.set_xlim([0, 255])
    ax2.set_xlabel("Grayscale Intensity")
    ax2.set_ylabel("Pixel Frequency")
    ax2_cdf.set_ylabel("Cumulative Probability", color='red')
    ax2.legend(loc='upper left')
    
    # Plot C: Gradient Magnitude (Edge Strength)
    ax3 = fig.add_subplot(grid[1, 0])
    im3 = ax3.imshow(grad_mag, cmap='hot')
    ax3.set_title(f"Pixel Gradient Magnitude (Edges)\nMean Gradient: {mean_grad:.2f}")
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Plot D: Gradient Histogram
    ax4 = fig.add_subplot(grid[1, 1])
    ax4.hist(grad_mag.ravel(), bins=50, range=(1, 50), color='crimson', alpha=0.7)
    ax4.set_title(f"Gradient Strength Distribution")
    ax4.set_xlabel("Gradient Magnitude")
    ax4.set_ylabel("Frequency")
    ax4.set_yscale('log') # Log scale to see the tail of strong edges

    # Save
    out_path = os.path.join(output_dir, f"{filename.split('.')[0]}_deep_profile.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    
    # Console Output (The Raw Data)
    print(f"========== DEEP PROFILE: {filename} ==========")
    print(f"Intensity Stats: Mean={mean_val:.2f}, Median={median_val:.2f}, StdDev={std_val:.2f}")
    print(f"Spread: Min={min_val}, Max={max_val}, Entropy={entropy:.3f} bits/pixel")
    print(f"Percentiles: 10%={int(p10)}, 25%={int(p25)}, 75%={int(p75)}, 90%={int(p90)}, 95%={int(p95)}")
    print(f"Top Peaks (Intensity, Freq): {[(int(p[0]), int(p[1])) for p in peaks[:3]]}")
    print(f"Edge Stats: Mean Gradient Magnitude={mean_grad:.2f}")
    print("==================================================\n")

if __name__ == "__main__":
    output_dir = "assignment3_profiling"
    os.makedirs(output_dir, exist_ok=True)
    
    files = ["test1.img", "test2.img", "test3.img"]
    
    for file in files:
        if os.path.exists(file):
            print(f"Analyzing {file}...")
            img = load_raw_img(file)
            if img is not None:
                extract_image_profile(img, file, output_dir)
        else:
            print(f"File {file} not found.")
            
    print(f"Profiling complete. High-res visuals saved to '{output_dir}'.")