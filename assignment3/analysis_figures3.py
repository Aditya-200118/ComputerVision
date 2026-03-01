import numpy as np
from matplotlib.pyplot import subplots # Added to fix potential matplotlib namespace issue internally
import matplotlib.pyplot as plt
import os
import math
from matplotlib.backends.backend_pdf import PdfPages

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

def optimized_iterative_threshold(flat_img, max_iter=100, tol=0.01):
    """
    Multi-start iterative threshold optimizer.
    Returns threshold minimizing intra-class variance.
    """
    best_T = None
    best_intra_var = float('inf')

    # Multiple starting guesses across intensity range
    initial_guesses = np.linspace(np.min(flat_img), np.max(flat_img), 7)

    for T_start in initial_guesses:
        T = T_start

        for _ in range(max_iter):
            G1 = flat_img[flat_img > T]
            G2 = flat_img[flat_img <= T]

            if len(G1) == 0 or len(G2) == 0:
                break

            mu1 = np.mean(G1)
            mu2 = np.mean(G2)
            T_new = (mu1 + mu2) / 2.0

            if abs(T_new - T) < tol:
                break

            T = T_new

        # Compute intra-class variance
        if len(G1) > 0 and len(G2) > 0:
            var1 = np.var(G1)
            var2 = np.var(G2)
            w1 = len(G1) / len(flat_img)
            w2 = len(G2) / len(flat_img)

            intra_var = w1 * var1 + w2 * var2

            if intra_var < best_intra_var:
                best_intra_var = intra_var
                best_T = T

    return best_T, best_intra_var

def calculate_gradients(img):
    """Calculates spatial pixel-to-pixel differences (edges)."""
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    # Central difference for interior pixels
    gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
    gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0
    
    magnitude = np.sqrt(gx**2 + gy**2)
    return magnitude

def save_pdf_table(output_dir, filename, stats_data):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('tight')
    ax.axis('off')
    ax.set_title(f"DEEP PROFILE: {filename}", fontweight='bold', pad=20)
    
    table = ax.table(cellText=stats_data, colLabels=["Metric", "Value"], loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.5)
    
    out_path = os.path.join(output_dir, f"{filename.split('.')[0]}_stats_table.pdf")
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()

def extract_image_profile(img, filename, output_dir):
    flat_img = img.ravel()
    
    # 1. Core Statistics
    mean_val = np.mean(flat_img)
    median_val = np.median(flat_img)
    std_val = np.std(flat_img)
    min_val, max_val = np.min(flat_img), np.max(flat_img)
    
    # --- NEW ADVANCED METRICS ---
    # Skewness (3rd Moment) and Kurtosis (4th Moment)
    if std_val > 0:
        skewness = np.mean(((flat_img - mean_val) / std_val) ** 3)
        kurtosis = np.mean(((flat_img - mean_val) / std_val) ** 4) - 3 # Fisher's definition (excess kurtosis)
    else:
        skewness = 0
        kurtosis = 0

    # Intra-Class Variance Heuristic Prep: 
    # Quickly find iterative T_H to isolate the foreground class, then get its StdDev
    # T_iter = mean_val
    # for _ in range(50): # Cap iterations just for profiling safety
    #     G1 = flat_img[flat_img > T_iter]
    #     G2 = flat_img[flat_img <= T_iter]
    #     mu1 = np.mean(G1) if len(G1) > 0 else 0
    #     mu2 = np.mean(G2) if len(G2) > 0 else 0
    #     T_new = (mu1 + mu2) / 2.0
    #     if abs(T_new - T_iter) < 1.0:
    #         break
    #     T_iter = T_new
        
    # T_H_profile = T_iter
    # fg_pixels = flat_img[flat_img >= T_H_profile]
    # fg_std = np.std(fg_pixels) if len(fg_pixels) > 0 else 0
    # ----------------------------

    T_H_profile, intra_var = optimized_iterative_threshold(flat_img)

    fg_pixels = flat_img[flat_img >= T_H_profile]
    fg_std = np.std(fg_pixels) if len(fg_pixels) > 0 else 0

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
    # CHANGED: Increased height to 15 to accommodate 3 rows
    # CHANGED: Made grid 3 rows instead of 2
    
    # Plot A: Original Image
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax1.set_title(f"Original: {filename}")
    ax1.axis('off')
    plt.savefig(os.path.join(output_dir, f"{filename.split('.')[0]}_original.pdf"), format='pdf', bbox_inches='tight')
    plt.close(fig1)
    
    # Plot B: Histogram and CDF
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    # NEW: Plot raw histogram first so it sits behind the smoothed one
    ax2.bar(range(256), hist, color='gray', alpha=0.5, width=1.0, label='Raw Hist')
    ax2.plot(smoothed_hist, color='black', label='Smoothed Hist')
    
    ax2_cdf = ax2.twinx()
    ax2_cdf.plot(cdf, color='red', linestyle='--', alpha=0.7, label='CDF')
    
    # NEW: Mean and Median lines
    ax2.axvline(mean_val, color='green', linestyle='-', linewidth=1.5, label=f'Mean ({mean_val:.1f})')
    ax2.axvline(median_val, color='magenta', linestyle='-.', linewidth=1.5, label=f'Median ({median_val:.1f})')
    
    ax2.axvline(p25, color='blue', linestyle=':', label=f'25th % ({int(p25)})')
    ax2.axvline(p75, color='purple', linestyle=':', label=f'75th % ({int(p75)})')
    ax2.axvline(p95, color='orange', linestyle=':', label=f'95th % ({int(p95)})')
    
    ax2.set_title(f"Intensity Distribution & CDF")
    ax2.set_xlim([-5, 255])
    ax2.set_xlabel("Grayscale Intensity")
    ax2.set_ylabel("Pixel Frequency")
    ax2_cdf.set_ylabel("Cumulative Probability", color='red')
    ax2.legend(loc='upper left')
    plt.savefig(os.path.join(output_dir, f"{filename.split('.')[0]}_cdf_histogram.pdf"), format='pdf', bbox_inches='tight')
    plt.close(fig2)
    
    # Plot C: Gradient Magnitude (Edges Strength)
    fig3, ax3 = plt.subplots(figsize=(7, 6))
    im3 = ax3.imshow(grad_mag, cmap='hot')
    ax3.set_title(f"Pixel Gradient Magnitude (Edges)\nMean Gradient: {mean_grad:.2f}")
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(output_dir, f"{filename.split('.')[0]}_gradient_magnitude.pdf"), format='pdf', bbox_inches='tight')
    plt.close(fig3)
    
    # Plot D: Gradient Histogram
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.hist(grad_mag.ravel(), bins=50, range=(1, 50), color='crimson', alpha=0.7)
    ax4.set_title(f"Gradient Strength Distribution")
    ax4.set_xlabel("Gradient Magnitude")
    ax4.set_ylabel("Frequency")
    ax4.set_yscale('log')
    plt.savefig(os.path.join(output_dir, f"{filename.split('.')[0]}_gradient_histogram.pdf"), format='pdf', bbox_inches='tight')
    plt.close(fig4)
    
    # --- NEW PLOT E: Prominence Ratio Analysis ---
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    ax5.plot(smoothed_hist, color='black', linewidth=1.5)
    ax5.set_title("Prominence Ratio Threshold Analysis")
    ax5.set_xlabel("Grayscale Intensity")
    ax5.set_ylabel("Pixel Frequency")
    ax5.set_xlim([0, 255])
    
    max_h = np.max(smoothed_hist)
    
    # Define the ratios we want to test visually
    test_ratios = [0.02, 0.05, 0.10]
    colors = ['green', 'orange', 'red']
    
    for r, c in zip(test_ratios, colors):
        thresh_val = max_h * r
        ax5.axhline(thresh_val, color=c, linestyle='--', alpha=0.7, 
                    label=f'{int(r*100)}% Threshold ({int(thresh_val)} px)')
        
        # Find peaks that pass this specific threshold
        valid_peaks_x = []
        valid_peaks_y = []
        for i in range(1, 255):
            if smoothed_hist[i] > smoothed_hist[i-1] and smoothed_hist[i] > smoothed_hist[i+1]:
                if smoothed_hist[i] > thresh_val:
                    valid_peaks_x.append(i)
                    valid_peaks_y.append(smoothed_hist[i])
        
        # Plot dots on the peaks that survive this ratio
        if valid_peaks_x:
            ax5.scatter(valid_peaks_x, valid_peaks_y, color=c, zorder=5, s=50)

    ax5.legend(loc='upper right')

    # Save
    plt.savefig(os.path.join(output_dir, f"{filename.split('.')[0]}_prominence_analysis.pdf"), format='pdf', bbox_inches='tight')
    plt.close(fig5)
    
    # Prepare Table Data
    stats_data = [
        ["Mean Intensity", f"{mean_val:.2f}"],
        ["Median Intensity", f"{median_val:.2f}"],
        ["StdDev", f"{std_val:.2f}"],
        ["Skewness", f"{skewness:.3f}"],
        ["Kurtosis", f"{kurtosis:.3f}"],
        ["T_H (Iterative)", f"{T_H_profile:.1f}"],
        ["Min Intra-Class Variance", f"{intra_var:.2f}"],
        ["Foreground StdDev", f"{fg_std:.2f}"],
        ["Min / Max", f"{int(min_val)} / {int(max_val)}"],
        ["Entropy", f"{entropy:.3f} bits/pixel"],
        ["Percentiles (10/25/75/90/95)", f"{int(p10)}, {int(p25)}, {int(p75)}, {int(p90)}, {int(p95)}"],
        ["Top Peaks", f"{[int(p[0]) for p in peaks[:3]]}"],
        ["Mean Gradient", f"{mean_grad:.2f}"],
    ]
    save_pdf_table(output_dir, filename, stats_data)

if __name__ == "__main__":
    output_dir = "assignment3_profiling_ver3"
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
            
    print(f"Profiling complete. Individual PDFs saved to '{output_dir}'.")