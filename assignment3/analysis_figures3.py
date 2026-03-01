import numpy as np
from matplotlib.pyplot import subplots # Added to fix potential matplotlib namespace issue internally
import matplotlib.pyplot as plt
import os
import math
from matplotlib.backends.backend_pdf import PdfPages

# Maintain the exact requested styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 600,
    "axes.grid": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
})

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

import matplotlib.pyplot as plt
import os

# Instead of save_pdf_table, add this to your Python script:
def print_latex_table(filename, stats_data):
    print(f"\n% LaTeX Table for {filename}")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{ll}")
    print("\\toprule")
    for row in stats_data:
        print(f"{row[0]} & {row[1]} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

def save_pdf_table(output_dir, filename, stats_data):
    # Narrower figure to prevent horizontal stretching
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    
    # Create the table
    table = ax.table(
        cellText=stats_data, 
        colLabels=["Metric", "Value"], 
        loc='center', 
        cellLoc='left',
        edges='horizontal' # Removes vertical lines for a cleaner academic look
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9) # Standard ICLR body text size
    
    # Style the header and lines
    for (row, col), cell in table.get_celld().items():
        # Make the header bold and add a thicker line
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_linewidth(1.5) 
        else:
            cell.set_linewidth(0.5) # Thin interior lines
            
        # Optional: Add padding to cells
        cell.set_edgecolor('#333333') # Dark gray lines instead of harsh black
        
    table.scale(1.0, 1.4) # Tighten row height for a compact profile
    
    out_path = os.path.join(output_dir, f"{filename.split('.')[0]}_stats_table.pdf")
    # Use 'tight' with a small pad to ensure no text is clipped
    plt.savefig(out_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
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
    hist = np.bincount(flat_img.astype(int), minlength=256)
        
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
        
    # Replace all fig creation lines with a fixed size
    FIG_WIDTH = 6
    FIG_HEIGHT = 6  # Make all plots square for alignment

    # Plot A: Original Image
    fig1, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax1.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax1.axis('off')
    plt.savefig(os.path.join(output_dir, f"{filename.split('.')[0]}_original.pdf"), format='pdf', bbox_inches='tight')
    plt.close(fig1)

    # Plot B: Histogram and CDF
    fig2, ax2 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    # 1. Histogram: Use a sophisticated Charcoal with slight transparency
    ax2.bar(range(256), hist, color='#2c3e50', alpha=0.3, width=1.0, label='Histogram', zorder=1)
    
    # 2. Smoothed Trend: Solid Slate Blue line
    ax2.plot(smoothed_hist, color='#34495e', linewidth=1.5, label='Smoothed Distribution', zorder=2)

    # 3. CDF: Use a high-contrast Crimson or Deep Blue for the secondary axis
    ax2_cdf = ax2.twinx()
    ax2_cdf.plot(cdf, color='#c0392b', linestyle='-', linewidth=2, alpha=0.8, label='CDF', zorder=3)
    
    # 4. Statistical Markers: Coordinated palette
    # Mean (Green), Median (Deep Orange), Percentiles (Muted Blue/Gray)
    ax2.axvline(mean_val, color='#27ae60', linestyle='-', linewidth=1.2, label=f'Mean ({mean_val:.1f})')
    ax2.axvline(median_val, color='#e67e22', linestyle='--', linewidth=1.2, label=f'Median ({median_val:.1f})')
    
    # Use vertical spans or lighter lines for percentiles to reduce clutter
    ax2.axvline(p25, color='#7f8c8d', linestyle=':', alpha=0.6, label=f'25th% ({int(p25)})')
    ax2.axvline(p75, color='#7f8c8d', linestyle=':', alpha=0.6, label=f'75th% ({int(p75)})')
    ax2.axvline(p95, color='#2980b9', linestyle=':', alpha=0.9, label=f'95th% ({int(p95)})')

    # Formatting
    ax2.set_xlim([-2, 257])
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("Grayscale Intensity (0-255)")
    ax2.set_ylabel("Pixel Frequency")
    
    ax2_cdf.set_ylabel("Cumulative Probability", color='#c0392b', weight='bold')
    ax2_cdf.tick_params(axis='y', labelcolor='#c0392b')
    ax2_cdf.set_ylim([0, 1.05])

    # Clean Legend: Combine both axes into one legend box
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_cdf.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename.split('.')[0]}_cdf_histogram.pdf"), 
                format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig2)

    # Plot C: Gradient Magnitude (Edges Strength)
    fig3, ax3 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    im3 = ax3.imshow(grad_mag, cmap='hot')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(output_dir, f"{filename.split('.')[0]}_gradient_magnitude.pdf"), format='pdf', bbox_inches='tight')
    plt.close(fig3)

    # Plot D: Gradient Histogram
    fig4, ax4 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax4.hist(grad_mag.ravel(), bins=50, range=(1, 50), color='crimson', alpha=0.7)
    ax4.set_xlabel("Gradient Magnitude")
    ax4.set_ylabel("Frequency")
    ax4.set_yscale('log')
    plt.savefig(os.path.join(output_dir, f"{filename.split('.')[0]}_gradient_histogram.pdf"), format='pdf', bbox_inches='tight')
    plt.close(fig4)

    # Plot E: Height ratio Analysis
    fig5, ax5 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax5.plot(smoothed_hist, color='black', linewidth=1.5)
    ax5.set_xlabel("Grayscale Intensity")
    ax5.set_ylabel("Pixel Frequency")
    ax5.set_xlim([0, 255])

    max_h = np.max(smoothed_hist)
    test_ratios = [0.02, 0.05, 0.10]
    colors = ['green', 'orange', 'red']

    for r, c in zip(test_ratios, colors):
        thresh_val = max_h * r
        ax5.axhline(thresh_val, color=c, linestyle='--', alpha=0.7, label=f'{int(r*100)}% Threshold ({int(thresh_val)} px)')
        
        valid_peaks_x = []
        valid_peaks_y = []
        for i in range(1, 255):
            if smoothed_hist[i] > smoothed_hist[i-1] and smoothed_hist[i] > smoothed_hist[i+1]:
                if smoothed_hist[i] > thresh_val:
                    valid_peaks_x.append(i)
                    valid_peaks_y.append(smoothed_hist[i])
        
        if valid_peaks_x:
            ax5.scatter(valid_peaks_x, valid_peaks_y, color=c, zorder=5, s=50)

    ax5.legend(loc='upper right')
    plt.savefig(os.path.join(output_dir, f"{filename.split('.')[0]}_height_analysis.pdf"), format='pdf', bbox_inches='tight')
    plt.close(fig5)
    
    # Prepare Table Data
    stats_data = [
        ["Mean Intensity", f"{mean_val:.2f}"],
        ["Median Intensity", f"{median_val:.2f}"],
        ["StdDev", f"{std_val:.2f}"],
        ["Skewness", f"{skewness:.3f}"],
        ["Kurtosis", f"{kurtosis:.3f}"],
        # ["T_H (Iterative)", f"{T_H_profile:.1f}"],
        # ["Min Intra-Class Variance", f"{intra_var:.2f}"],
        ["Foreground StdDev", f"{fg_std:.2f}"],
        ["Min / Max", f"{int(min_val)} / {int(max_val)}"],
        ["Entropy", f"{entropy:.3f} bits/pixel"],
        ["Percentiles (10/25/75/90/95)", f"{int(p10)}, {int(p25)}, {int(p75)}, {int(p90)}, {int(p95)}"],
        ["Top Peaks", f"{[int(p[0]) for p in peaks[:3]]}"],
        # ["Mean Gradient", f"{mean_grad:.2f}"],
    ]
    save_pdf_table(output_dir, filename, stats_data)
    print_latex_table(filename, stats_data)
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