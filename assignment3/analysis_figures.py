import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "figure.dpi": 300,
    }
)

def load_raw_img(path):
    try:
        with open(path, "rb") as f:
            f.seek(512)
            img_data = np.fromfile(f, dtype=np.uint8)
        return img_data.reshape((512, 512))
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def analyze_and_plot(img, filename, output_dir):
    # Calculate statistics
    mean_val = np.mean(img)
    median_val = np.median(img)
    std_val = np.std(img)
    
    # Calculate Histogram
    hist = np.zeros(256, dtype=int)
    for val in img.ravel():
        hist[val] += 1
        
    # Apply a slight moving average to find the actual structural peaks
    window = 5
    smoothed_hist = np.zeros(256, dtype=float)
    for i in range(256):
        start = max(0, i - window)
        end = min(255, i + window)
        smoothed_hist[i] = np.mean(hist[start:end+1])

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Image Plot
    ax1.imshow(img, cmap='gray')
    ax1.set_title(f"Original: {filename}")
    ax1.axis('off')
    
    # Histogram Plot
    ax2.plot(smoothed_hist, color='black', label='Smoothed Histogram')
    ax2.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.1f}')
    ax2.axvline(median_val, color='blue', linestyle='dotted', linewidth=1.5, label=f'Median: {median_val:.1f}')
    
    # Fill standard deviation range
    ax2.axvspan(mean_val - std_val, mean_val + std_val, color='red', alpha=0.1, label=f'1 Std Dev ($\sigma$={std_val:.1f})')
    
    ax2.set_title(f"Histogram Analysis: {filename}")
    ax2.set_xlim([0, 255])
    ax2.set_xlabel("Grayscale Intensity")
    ax2.set_ylabel("Pixel Frequency")
    ax2.legend()
    
    # Save
    out_path = os.path.join(output_dir, f"{filename.split('.')[0]}_analysis.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    
    print(f"--- Analysis for {filename} ---")
    print(f"  Mean:   {mean_val:.2f}")
    print(f"  Median: {median_val:.2f}")
    print(f"  StdDev: {std_val:.2f}")
    print(f"  Min: {np.min(img)}, Max: {np.max(img)}\n")

if __name__ == "__main__":
    output_dir = "analysis_figures"
    os.makedirs(output_dir, exist_ok=True)
    
    files = ["test1.img", "test2.img", "test3.img"]
    
    for file in files:
        if os.path.exists(file):
            img = load_raw_img(file)
            if img is not None:
                analyze_and_plot(img, file, output_dir)
        else:
            print(f"File {file} not found.")
            
    print(f"Analysis complete. Check the '{output_dir}' directory for histogram plots.")