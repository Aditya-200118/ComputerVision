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

def load_and_preprocess(path):
    """Loads the raw 512x512 byte image."""
    try:
        with open(path, "rb") as f:
            _ = f.read(512)
            img_data = np.fromfile(f, dtype=np.uint8)
        return img_data.reshape((512, 512)).astype(float) # Cast to float for math
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()

def calculate_dynamic_stats(image):
    """
    Computes image statistics on the fly to drive adaptive thresholding.
    Ensures robustness across any hidden test images.
    """
    mean_val = np.mean(image)
    std_val = np.std(image)

    # 1. Calculate Skewness for Dynamic Multiplier
    if std_val > 0:
        skewness = np.mean(((image - mean_val) / std_val) ** 3)
    else:
        skewness = 0

    s_norm = min(1.0, abs(skewness))
    dynamic_multiplier = max(0.3, 1.0 - s_norm)

    # 2. Iterative Bimodal Thresholding (Foreground vs Background separation)
    T_temp = mean_val
    for _ in range(50): # Max iterations to prevent infinite loops
        G1 = image[image > T_temp]
        G2 = image[image <= T_temp]
        mu1 = np.mean(G1) if len(G1) > 0 else 0
        mu2 = np.mean(G2) if len(G2) > 0 else 0

        T_new = (mu1 + mu2) / 2.0
        if abs(T_new - T_temp) < 1.0:
            break
        T_temp = T_new

    return mean_val, std_val, T_temp, dynamic_multiplier

def generate_log_kernel(sigma):
    """Generates a dynamic LoG kernel bounded by [-4\sigma, +4\sigma]."""
    radius = int(np.ceil(4 * sigma))
    
    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(x, y)
    
    term1 = (X**2 + Y**2) / (2 * sigma**2)
    kernel = - (1.0 / (np.pi * sigma**4)) * (1.0 - term1) * np.exp(-term1)
    
    kernel = kernel - np.mean(kernel) # Zero sum correction
    return kernel

def convolve2d_fft(image, kernel):
    """2D convolution using Fast Fourier Transforms."""
    shape = (image.shape[0] + kernel.shape[0] - 1, image.shape[1] + kernel.shape[1] - 1)
    
    fft_img = np.fft.fft2(image, s=shape)
    fft_ker = np.fft.fft2(kernel, s=shape)
    
    result = np.real(np.fft.ifft2(fft_img * fft_ker))
    
    offset_y = kernel.shape[0] // 2
    offset_x = kernel.shape[1] // 2
    
    return result[offset_y:offset_y+image.shape[0], offset_x:offset_x+image.shape[1]]

def dilate_mask(mask, iterations=2):
    """Dilates a binary mask to create a tracking window."""
    dilated = np.copy(mask)
    shifts = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    
    for _ in range(iterations):
        current_iter_mask = np.copy(dilated)
        for dy, dx in shifts:
            current_iter_mask |= np.roll(np.roll(dilated, dy, axis=0), dx, axis=1)
        dilated = current_iter_mask
        
    return dilated

def find_zero_crossings(log_img, search_mask=None, variance_thresh_map=0.0):
    """
    Detects precise zero-crossings using an adaptive, spatially-aware 2D threshold map.
    """
    right = np.roll(log_img, -1, axis=1)
    bottom = np.roll(log_img, -1, axis=0)
    br = np.roll(np.roll(log_img, -1, axis=0), -1, axis=1)
    bl = np.roll(np.roll(log_img, -1, axis=0), 1, axis=1)

    # Note: variance_thresh_map is now a 2D array, so thresholds adapt per-pixel
    zc_h = (log_img * right < 0) & (np.abs(log_img - right) > variance_thresh_map)
    zc_v = (log_img * bottom < 0) & (np.abs(log_img - bottom) > variance_thresh_map)
    zc_d1 = (log_img * br < 0) & (np.abs(log_img - br) > variance_thresh_map)
    zc_d2 = (log_img * bl < 0) & (np.abs(log_img - bl) > variance_thresh_map)

    crossings = zc_h | zc_v | zc_d1 | zc_d2
    
    if search_mask is not None:
        crossings = crossings & search_mask
        
    crossings[:2, :] = False; crossings[-2:, :] = False
    crossings[:, :2] = False; crossings[:, -2:] = False
    
    return crossings

def run_edge_focusing(image):
    """Executes multi-scale edge focusing with adaptive statistics."""
    
    # 1. On-the-fly Image Profiling
    mean_val, std_val, T_iterative, dynamic_multiplier = calculate_dynamic_stats(image)
    
    # 2. Contrast Normalization 
    norm_image = (image - mean_val) / (std_val + 1e-5)
    
    # 3. Create Bimodal Spatial Threshold Map
    # Slightly relaxed from 1.5 to 1.2 so we don't kill the dark edges of the monitor
    spatial_penalty = np.ones_like(image, dtype=float)
    spatial_penalty[image <= T_iterative] = 1.2 
    spatial_penalty[image > T_iterative] = 0.8

    # Constraints Check: Start at 5.0, End at 1.0, Step 0.5
    sigmas_to_process = np.arange(5.0, 0.5, -0.5)
    sigmas_to_save = [5.0, 4.0, 3.0, 2.0, 1.0]
    
    saved_edge_maps = {}
    current_search_mask = None
    
    for sigma in sigmas_to_process:
        print(f"  -> Processing scale \u03C3 = {sigma:.1f}")
        
        # Constraints Check: Kernel bounds are correctly [-4σ, +4σ]
        kernel = generate_log_kernel(sigma)
        log_img = convolve2d_fft(norm_image, kernel)
        
        # 4. FIX: Statistical Normalization instead of Max Normalization
        # By standardizing the LoG response, our base_thresh means "fractions of a standard deviation"
        log_std = np.std(log_img)
        if log_std > 0:
            log_img = log_img / log_std
            
        # A threshold of ~0.3 standard deviations produces solid, connected edges
        base_thresh = 0.3 * dynamic_multiplier
        
        adaptive_thresh_map = base_thresh * spatial_penalty
        
        edges = find_zero_crossings(log_img, search_mask=current_search_mask, variance_thresh_map=adaptive_thresh_map)
        
        if sigma in sigmas_to_save:
            saved_edge_maps[sigma] = edges
            
        # 5. FIX: Increased dilation from 2 to 3 to prevent the mask from choking valid shifting edges
        current_search_mask = dilate_mask(edges, iterations=3)
        
    return saved_edge_maps

def plot_and_save_results(image, edge_maps, filename, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f"Original: {filename}")
    axes[0].axis('off')
    
    plot_idx = 1
    for sigma in [5.0, 4.0, 3.0, 2.0, 1.0]:
        axes[plot_idx].imshow(edge_maps[sigma], cmap='gray')
        axes[plot_idx].set_title(f"Edge Map (\u03C3 = {sigma})")
        axes[plot_idx].axis('off')
        plot_idx += 1
        
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{filename.split('.')[0]}_edge_focusing_ver3.pdf")
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  [+] Saved {out_path}\n")

if __name__ == "__main__":
    output_dir = "assignment4_figures_ver4"
    os.makedirs(output_dir, exist_ok=True)
    
    files = ["test1.img", "test2.img", "test3.img"]

    for file in files:
        if not os.path.exists(file):
            continue

        print(f"Starting Adaptive Edge Focusing on: {file}")
        img = load_and_preprocess(file) 
        edge_maps = run_edge_focusing(img)
        plot_and_save_results(img, edge_maps, file, output_dir)
        print("-" * 40)