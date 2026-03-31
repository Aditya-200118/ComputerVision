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
    """
    Generates a Scale-Normalized LoG kernel.
    Multiplying by sigma^2 ensures consistent peak response across scales.
    """
    radius = int(np.ceil(4 * sigma))
    
    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(x, y)
    
    # Gaussian exponent term
    arg = -(X**2 + Y**2) / (2 * sigma**2)
    
    # Scale-Normalized LoG Formula: 
    # Normalizing by (1 / (pi * sigma^2)) instead of sigma^4 
    # effectively applies the sigma^2 scaling factor.
    kernel = -(1.0 / (np.pi * sigma**2)) * (1.0 + arg) * np.exp(arg)
    
    # Ensure the kernel sums to zero to ignore constant intensity regions
    kernel -= np.mean(kernel) 
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

def generate_opencv_benchmark(image):
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_uint8, (5, 5), 1.0)
    high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    canny_edges = cv2.Canny(blurred, low_thresh, high_thresh)
    return canny_edges > 0

def run_single_scale_log(norm_image, image, sigma, T_iterative, dynamic_multiplier):
    """Applies LoG without any edge tracking/focusing."""
    spatial_penalty = np.ones_like(image, dtype=float)
    spatial_penalty[image <= T_iterative] = 1.5
    spatial_penalty[image > T_iterative] = 0.8
    
    kernel = generate_log_kernel(sigma)
    log_img = convolve2d_fft(norm_image, kernel)
    
    max_log_val = np.max(np.abs(log_img))
    if max_log_val > 0:
        log_img = log_img / max_log_val
        
    base_thresh = 0.05 * dynamic_multiplier
    adaptive_thresh_map = base_thresh * spatial_penalty
    
    # NO SEARCH MASK PASSED HERE
    return find_zero_crossings(log_img, search_mask=None, variance_thresh_map=adaptive_thresh_map)

def pratt_figure_of_merit(detected_edges, true_edges, alpha=1.0/9.0):
    """
    Calculates Pratt's Figure of Merit (PFOM).
    1.0 is perfect overlap. Closer to 0 means excessive noise or edge drift.
    """
    N_I = np.sum(true_edges)
    N_A = np.sum(detected_edges)
    
    if N_A == 0 or N_I == 0:
        return 0.0
        
    # Get euclidean distances from every pixel to the nearest TRUE edge
    inverse_true = np.logical_not(true_edges)
    distances = distance_transform_edt(inverse_true)
    
    # We only care about the penalties where we DETECTED an edge
    d_i = distances[detected_edges > 0]
    
    # Calculate PFOM Formula
    pfom = np.sum(1.0 / (1.0 + alpha * (d_i ** 2)))
    pfom /= max(N_I, N_A)
    
    return pfom

def run_edge_focusing(image):
    """Executes multi-scale edge focusing with adaptive statistics."""
    
    # 1. On-the-fly Image Profiling
    mean_val, std_val, T_iterative, dynamic_multiplier = calculate_dynamic_stats(image)
    
    # 2. Contrast Normalization (Stabilizes math across all images)
    norm_image = (image - mean_val) / (std_val + 1e-5)
    
    # 3. Create Bimodal Spatial Threshold Map
    # Background gets harsher threshold (1.5x) to kill noise. 
    # Foreground gets relaxed threshold (0.8x) to preserve fine details.
    spatial_penalty = np.ones_like(image, dtype=float)
    spatial_penalty[image <= T_iterative] = 1.5
    spatial_penalty[image > T_iterative] = 0.8

    sigmas_to_process = np.arange(5.0, 0.5, -0.5)
    sigmas_to_save = [5.0, 4.0, 3.0, 2.0, 1.0]
    
    saved_edge_maps = {}
    current_search_mask = None
    
    for sigma in sigmas_to_process:
        print(f"  -> Processing scale \u03C3 = {sigma:.1f}")
        
        kernel = generate_log_kernel(sigma)
        log_img = convolve2d_fft(norm_image, kernel)
        
        # 4. Apply Adaptive Thresholding
        # Base threshold scales based on the image's skewness
        # By normalizing the log_img to have a max absolute value of 1.0 at every scale,
        # we ensure our threshold logic remains consistent and doesn't decay to zero.
        max_log_val = np.max(np.abs(log_img))
        if max_log_val > 0:
            log_img = log_img / max_log_val
            
        # base_thresh = dynamic_multiplier * np.mean(np.abs(log_img))
        base_thresh = 0.05 * dynamic_multiplier
        
        # Multiply base by the 2D penalty map to get a pixel-perfect local threshold
        adaptive_thresh_map = base_thresh * spatial_penalty
        
        edges = find_zero_crossings(log_img, search_mask=current_search_mask, variance_thresh_map=adaptive_thresh_map)
        
        if sigma in sigmas_to_save:
            saved_edge_maps[sigma] = edges
            
        # 5. Dilate edges to create the search window for the next scale
        current_search_mask = dilate_mask(edges, iterations=2)
        
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
    files = ["test1.img", "test2.img", "test3.img"]

    for file in files:
        if not os.path.exists(file):
            continue
            
        print(f"\nEvaluating: {file}")
        img = load_and_preprocess(file)
        base_name = file.split(".")[0]
        
        # Create unique subfolder for this image
        out_dir = f"single_scale_log_{base_name}"
        os.makedirs(out_dir, exist_ok=True)
        
        mean_val, std_val, T_iterative, dyn_mult = calculate_dynamic_stats(img)
        norm_image = (img - mean_val) / (std_val + 1e-5)
        
        # 1. Generate Proxy Ground Truth
        proxy_gt = generate_opencv_benchmark(img)
        
        # 2. Generate and save Single-Scale maps for 1.0 to 5.0
        single_scale_maps = {}
        for s in [5.0, 4.0, 3.0, 2.0, 1.0]:
            ss_edges = run_single_scale_log(norm_image, img, s, T_iterative, dyn_mult)
            single_scale_maps[s] = ss_edges
            
            # Save individual image to the subfolder
            out_path = os.path.join(out_dir, f"{base_name}_sigma_{int(s)}.png")
            plt.imsave(out_path, ss_edges, cmap='gray')
            
        # 3. Get Multi-Scale target map
        multi_scale_edges = run_edge_focusing(norm_image, img, T_iterative, dyn_mult)
        
        # 4. Calculate PFOM
        pfom_single = pratt_figure_of_merit(single_scale_maps[1.0], proxy_gt)
        pfom_multi = pratt_figure_of_merit(multi_scale_edges, proxy_gt)
        
        print(f"  -> PFOM Single-Scale (Sigma=1.0): {pfom_single:.4f}")
        print(f"  -> PFOM Multi-Scale  (Sigma=1.0): {pfom_multi:.4f}")
        
        # 5. Create the Comparative Hardcopy for Requirement #3
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(single_scale_maps[1.0], cmap='gray')
        axes[0].set_title(f"Single-Scale (\u03C3=1.0)\nPFOM: {pfom_single:.4f}")
        axes[0].axis('off')
        
        axes[1].imshow(multi_scale_edges, cmap='gray')
        axes[1].set_title(f"Multi-Scale Focusing (\u03C3=1.0)\nPFOM: {pfom_multi:.4f}")
        axes[1].axis('off')
        
        axes[2].imshow(proxy_gt, cmap='gray')
        axes[2].set_title("Proxy Ground Truth (OpenCV)")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base_name}_verification_comparison.pdf"), format='pdf', bbox_inches='tight')
        plt.close()
        print(f"  [+] Saved verification PDF to {out_dir}")