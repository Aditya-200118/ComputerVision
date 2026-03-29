import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "figure.dpi": 300,  # Adjusted slightly for faster plotting, feel free to restore to 600
    }
)

def load_and_preprocess(path):
    """Loads the raw 512x512 byte image."""
    try:
        with open(path, "rb") as f:
            _ = f.read(512)
            img_data = np.fromfile(f, dtype=np.uint8)
        return img_data.reshape((512, 512))
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()

# def generate_log_kernel(sigma):
#     """
#     Generates a Laplacian of Gaussian (LoG) kernel.
#     The assignment specifies a sampling range of [-46, +46].
#     """
#     grid_size = 46
#     x = np.arange(-grid_size, grid_size + 1)
#     y = np.arange(-grid_size, grid_size + 1)
#     X, Y = np.meshgrid(x, y)
    
#     # Mathematical formula for the 2D LoG
#     term1 = (X**2 + Y**2) / (2 * sigma**2)
#     kernel = - (1.0 / (np.pi * sigma**4)) * (1.0 - term1) * np.exp(-term1)
    
#     # Ensure the kernel sums to exactly zero to avoid DC bias on uniform regions
#     kernel = kernel - np.mean(kernel)
#     return kernel

def generate_log_kernel(sigma):
    """
    Generates a dynamic Laplacian of Gaussian (LoG) kernel.
    Follows the assignment rule of [-4\sigma, +4\sigma].
    """
    radius = int(np.ceil(4 * sigma))
    
    # This guarantees the kernel is always an odd size (2 * radius + 1)
    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(x, y)
    
    term1 = (X**2 + Y**2) / (2 * sigma**2)
    kernel = - (1.0 / (np.pi * sigma**4)) * (1.0 - term1) * np.exp(-term1)
    
    # Ensure kernel sums to exactly zero
    kernel = kernel - np.mean(kernel)
    return kernel

def convolve2d_fft(image, kernel):
    """
    Performs 2D convolution from scratch using Fast Fourier Transforms (FFT).
    This is drastically faster than nested for-loops and avoids OpenCV.
    """
    # Pad to avoid circular convolution artifacts
    shape = (image.shape[0] + kernel.shape[0] - 1, image.shape[1] + kernel.shape[1] - 1)
    
    fft_img = np.fft.fft2(image, s=shape)
    fft_ker = np.fft.fft2(kernel, s=shape)
    
    # Convolution in spatial domain is multiplication in frequency domain
    result = np.fft.ifft2(fft_img * fft_ker)
    result = np.real(result)
    
    # Crop back to the original image size ("same" convolution mode)
    offset_y = kernel.shape[0] // 2
    offset_x = kernel.shape[1] // 2
    
    return result[offset_y:offset_y+image.shape[0], offset_x:offset_x+image.shape[1]]

# def dilate_mask(mask):
#     """
#     Dilates a binary mask by 1 pixel in all 8 directions.
#     This creates the "search window" for the next scale in edge focusing.
#     """
#     dilated = np.copy(mask)
#     # Using np.roll to simulate shifting a 3x3 window over the boolean array
#     shifts = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
#     for dy, dx in shifts:
#         dilated |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
#     return dilated

def dilate_mask(mask, iterations=2):
    """
    Dilates a binary mask. We use 2 iterations (a 2-pixel radius) 
    to ensure we don't lose fast-shifting edges during the step down.
    """
    dilated = np.copy(mask)
    shifts = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    
    for _ in range(iterations):
        current_iter_mask = np.copy(dilated)
        for dy, dx in shifts:
            current_iter_mask |= np.roll(np.roll(dilated, dy, axis=0), dx, axis=1)
        dilated = current_iter_mask
        
    return dilated

# def find_zero_crossings(log_img, search_mask=None, variance_thresh=0.01):
#     """
#     Detects zero-crossings in the LoG image.
#     A zero-crossing occurs when a positive pixel is adjacent to a negative pixel.
#     """
#     # Find the local max and min in a 3x3 neighborhood around every pixel
#     shifts = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
#     max_val = np.copy(log_img)
#     min_val = np.copy(log_img)
    
#     for dy, dx in shifts:
#         shifted = np.roll(np.roll(log_img, dy, axis=0), dx, axis=1)
#         max_val = np.maximum(max_val, shifted)
#         min_val = np.minimum(min_val, shifted)
        
#     # An edge crosses zero if the neighborhood max is positive and min is negative
#     crossings = (max_val > 0) & (min_val < 0)
    
#     # Filter out fake zero-crossings in flat, uniform regions (noise floor)
#     # by ensuring the difference across the zero-crossing is steep enough.
#     local_gradient = max_val - min_val
#     crossings = crossings & (local_gradient > variance_thresh)
    
#     # If edge focusing is active, only allow edges within the provided spatial mask
#     if search_mask is not None:
#         crossings = crossings & search_mask
        
#     # Clean up the boundary artifacts caused by np.roll wrapping around
#     crossings[:2, :] = False
#     crossings[-2:, :] = False
#     crossings[:, :2] = False
#     crossings[:, -2:] = False
    
#     return crossings

def find_zero_crossings(log_img, search_mask=None, variance_thresh=0.0):
    """
    Detects precise zero-crossings by checking explicit adjacent pairs.
    """
    # Shift arrays to check neighbors: Right, Bottom, Bottom-Right, Bottom-Left
    right = np.roll(log_img, -1, axis=1)
    bottom = np.roll(log_img, -1, axis=0)
    br = np.roll(np.roll(log_img, -1, axis=0), -1, axis=1)
    bl = np.roll(np.roll(log_img, -1, axis=0), 1, axis=1)

    # Check for sign changes (if multiplying them is negative, signs differ)
    # We also verify that the absolute difference crosses the threshold
    zc_h = (log_img * right < 0) & (np.abs(log_img - right) > variance_thresh)
    zc_v = (log_img * bottom < 0) & (np.abs(log_img - bottom) > variance_thresh)
    zc_d1 = (log_img * br < 0) & (np.abs(log_img - br) > variance_thresh)
    zc_d2 = (log_img * bl < 0) & (np.abs(log_img - bl) > variance_thresh)

    crossings = zc_h | zc_v | zc_d1 | zc_d2
    
    if search_mask is not None:
        crossings = crossings & search_mask
        
    # Clean up edge artifacts
    crossings[:2, :] = False; crossings[-2:, :] = False
    crossings[:, :2] = False; crossings[:, -2:] = False
    
    return crossings

def run_edge_focusing(image):
    """
    Executes the multi-scale edge focusing algorithm.
    Starts at sigma=5.0 and tracks edges down to sigma=1.0.
    """
    sigmas_to_process = np.arange(5.0, 0.5, -0.5)
    sigmas_to_save = [5.0, 4.0, 3.0, 2.0, 1.0]
    
    saved_edge_maps = {}
    current_search_mask = None
    
    for sigma in sigmas_to_process:
        print(f"  -> Processing scale \u03C3 = {sigma:.1f}")
        
        # 1. Generate Kernel & Convolve
        kernel = generate_log_kernel(sigma)
        log_img = convolve2d_fft(image, kernel)
        
        # 2. Dynamic Noise Threshold (scales with the blurring effect)
        # dynamic_thresh = 0.05 * np.max(np.abs(log_img))
        dynamic_thresh = 0.8 * np.mean(np.abs(log_img))
        
        # 3. Find Zero Crossings
        edges = find_zero_crossings(log_img, search_mask=current_search_mask, variance_thresh=dynamic_thresh)
        
        # 4. Save if it's one of the required integer scales
        if sigma in sigmas_to_save:
            saved_edge_maps[sigma] = edges
            
        # 5. Update Search Mask for the next iteration
        # Dilate the current edges to create a tight tracking window for the next scale
        current_search_mask = dilate_mask(edges)
        
    return saved_edge_maps

def plot_and_save_results(image, edge_maps, filename, output_dir):
    """Generates the hardcopy visual requirements."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Plot Original
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f"Original: {filename}")
    axes[0].axis('off')
    
    # Plot Scales
    plot_idx = 1
    for sigma in [5.0, 4.0, 3.0, 2.0, 1.0]:
        axes[plot_idx].imshow(edge_maps[sigma], cmap='gray')
        axes[plot_idx].set_title(f"Edge Map (\u03C3 = {sigma})")
        axes[plot_idx].axis('off')
        plot_idx += 1
        
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{filename.split('.')[0]}_edge_focusing.pdf")
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  [+] Saved {out_path}\n")

if __name__ == "__main__":
    output_dir = "assignment4_figures_ver2"
    os.makedirs(output_dir, exist_ok=True)
    
    files = ["test1.img", "test2.img", "test3.img"]

    for file in files:
        if not os.path.exists(file):
            print(f"Warning: {file} not found. Skipping.")
            continue

        print(f"Starting Edge Focusing on: {file}")
        img = load_and_preprocess(file) 
        
        # Run the multi-scale algorithm
        edge_maps = run_edge_focusing(img)
        
        # Save the hardcopies
        plot_and_save_results(img, edge_maps, file, output_dir)
        
    print("Assignment 4 processing complete! All PDFs saved in 'assignment4_figures_ver2/' directory.")