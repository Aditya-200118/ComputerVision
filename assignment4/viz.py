import numpy as np
import matplotlib.pyplot as plt
import os
import time

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "figure.dpi": 150, # Slightly reduced for smoother real-time rendering
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
    """Computes image statistics on the fly to drive adaptive thresholding."""
    mean_val = np.mean(image)
    std_val = np.std(image)

    if std_val > 0:
        skewness = np.mean(((image - mean_val) / std_val) ** 3)
    else:
        skewness = 0

    s_norm = min(1.0, abs(skewness))
    dynamic_multiplier = max(0.3, 1.0 - s_norm)

    T_temp = mean_val
    for _ in range(50):
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
    
    kernel = kernel - np.mean(kernel) 
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
    """Detects precise zero-crossings using an adaptive 2D threshold map."""
    right = np.roll(log_img, -1, axis=1)
    bottom = np.roll(log_img, -1, axis=0)
    br = np.roll(np.roll(log_img, -1, axis=0), -1, axis=1)
    bl = np.roll(np.roll(log_img, -1, axis=0), 1, axis=1)

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

# def run_edge_focusing_realtime(image, filename):
#     """Executes multi-scale edge focusing with real-time matplotlib visualization."""
    
#     # --- SETUP REAL-TIME VISUALIZATION ---
#     plt.ion() # Enable interactive mode
#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#     fig.canvas.manager.set_window_title(f"Edge Focusing: {filename}")
#     fig.patch.set_facecolor('#1e1e1e') # Dark mode aesthetic
    
#     for ax in axes.ravel():
#         ax.axis('off')
#         ax.title.set_color('white')

#     # Initialize plot objects with empty/initial data to update later
#     im_orig = axes[0, 0].imshow(image, cmap='gray')
#     axes[0, 0].set_title("1. Original Image")

#     # RdBu colormap is perfect here: 0 is white, + is red, - is blue
#     im_log = axes[0, 1].imshow(np.zeros_like(image), cmap='RdBu', vmin=-1, vmax=1)
#     axes[0, 1].set_title("2. LoG Response (Red: +, Blue: -)")
    
#     im_edges = axes[1, 0].imshow(np.zeros_like(image), cmap='gray', vmin=0, vmax=1)
#     axes[1, 0].set_title("3. Detected Zero-Crossings")

#     im_mask = axes[1, 1].imshow(np.zeros_like(image), cmap='magma', vmin=0, vmax=1)
#     axes[1, 1].set_title("4. Next Iteration Search Mask")

#     plt.tight_layout()
#     plt.show(block=False)
    
#     # --- START ALGORITHM LOGIC ---
#     mean_val, std_val, T_iterative, dynamic_multiplier = calculate_dynamic_stats(image)
#     norm_image = (image - mean_val) / (std_val + 1e-5)
    
#     spatial_penalty = np.ones_like(image, dtype=float)
#     spatial_penalty[image <= T_iterative] = 1.5
#     spatial_penalty[image > T_iterative] = 0.8

#     sigmas_to_process = np.arange(5.0, 0.4, -0.5)
#     current_search_mask = None
#     saved_edge_maps = {}
    
#     for sigma in sigmas_to_process:
#         print(f" -> Processing scale σ = {sigma:.1f}")
        
#         # Update Main Title
#         fig.suptitle(f"Adaptive Edge Focusing | Current Scale: σ = {sigma:.1f}", 
#                      color='white', fontsize=16, fontweight='bold')
        
#         kernel = generate_log_kernel(sigma)
#         log_img = convolve2d_fft(norm_image, kernel)
        
#         max_log_val = np.max(np.abs(log_img))
#         if max_log_val > 0:
#             log_img = log_img / max_log_val
            
#         base_thresh = 0.05 * dynamic_multiplier
#         adaptive_thresh_map = base_thresh * spatial_penalty
        
#         edges = find_zero_crossings(log_img, search_mask=current_search_mask, variance_thresh_map=adaptive_thresh_map)
        
#         if sigma in [5.0, 4.0, 3.0, 2.0, 1.0]:
#             saved_edge_maps[sigma] = edges
            
#         # Dilate for next iteration
#         next_search_mask = dilate_mask(edges, iterations=2)
        
#         # --- UPDATE VISUALIZATION REAL-TIME ---
#         im_log.set_data(log_img)
#         im_edges.set_data(edges)
        
#         if current_search_mask is not None:
#             im_mask.set_data(current_search_mask)
#         else:
#             # First iteration has no mask, show full frame
#             im_mask.set_data(np.ones_like(image)) 

#         # Force UI update
#         fig.canvas.draw()
#         fig.canvas.flush_events()
        
#         # Pause to let the user visually digest the frame (0.8 seconds)
#         time.sleep(0.8) 
        
#         # Advance the mask
#         current_search_mask = next_search_mask

#     # Leave window open at the end
#     plt.ioff()
#     print("Processing complete. Close the figure window to continue.")
#     plt.show()

#     return saved_edge_maps

def save_edge_focusing_grid(image, filename, output_dir="figures_viz/"):
    """Executes multi-scale edge focusing and saves a 4x4 high-res grid."""
    
    # 1. Setup the 4x4 Figure
    fig, axes = plt.subplots(4, 4, figsize=(20, 20), dpi=300)
    fig.suptitle(f"Multi-Scale Edge Focusing Pipeline: {filename}", fontsize=24, fontweight='bold', y=0.98)
    
    # We will pick exactly 4 scales to capture in our 4 rows
    target_sigmas = [4.0, 3.0, 2.0, 1.0] 
    row_idx = 0
    
    # --- START ALGORITHM LOGIC ---
    mean_val, std_val, T_iterative, dynamic_multiplier = calculate_dynamic_stats(image)
    norm_image = (image - mean_val) / (std_val + 1e-5)
    
    spatial_penalty = np.ones_like(image, dtype=float)
    spatial_penalty[image <= T_iterative] = 1.5
    spatial_penalty[image > T_iterative] = 0.8

    # We still PROCESS all steps (step=0.5) to keep the tracking accurate, 
    # but we only PLOT the target_sigmas.
    sigmas_to_process = np.arange(5.0, 0.4, -0.5)
    current_search_mask = None
    saved_edge_maps = {}
    
    for sigma in sigmas_to_process:
        print(f" -> Processing scale σ = {sigma:.1f}")
        
        kernel = generate_log_kernel(sigma)
        log_img = convolve2d_fft(norm_image, kernel)
        
        max_log_val = np.max(np.abs(log_img))
        if max_log_val > 0:
            log_img = log_img / max_log_val
            
        base_thresh = 0.05 * dynamic_multiplier
        adaptive_thresh_map = base_thresh * spatial_penalty
        
        edges = find_zero_crossings(log_img, search_mask=current_search_mask, variance_thresh_map=adaptive_thresh_map)
        
        # --- PLOTTING LOGIC (Only trigger on our 4 target scales) ---
        # We round to 1 decimal to avoid floating-point comparison issues
        if round(sigma, 1) in target_sigmas and row_idx < 4:
            
            # Col 1: Original Image Reference
            axes[row_idx, 0].imshow(image, cmap='gray')
            axes[row_idx, 0].set_title(f"Scale: σ={sigma:.1f} | Original", fontsize=14)
            axes[row_idx, 0].axis('off')
            
            # Col 2: LoG Response
            axes[row_idx, 1].imshow(log_img, cmap='RdBu', vmin=-1, vmax=1)
            axes[row_idx, 1].set_title("LoG Response", fontsize=14)
            axes[row_idx, 1].axis('off')
            
            # Col 3: Search Mask (Input to this step)
            mask_to_plot = current_search_mask if current_search_mask is not None else np.ones_like(image)
            axes[row_idx, 2].imshow(mask_to_plot.astype(float), cmap='magma', vmin=0, vmax=1)
            axes[row_idx, 2].set_title("Active Search Mask", fontsize=14)
            axes[row_idx, 2].axis('off')
            
            # Col 4: Zero-Crossings (Detected Edges)
            axes[row_idx, 3].imshow(edges.astype(float), cmap='gray', vmin=0, vmax=1)
            axes[row_idx, 3].set_title("Detected Zero-Crossings", fontsize=14)
            axes[row_idx, 3].axis('off')
            
            row_idx += 1

        if sigma in [5.0, 4.0, 3.0, 2.0, 1.0]:
            saved_edge_maps[sigma] = edges
            
        # Dilate for next iteration
        current_search_mask = dilate_mask(edges, iterations=2)

    # --- FINAL SAVING LOGIC ---
    plt.tight_layout()
    # Adjust top to make room for the suptitle
    plt.subplots_adjust(top=0.95) 
    
    out_name = f"{filename.split('.')[0]}_4x4_grid.png"
    out_path = os.path.join(output_dir, out_name)
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig) # Close to free up memory
    print(f" [+] Saved 300 DPI Grid to: {out_path}")

    return saved_edge_maps

def save_edge_focusing_grid_ver4(image, filename, output_dir="figures_viz_ver4/"):
    """Executes multi-scale edge focusing and saves a 4x4 high-res grid. Matches Ver 4"""
    
    # 1. Setup the 4x4 Figure
    fig, axes = plt.subplots(4, 4, figsize=(20, 20), dpi=300)
    fig.suptitle(f"Multi-Scale Edge Focusing Pipeline: {filename}", fontsize=24, fontweight='bold', y=0.98)
    
    # We will pick exactly 4 scales to capture in our 4 rows
    target_sigmas = [4.0, 3.0, 2.0, 1.0] 
    row_idx = 0
    
    # --- START ALGORITHM LOGIC ---
    mean_val, std_val, T_iterative, dynamic_multiplier = calculate_dynamic_stats(image)
    norm_image = (image - mean_val) / (std_val + 1e-5)
    
    spatial_penalty = np.ones_like(image, dtype=float)
    spatial_penalty[image <= T_iterative] = 1.2
    spatial_penalty[image > T_iterative] = 0.8

    # We still PROCESS all steps (step=0.5) to keep the tracking accurate, 
    # but we only PLOT the target_sigmas.
    sigmas_to_process = np.arange(5.0, 0.4, -0.5)
    current_search_mask = None
    saved_edge_maps = {}
    
    for sigma in sigmas_to_process:
        print(f" -> Processing scale σ = {sigma:.1f}")
        
        kernel = generate_log_kernel(sigma)
        log_img = convolve2d_fft(norm_image, kernel)
        
        log_std = np.std(log_img)
        if log_std > 0:
            log_img = log_img / log_std
            
        base_thresh = 0.3 * dynamic_multiplier
        adaptive_thresh_map = base_thresh * spatial_penalty
        
        edges = find_zero_crossings(log_img, search_mask=current_search_mask, variance_thresh_map=adaptive_thresh_map)
        
        # --- PLOTTING LOGIC (Only trigger on our 4 target scales) ---
        # We round to 1 decimal to avoid floating-point comparison issues
        if round(sigma, 1) in target_sigmas and row_idx < 4:
            
            # Col 1: Original Image Reference
            axes[row_idx, 0].imshow(image, cmap='gray')
            axes[row_idx, 0].set_title(f"Scale: σ={sigma:.1f} | Original", fontsize=14)
            axes[row_idx, 0].axis('off')
            
            # Col 2: LoG Response
            axes[row_idx, 1].imshow(log_img, cmap='RdBu', vmin=-1, vmax=1)
            axes[row_idx, 1].set_title("LoG Response", fontsize=14)
            axes[row_idx, 1].axis('off')
            
            # Col 3: Search Mask (Input to this step)
            mask_to_plot = current_search_mask if current_search_mask is not None else np.ones_like(image)
            axes[row_idx, 2].imshow(mask_to_plot.astype(float), cmap='magma', vmin=0, vmax=1)
            axes[row_idx, 2].set_title("Active Search Mask", fontsize=14)
            axes[row_idx, 2].axis('off')
            
            # Col 4: Zero-Crossings (Detected Edges)
            axes[row_idx, 3].imshow(edges.astype(float), cmap='gray', vmin=0, vmax=1)
            axes[row_idx, 3].set_title("Detected Zero-Crossings", fontsize=14)
            axes[row_idx, 3].axis('off')
            
            row_idx += 1

        if sigma in [5.0, 4.0, 3.0, 2.0, 1.0]:
            saved_edge_maps[sigma] = edges
            
        # Dilate for next iteration
        current_search_mask = dilate_mask(edges, iterations=3)

    # --- FINAL SAVING LOGIC ---
    plt.tight_layout()
    # Adjust top to make room for the suptitle
    plt.subplots_adjust(top=0.95) 
    
    out_name = f"{filename.split('.')[0]}_4x4_grid.png"
    out_path = os.path.join(output_dir, out_name)
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig) # Close to free up memory
    print(f" [+] Saved 300 DPI Grid to: {out_path}")

    return saved_edge_maps

# --- Helper to create a dummy image if you want to test without a file ---
def create_dummy_img_file(filename):
    """Creates a synthetic 512x512 image file for testing purposes."""
    x, y = np.meshgrid(np.linspace(-10, 10, 512), np.linspace(-10, 10, 512))
    img = np.sin(x**2 + y**2) * 127 + 128
    img = img.astype(np.uint8)
    
    # Prepend 512 bytes of fake header to match your loader's requirement
    header = np.zeros(512, dtype=np.uint8)
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(img.tobytes())

if __name__ == "__main__":
    files = ["test1.img", "test2.img", "test3.img"]

    # Ensure at least one test file exists to see the visualization
    if not os.path.exists(files[0]):
        print(f"Test file not found. Generating a synthetic '{files[0]}' to demonstrate visualization...")
        create_dummy_img_file(files[0])

    for file in files:
        if not os.path.exists(file):
            continue

        print(f"\nStarting Adaptive Edge Focusing on: {file}")
        img = load_and_preprocess(file) 
        
        # Call the new real-time visualization function
        edge_maps = save_edge_focusing_grid_ver4(img, file)
        print("-" * 50)