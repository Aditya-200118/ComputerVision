import numpy as np
import matplotlib.pyplot as plt
import os
import cv2  # ONLY used for the baseline benchmark and proxy ground truth
import itertools

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.dpi": 300, 
})

def load_and_preprocess(path):
    try:
        with open(path, "rb") as f:
            _ = f.read(512)
            img_data = np.fromfile(f, dtype=np.uint8)
        return img_data.reshape((512, 512)).astype(float)
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()

def calculate_dynamic_stats(image):
    mean_val = np.mean(image)
    std_val = np.std(image)
    skewness = np.mean(((image - mean_val) / std_val) ** 3) if std_val > 0 else 0
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
    radius = int(np.ceil(4 * sigma))
    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(x, y)
    term1 = (X**2 + Y**2) / (2 * sigma**2)
    kernel = - (1.0 / (np.pi * sigma**4)) * (1.0 - term1) * np.exp(-term1)
    kernel = kernel - np.mean(kernel) 
    return kernel

def convolve2d_fft(image, kernel):
    shape = (image.shape[0] + kernel.shape[0] - 1, image.shape[1] + kernel.shape[1] - 1)
    fft_img = np.fft.fft2(image, s=shape)
    fft_ker = np.fft.fft2(kernel, s=shape)
    result = np.real(np.fft.ifft2(fft_img * fft_ker))
    offset_y = kernel.shape[0] // 2
    offset_x = kernel.shape[1] // 2
    return result[offset_y:offset_y+image.shape[0], offset_x:offset_x+image.shape[1]]

def dilate_mask(mask, iterations=2):
    dilated = np.copy(mask)
    shifts = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    for _ in range(iterations):
        current_iter_mask = np.copy(dilated)
        for dy, dx in shifts:
            current_iter_mask |= np.roll(np.roll(dilated, dy, axis=0), dx, axis=1)
        dilated = current_iter_mask
    return dilated

def find_zero_crossings(log_img, search_mask=None, variance_thresh_map=0.0):
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

def run_edge_focusing_target(image, target_multiplier, penalty_bg, penalty_fg):
    """Modified to accept the 3D optimization parameters and return ONLY the final sigma=1.0 map."""
    mean_val, std_val, T_iterative, dynamic_multiplier = calculate_dynamic_stats(image)
    norm_image = (image - mean_val) / (std_val + 1e-5)
    
    # 3D Opt: Injecting the dynamically searched spatial penalties
    spatial_penalty = np.ones_like(image, dtype=float)
    spatial_penalty[image <= T_iterative] = penalty_bg
    spatial_penalty[image > T_iterative] = penalty_fg

    sigmas_to_process = np.arange(5.0, 0.5, -0.5)
    current_search_mask = None
    structural_base_thresh = 0.0  
    
    final_edges = None
    
    for sigma in sigmas_to_process:
        kernel = generate_log_kernel(sigma)
        log_img = convolve2d_fft(norm_image, kernel)
        
        max_log_val = np.max(np.abs(log_img))
        if max_log_val > 0:
            log_img = log_img / max_log_val
        
        if sigma == 5.0:
            structural_base_thresh = np.mean(np.abs(log_img))
            
        # 3D Opt: Test the dynamically injected multiplier against our pipeline stats
        base_thresh = structural_base_thresh * dynamic_multiplier * (target_multiplier / 0.05) 
        
        adaptive_thresh_map = base_thresh * spatial_penalty
        edges = find_zero_crossings(log_img, search_mask=current_search_mask, variance_thresh_map=adaptive_thresh_map)
        
        if sigma == 1.0:
            final_edges = edges
            
        current_search_mask = dilate_mask(edges, iterations=2)
        
    return final_edges

# --- NEW OPTIMIZATION & BENCHMARKING FUNCTIONS ---

def generate_opencv_benchmark(image):
    """Uses OpenCV's state-of-the-art Canny edge detector with automated Otsu thresholding."""
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    
    blurred = cv2.GaussianBlur(img_uint8, (5, 5), 1.0)
    high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    
    canny_edges = cv2.Canny(blurred, low_thresh, high_thresh)
    return canny_edges > 0

def calculate_f1_score(pred_edges, true_edges):
    """Calculates the Dice Coefficient (F1 Score) between two boolean edge maps."""
    intersection = np.logical_and(pred_edges, true_edges).sum()
    if intersection == 0:
        return 0.0
    precision = intersection / pred_edges.sum()
    recall = intersection / true_edges.sum()
    return 2 * (precision * recall) / (precision + recall)

def optimize_pipeline(image, filename):
    print(f"\n[Opt-Pipeline] Starting 3D Grid Search for: {filename}")
    print("  -> Generating OpenCV Canny benchmark...")
    canny_baseline = generate_opencv_benchmark(image)
    
    # Define our 3D hyperparameter search grid
    # Testing 5 multipliers * 3 BG penalties * 3 FG penalties = 45 combinations
    mult_space = np.linspace(0.02, 0.10, 5)  # 0.02, 0.04, 0.06, 0.08, 0.10
    bg_space = [1.0, 1.5, 2.0]               # Normal, Harsh, Very Harsh
    fg_space = [0.5, 0.8, 1.0]               # Very Lenient, Lenient, Normal
    
    best_params = (0.0, 0.0, 0.0)
    best_f1 = -1.0
    best_edge_map = None
    
    combinations = list(itertools.product(mult_space, bg_space, fg_space))
    total_iters = len(combinations)
    
    print(f"  -> Testing {total_iters} parameter combinations...")
    
    for idx, (m, bg, fg) in enumerate(combinations):
        test_edges = run_edge_focusing_target(image, m, bg, fg)
        f1 = calculate_f1_score(test_edges, canny_baseline)
        
        if f1 > best_f1:
            best_f1 = f1
            best_params = (m, bg, fg)
            best_edge_map = test_edges
            
    opt_m, opt_bg, opt_fg = best_params
    print(f"  [!] Optimal Found -> Mult: {opt_m:.3f} | BG Pen: {opt_bg:.1f} | FG Pen: {opt_fg:.1f} (F1: {best_f1:.4f})")
    
    return best_edge_map, canny_baseline, best_params, best_f1

def plot_optimization_results(image, log_edges, canny_edges, opt_params, f1_score, filename, output_dir):
    opt_m, opt_bg, opt_fg = opt_params
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f"Original: {filename}")
    axes[0].axis('off')
    
    axes[1].imshow(canny_edges, cmap='gray')
    axes[1].set_title(f"OpenCV Canny (Ground Truth Proxy)")
    axes[1].axis('off')
    
    axes[2].imshow(log_edges, cmap='gray')
    axes[2].set_title(f"Optimized LoG (m={opt_m:.3f}, bg={opt_bg:.1f}, fg={opt_fg:.1f})\nF1-Score: {f1_score:.3f}")
    axes[2].axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{filename.split('.')[0]}_3D_optimization.pdf")
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  [+] Saved Comparison to {out_path}")

if __name__ == "__main__":
    output_dir = "assignment4_optimization"
    os.makedirs(output_dir, exist_ok=True)
    
    files = ["test1.img", "test2.img", "test3.img"]

    for file in files:
        if not os.path.exists(file):
            continue
            
        img = load_and_preprocess(file) 
        
        # Run the full 3D self-tuning pipeline
        best_log, cv2_canny, best_params, final_f1 = optimize_pipeline(img, file)
        
        # Plot the comparison against the State of the Art
        plot_optimization_results(img, best_log, cv2_canny, best_params, final_f1, file, output_dir)