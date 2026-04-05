
import numpy as np
import cv2
import os

def create_1080p_60fps_animation(image, filename, output_dir="."):
    """Generates a smooth 1080p 60FPS video of the Edge Focusing process."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[Video] Initializing 1080p 60FPS Render for {filename}...")
    
    # --- Video Settings ---
    WIDTH, HEIGHT = 1920, 1080
    FPS = 60
    # Use 'mp4v' or 'avc1' for MP4 compression
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out_path = os.path.join(output_dir, f"{filename.split('.')[0]}_edge_focusing_1080p.mp4")
    video = cv2.VideoWriter(out_path, fourcc, FPS, (WIDTH, HEIGHT))
    
    # --- Precompute Image Stats ---
    mean_val, std_val, T_iterative, dynamic_multiplier = calculate_dynamic_stats(image)
    norm_image = (image - mean_val) / (std_val + 1e-5)
    
    spatial_penalty = np.ones_like(image, dtype=float)
    spatial_penalty[image <= T_iterative] = 1.5
    spatial_penalty[image > T_iterative] = 0.8

    # --- Micro-Stepping for Smooth Animation ---
    # Going from 5.0 to 1.0 with a step of 0.01 gives us 400 frames (~6.6 seconds of video at 60fps)
    sigmas_to_process = np.arange(5.0, 0.99, -0.01)
    current_search_mask = None
    
    # Pre-process the original image for the display grid
    img_display = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_display_color = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)

    for i, sigma in enumerate(sigmas_to_process):
        if i % 50 == 0:
            print(f"  -> Rendering frame {i}/{len(sigmas_to_process)} (\u03C3 = {sigma:.2f})")
            
        # 1. Math Operations
        kernel = generate_log_kernel(sigma)
        log_img = convolve2d_fft(norm_image, kernel)
        
        max_log_val = np.max(np.abs(log_img))
        if max_log_val > 0:
            log_img = log_img / max_log_val
            
        base_thresh = 0.05 * dynamic_multiplier
        adaptive_thresh_map = base_thresh * spatial_penalty
        
        edges = find_zero_crossings(log_img, search_mask=current_search_mask, variance_thresh_map=adaptive_thresh_map)
        
        # Because delta_sigma is so small (0.01), edges barely move per frame. 
        # 1 iteration of dilation is plenty to track the smooth sub-pixel drift.
        current_search_mask = dilate_mask(edges, iterations=1)
        
        # ==========================================
        # VISUALIZATION & FRAME ASSEMBLY
        # ==========================================
        
        # Canvas: Dark gray background
        frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 30 
        
        # Panel 1: Original Image + Red Edges Overlay
        overlay = img_display_color.copy()
        overlay[edges > 0] = [0, 0, 255] # BGR format (Red)
        
        # Panel 2: LoG Response (Custom Blue/Red Colormap via OpenCV)
        # Map [-1, 1] to [0, 255]
        log_norm = cv2.normalize(log_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        log_color = cv2.applyColorMap(log_norm, cv2.COLORMAP_TWILIGHT_SHIFTED)
        
        # Panel 3: Active Search Mask
        mask_display = (current_search_mask.astype(np.uint8) * 255)
        mask_color = cv2.applyColorMap(mask_display, cv2.COLORMAP_MAGMA)
        if i == 0: mask_color = np.zeros_like(mask_color) # Blank for first frame
        
        # Panel 4: Zero Crossings
        edges_display = (edges.astype(np.uint8) * 255)
        edges_color = cv2.cvtColor(edges_display, cv2.COLOR_GRAY2BGR)
        
        # Resize panels to fit a 2x2 grid inside 1080p (leaving room for titles)
        panel_w, panel_h = 800, 450
        p1 = cv2.resize(overlay, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
        p2 = cv2.resize(log_color, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
        p3 = cv2.resize(mask_color, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
        p4 = cv2.resize(edges_color, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
        
        # Place panels on the 1080p frame
        # Top Left
        frame[80:530, 100:900] = p1
        # Top Right
        frame[80:530, 1020:1820] = p2
        # Bottom Left
        frame[600:1050, 100:900] = p3
        # Bottom Right
        frame[600:1050, 1020:1820] = p4
        
        # --- Add Text overlays ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Multi-Scale Edge Focusing | Current Scale: Sigma = {sigma:.3f}", 
                    (WIDTH//2 - 450, 45), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(frame, "1. Edge Tracking Overlay", (100, 70), font, 0.8, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "2. LoG Convolution Response", (1020, 70), font, 0.8, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "3. Morphological Search Mask", (100, 590), font, 0.8, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "4. Detected Zero-Crossings", (1020, 590), font, 0.8, (200, 200, 200), 1, cv2.LINE_AA)
        
        video.write(frame)

    video.release()
    print(f"[+] Video successfully saved to: {out_path}\n")

# To run this, add to your __main__ block:
#         create_1080p_60fps_animation(img, file, output_dir="assignment4_videos")