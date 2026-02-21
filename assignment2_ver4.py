import numpy as np
import matplotlib.pyplot as plt
import hashlib

# APA 7 style settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titlesize": 11,
    "figure.dpi": 600,
})

def load_and_preprocess(path):
    try:
        with open(path, "rb") as f:
            # Capturing original header to allow bit-perfect matching later
            header = f.read(512)
            img_data = np.fromfile(f, dtype=np.uint8)
        
        original_2d = img_data.reshape((512, 512))
        # Threshold to binary (0=bg, 1=fg)
        # Objects are black (0), background is white (255)
        binary = (original_2d <= 128).astype(np.uint8)
        return original_2d, binary, header
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()

def revert_and_save(binary_img, output_path, original_header):
    # Undo inversion: 1 (fg/object) becomes 0, 0 (bg) becomes 255
    restored_gray = np.where(binary_img == 1, 0, 255).astype(np.uint8)
    flat = restored_gray.flatten()

    with open(output_path, "wb") as f:
        # Writing back the exact original header for MD5 matching
        f.write(original_header)
        f.write(flat.tobytes())
    print(f"File saved: {output_path}")

def compare_files(file1, file2):
    """Byte-by-byte precision check using MD5 hashes."""
    def get_hash(file):
        hasher = hashlib.md5()
        with open(file, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    
    h1, h2 = get_hash(file1), get_hash(file2)
    is_match = h1 == h2
    print(f"Verification: {file1} vs {file2}")
    print(f"Match: {is_match}")
    print(f"Original: {h1}\nRecon:    {h2}")
    return is_match

# --- Distance Transform Algorithms ---

def compute_manhattan_distance(mask):
    H, W = mask.shape
    dt = np.where(mask > 0, 1e6, 0).astype(np.float32)
    for y in range(H):
        for x in range(W):
            if dt[y, x] > 0:
                t = dt[y-1, x] if y > 0 else 1e6
                l = dt[y, x-1] if x > 0 else 1e6
                dt[y, x] = min(dt[y, x], min(t, l) + 1)
    for y in range(H-1, -1, -1):
        for x in range(W-1, -1, -1):
            if dt[y, x] > 0:
                b = dt[y+1, x] if y < H-1 else 1e6
                r = dt[y, x+1] if x < W-1 else 1e6
                dt[y, x] = min(dt[y, x], min(b, r) + 1)
    return dt

def compute_chessboard_distance(mask):
    H, W = mask.shape
    dt = np.where(mask > 0, 1e6, 0).astype(np.float32)
    for y in range(H):
        for x in range(W):
            if dt[y, x] > 0:
                n = [dt[y-1, x] if y > 0 else 1e6, dt[y, x-1] if x > 0 else 1e6,
                     dt[y-1, x-1] if y > 0 and x > 0 else 1e6, dt[y-1, x+1] if y > 0 and x < W-1 else 1e6]
                dt[y, x] = min(dt[y, x], min(n) + 1)
    for y in range(H-1, -1, -1):
        for x in range(W-1, -1, -1):
            if dt[y, x] > 0:
                n = [dt[y+1, x] if y < H-1 else 1e6, dt[y, x+1] if x < W-1 else 1e6,
                     dt[y+1, x+1] if y < H-1 and x < W-1 else 1e6, dt[y+1, x-1] if y < H-1 and x > 0 else 1e6]
                dt[y, x] = min(dt[y, x], min(n) + 1)
    return dt

def compute_euclidean_distance(mask):
    H, W = mask.shape
    dt = np.where(mask > 0, 1e6, 0).astype(np.float32)
    s2 = np.sqrt(2)
    for y in range(H):
        for x in range(W):
            if dt[y, x] > 0:
                n = [dt[y-1, x]+1 if y > 0 else 1e6, dt[y, x-1]+1 if x > 0 else 1e6,
                     dt[y-1, x-1]+s2 if y > 0 and x > 0 else 1e6, dt[y-1, x+1]+s2 if y > 0 and x < W-1 else 1e6]
                dt[y, x] = min(dt[y, x], min(n))
    for y in range(H-1, -1, -1):
        for x in range(W-1, -1, -1):
            if dt[y, x] > 0:
                n = [dt[y+1, x]+1 if y < H-1 else 1e6, dt[y, x+1]+1 if x < W-1 else 1e6,
                     dt[y+1, x+1]+s2 if y < H-1 and x < W-1 else 1e6, dt[y+1, x-1]+s2 if y < H-1 and x > 0 else 1e6]
                dt[y, x] = min(dt[y, x], min(n))
    return dt

# --- Skeleton & Reconstruction ---

def get_skeleton(dt, metric):
    H, W = dt.shape
    skeleton = np.zeros_like(dt)
    padded = np.pad(dt, 1, mode='constant', constant_values=0)
    offsets = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
    
    for y in range(H):
        for x in range(W):
            if dt[y, x] > 0:
                is_max = True
                d_curr = dt[y, x]
                for dy, dx in offsets:
                    if metric == 'manhattan' and (dy!=0 and dx!=0): continue 
                    dist_to_neighbor = 1.0 if (dy==0 or dx==0) else np.sqrt(2)
                    
                    if padded[y+1+dy, x+1+dx] >= d_curr + (1.0 if metric != 'euclidean' else dist_to_neighbor) - 0.001:
                        is_max = False
                        break
                if is_max:
                    skeleton[y, x] = d_curr
    return skeleton

def reconstruct_lossless(skeleton, metric):
    H, W = skeleton.shape
    recon = np.zeros_like(skeleton, dtype=np.uint8)
    y_idx, x_idx = np.nonzero(skeleton)
    for y, x in zip(y_idx, x_idx):
        d = skeleton[y, x]
        r = int(np.ceil(d))
        y0, y1 = max(0, y-r), min(H, y+r+1)
        x0, x1 = max(0, x-r), min(W, x+r+1)
        Y, X = np.ogrid[y0-y:y1-y, x0-x:x1-x]
        
        # Using strict inequality < d ensures no bleeding into the background
        # while keeping the full tool thickness.
        if metric == 'manhattan': mask = (np.abs(Y) + np.abs(X) < d)
        elif metric == 'chessboard': mask = (np.maximum(np.abs(Y), np.abs(X)) < d)
        else: mask = (np.sqrt(Y**2 + X**2) < d)
        recon[y0:y1, x0:x1] |= mask.astype(np.uint8)
    return recon

# --- Execution ---
from scipy import ndimage
def verify_with_scipy(original_binary, skeleton_image, metric='chessboard'):
    """
    Verifies if reconstruction is lossless using SciPy's grayscale dilation.
    """
    # 1. Define the footprint (Structuring Element) based on the metric
    if metric == 'chessboard':
        # L_infinity uses a 3x3 square of 1s
        footprint = np.ones((3, 3))
    elif metric == 'manhattan':
        # L_1 uses a 3x3 diamond (cross)
        footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    elif metric == 'euclidean':
        # L_2 is approximated; for true L2, SciPy's distance_transform_edt is preferred
        footprint = np.ones((3, 3)) 

    # 2. Perform Grayscale Dilation
    # The distance values in the skeleton are propagated outwards
    reconstructed_dt = ndimage.grey_dilation(skeleton_image, footprint=footprint)
    
    # 3. Threshold to get the binary image
    # In SciPy's CDT/EDT, a pixel is foreground if its distance > 0
    reconstructed_binary = (reconstructed_dt > 0).astype(np.uint8)

    # 4. Logical Check for Lossless Integrity
    is_lossless = np.array_equal(original_binary, reconstructed_binary)
    pixel_diff = np.sum(np.abs(original_binary.astype(np.int32) - reconstructed_binary.astype(np.int32)))
    
    return is_lossless, pixel_diff

def save_final_comparison(original_binary, reconstructed_binary, label, metric_key):
    """
    Saves a 600 DPI figure comparing the Original Binary Image (B) 
    and the Reconstructed Binary Image (BR) side-by-side.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    plt.subplots_adjust(bottom=0.2)
    
    # Left: Original Image B
    ax[0].imshow(original_binary, cmap='gray')
    ax[0].set_title("Original Binary Image (B)")
    ax[0].axis('off')
    
    # Right: Reconstructed Image BR
    ax[1].imshow(reconstructed_binary, cmap='gray')
    ax[1].set_title("Reconstructed Binary Image (BR)")
    ax[1].axis('off')
    
    # APA 7 Style Caption
    fig.text(0.5, 0.08, f"Figure. Comparison of Original (B) and Lossless Reconstruction (BR) using {label} Metric.", 
             ha='center', fontsize=12, fontweight='bold')
    
    output_filename = f"Final_Comparison_{metric_key}.png"
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Comparison figure saved: {output_filename}")

def save_difference_analysis(original_binary, reconstructed_binary, label, metric_key):
    """
    Saves a 600 DPI figure with three panels: Original, Reconstructed, 
    and the Absolute Difference (Error Map).
    """
    # Calculate the absolute difference
    # 0 means pixels match; 1 means there is an error
    difference_map = np.abs(original_binary.astype(np.int32) - reconstructed_binary.astype(np.int32))
    error_count = np.sum(difference_map)
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    plt.subplots_adjust(bottom=0.2, wspace=0.3)
    
    # Panel 1: Original B
    ax[0].imshow(original_binary, cmap='gray')
    ax[0].set_title("Original (B)")
    ax[0].axis('off')
    
    # Panel 2: Reconstructed BR
    ax[1].imshow(reconstructed_binary, cmap='gray')
    ax[1].set_title("Reconstructed (BR)")
    ax[1].axis('off')
    
    # Panel 3: Difference Map (B - BR)
    # Using 'hot' or 'inferno' colormap makes errors stand out
    ax[2].imshow(difference_map, cmap='hot')
    ax[2].set_title(f"Difference Map (Errors: {error_count})")
    ax[2].axis('off')
    
    # APA 7 Style Caption
    fig.text(0.5, 0.08, f"Figure. Error Analysis for {label} Metric. "
             f"Total Pixel Discrepancy: {error_count}.", 
             ha='center', fontsize=12, fontweight='bold')
    
    output_filename = f"Error_Analysis_{metric_key}.png"
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Error analysis saved: {output_filename} (Errors: {error_count})")

if __name__ == "__main__":
    orig_raw, binary_mask, saved_header = load_and_preprocess("comb.img")
    
    metrics = [
        ("Manhattan", compute_manhattan_distance, 'manhattan'),
        ("Chessboard", compute_chessboard_distance, 'chessboard'),
        ("Euclidean", compute_euclidean_distance, 'euclidean')
    ]
    
    for label, func, key in metrics:
        print(f"\n--- Processing {label} ---")
        dt = func(binary_mask)
        skel = get_skeleton(dt, key)
        recon = reconstruct_lossless(skel, key)
        
        # Internal check: Comparing original binary mask with the reconstruction
        diff = np.sum(np.abs(binary_mask.astype(np.int32) - recon.astype(np.int32)))
        print(f"Internal Pixel Error: {diff}")
        
        # Revert and Save: Restoring original color mapping and header
        out_name = f"reconstructed_{key}.img"
        revert_and_save(recon, out_name, saved_header)
        compare_files("comb.img", out_name)
        
        # Subplot style visualization
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        plt.subplots_adjust(bottom=0.2)
        ax[0].imshow(skel > 0, cmap='gray'); ax[0].set_title("Medial Axis"); ax[0].axis('off')
        ax[1].imshow(recon, cmap='gray'); ax[1].set_title("Reconstruction"); ax[1].axis('off')
        fig.text(0.5, 0.08, f"Figure. {label} Metric: Skeleton and Lossless Reconstruction.", 
                 ha='center', fontsize=12, fontweight='bold')
        plt.savefig(f"APA_{key}.png", dpi=600, bbox_inches='tight')
        plt.close()
        # Add this in the main block before the loop
        save_final_comparison(binary_mask, recon, label, key)
        
        save_difference_analysis(binary_mask, recon, label, key)
    plt.imsave("Original_Binary_B.png", binary_mask, cmap='gray')