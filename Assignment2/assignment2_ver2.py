import numpy as np
import matplotlib.pyplot as plt
import hashlib

# APA 7 style settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.dpi": 600,
})

def load_and_preprocess(path):
    try:
        with open(path, "rb") as f:
            header = f.read(512)
            img_data = np.fromfile(f, dtype=np.uint8)
        
        original_2d = img_data.reshape((512, 512))
        # Threshold to binary (0=bg, 1=fg)
        # Assuming original is 0 for object and 255 for background based on '1 - binary' logic
        binary = (original_2d <= 128).astype(np.uint8)
        return original_2d, binary
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()

def revert_and_save(binary_img, output_path, header_size=512):
    # Undo inversion: 1 (fg) becomes 0, 0 (bg) becomes 1
    restored_binary = 1 - binary_img
    # Convert back to grayscale (0 and 255)
    restored_gray = (restored_binary * 255).astype(np.uint8)
    flat = restored_gray.flatten()
    header = bytes(header_size)

    with open(output_path, "wb") as f:
        f.write(header)
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
    print(f"Match: {is_match} ({h1})")
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
                    # If neighbor's distance is >= current + 1 (or sqrt2), neighbor covers this pixel
                    dist_to_neighbor = 1.0 if (dy==0 or dx==0) else np.sqrt(2)
                    if metric == 'manhattan' and (dy!=0 and dx!=0): continue 
                    
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
        if metric == 'manhattan': mask = (np.abs(Y) + np.abs(X) < d + 0.01)
        elif metric == 'chessboard': mask = (np.maximum(np.abs(Y), np.abs(X)) < d + 0.01)
        else: mask = (np.sqrt(Y**2 + X**2) < d + 0.01)
        recon[y0:y1, x0:x1] |= mask.astype(np.uint8)
    return recon

# --- Execution ---

if __name__ == "__main__":
    orig_raw, binary_mask = load_and_preprocess("comb.img")
    
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
        
        # Internal check
        diff = np.sum(np.abs(binary_mask.astype(np.int32) - recon.astype(np.int32)))
        print(f"Internal Pixel Error: {diff}")
        
        # Save and Binary Comparison
        out_name = f"reconstructed_{key}.img"
        revert_and_save(recon, out_name)
        compare_files("comb.img", out_name)
        
        # Plotting
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        plt.subplots_adjust(bottom=0.2)
        ax[0].imshow(skel > 0, cmap='gray'); ax[0].set_title("Medial Axis"); ax[0].axis('off')
        ax[1].imshow(recon, cmap='gray'); ax[1].set_title("Reconstruction"); ax[1].axis('off')
        fig.text(0.5, 0.08, f"Figure. {label} Metric: Skeleton and Lossless Reconstruction.", ha='center', fontsize=12, fontweight='bold')
        plt.savefig(f"APA_{key}.png", dpi=600, bbox_inches='tight')
        plt.close()