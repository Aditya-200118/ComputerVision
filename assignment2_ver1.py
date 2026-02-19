import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 600,
    }
)

def load_and_preprocess(path):
    try:
        with open(path, "rb") as f:
            _ = f.read(512)
            img = np.fromfile(f, dtype=np.uint8)

        original = img.reshape((512, 512))
        binary = (original > 128).astype(np.uint8)
        binary = 1 - binary

        return original, binary
    except FileNotFoundError:
        print(f"Error: File {path} not found.")
        exit()

def compute_chessboard_distance(binary_mask):
    """
    Computes the Chessboard (L_infinity) distance transform from scratch 
    using a 2-pass sequential algorithm.
    """
    H, W = binary_mask.shape
    
    # Initialize Distance Matrix: 0 for background, infinity for foreground objects
    dt = np.where(binary_mask > 0, float('inf'), 0)
    
    # Pass 1: Top-Left to Bottom-Right
    for y in range(H):
        for x in range(W):
            if dt[y, x] > 0:
                neighbors = []
                if y > 0: neighbors.append(dt[y-1, x])          # Top
                if x > 0: neighbors.append(dt[y, x-1])          # Left
                if y > 0 and x > 0: neighbors.append(dt[y-1, x-1])      # Top-Left
                if y > 0 and x < W - 1: neighbors.append(dt[y-1, x+1])  # Top-Right
                
                if neighbors:
                    dt[y, x] = min(dt[y, x], min(neighbors) + 1)
                    
    # Pass 2: Bottom-Right to Top-Left
    for y in range(H - 1, -1, -1):
        for x in range(W - 1, -1, -1):
            if dt[y, x] > 0:
                neighbors = []
                if y < H - 1: neighbors.append(dt[y+1, x])          # Bottom
                if x < W - 1: neighbors.append(dt[y, x+1])          # Right
                if y < H - 1 and x < W - 1: neighbors.append(dt[y+1, x+1])  # Bottom-Right
                if y < H - 1 and x > 0: neighbors.append(dt[y+1, x-1])      # Bottom-Left
                
                if neighbors:
                    dt[y, x] = min(dt[y, x], min(neighbors) + 1)
                    
    return dt.astype(np.int32)

def get_local_maxima(dt):
    """
    Finds local maxima in a 3x3 neighborhood using vectorized NumPy slicing.
    """
    H, W = dt.shape
    # Pad the distance transform with 0s to safely handle edge pixels
    padded = np.pad(dt, pad_width=1, mode='constant', constant_values=0)
    max_filter = np.zeros_like(dt)
    
    # Shift the image in all 9 directions (including center)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            # Slice the padded array to represent a shifted neighborhood
            shifted = padded[1+dy : H+1+dy, 1+dx : W+1+dx]
            max_filter = np.maximum(max_filter, shifted)
            
    # A pixel is a local maximum if its original value equals the neighborhood maximum
    return dt == max_filter

import numpy as np

def revert_and_save(binary_img, output_path, header_size=512):
    """
    Reverts preprocessing:
    - undo inversion
    - convert binary back to grayscale (0 or 255)
    - save with header padding
    """

    # Undo inversion
    restored_binary = 1 - binary_img

    # Convert back to grayscale values
    restored_gray = (restored_binary * 255).astype(np.uint8)

    # Flatten for saving
    flat = restored_gray.flatten()

    # Create dummy header (same size as original skip)
    header = bytes(header_size)

    # Write file
    with open(output_path, "wb") as f:
        f.write(header)
        f.write(flat.tobytes())

    print(f"Reverted image saved to: {output_path}")


if __name__ == "__main__":
    print("Starting\n")

    original_img, binary_mask = load_and_preprocess("comb.img")
    plt.imsave("Image_B_Original.png", original_img, cmap="gray")
    plt.imsave("Image_BT_Binary.png", binary_mask, cmap="gray")

    # ==========================================
    # TASK 1: MEDIAL AXIS COMPUTATION (FROM SCRATCH)
    # ==========================================
    print("Computing Distance Transform...")
    dt = compute_chessboard_distance(binary_mask)
    
    print("Finding Local Maxima...")
    local_max = get_local_maxima(dt)
    
    # M retains the distance transform value at each skeleton pixel 
    M = np.where(local_max & (binary_mask > 0), dt, 0)
    
    # Display the resulting Medial Axis image M as a binary image [cite: 5]
    M_binary = (M > 0).astype(np.uint8)
    plt.imsave("Image_M_Binary_Skeleton.png", M_binary, cmap="gray")

    # ==========================================
    # TASK 2: IMAGE RECONSTRUCTION
    # ==========================================
    print("Reconstructing Image...")
    # Initialize B_R
    B_R = np.zeros_like(M, dtype=np.uint8)
    
    # Find all skeleton pixels in M
    y_coords, x_coords = np.nonzero(M)
    
    # Reconstruct the original binary image from the Medial Axis image M [cite: 6]
    for y, x in zip(y_coords, x_coords):
        d = M[y, x]
        r = d - 1  # L_infinity distance maps to a square of radius (d-1)
        
        y_min = max(0, y - r)
        y_max = min(M.shape[0], y + r + 1)
        x_min = max(0, x - r)
        x_max = min(M.shape[1], x + r + 1)
        
        B_R[y_min:y_max, x_min:x_max] = 1
        
    plt.imsave("Image_BR_Reconstructed.png", B_R, cmap="gray")

    # ==========================================
    # VALIDATION & VISUALIZATION
    # ==========================================
    
    # Verify the reconstruction should be lossless [cite: 7]
    difference = np.sum(np.abs(binary_mask - B_R))
    is_lossless = (difference == 0)
    print(f"Reconstruction verification (Differences = {difference}). Lossless: {is_lossless}")
    revert_and_save(B_R, "restored.img")

    # Generate Hardcopy Figure 
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # axes[0].imshow(binary_mask, cmap='gray')
    # axes[0].set_title('Original Binary Image (B)')
    # axes[0].axis('off')

    # axes[1].imshow(M_binary, cmap='gray')
    # axes[1].set_title('Medial Axis (M_binary)')
    # axes[1].axis('off')

    # axes[2].imshow(B_R, cmap='gray')
    # axes[2].set_title('Reconstructed Image (B_R)')
    # axes[2].axis('off')

    # plt.tight_layout()
    # plt.savefig("Assignment2_Hardcopies.png") 
    # plt.show()

    print("\nEnded")