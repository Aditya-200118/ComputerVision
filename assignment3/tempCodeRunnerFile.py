def threshold_dual_region_growing(image):
    """
    Task 3: Dual thresholding with region growing.
    Uses Cumulative Distribution Function (CDF) heuristics for threshold selection.
    """
    # 1. Automatic Threshold Selection Logic via Histogram CDF
    hist = np.zeros(256, dtype=int)
    for val in image.ravel():
        hist[val] += 1
        
    cdf = np.cumsum(hist) / image.size
    
    T_H = 0
    T_L = 0
    
    # Heuristic Logic: 
    # High threshold targets the top 10% brightest pixels (confident seeds).
    # Low threshold targets the top 25% brightest pixels (relaxed connectivity).
    for i in range(256):
        if cdf[i] >= 0.75 and T_L == 0:
            T_L = i
        if cdf[i] >= 0.90 and T_H == 0:
            T_H = i
            break
            
    # 2. Region Growing (8-connected flood-fill via stack)
    rows, cols = image.shape
    binary_image = np.zeros((rows, cols), dtype=np.uint8)
    seeds = []
    
    # Seed phase: Flag all highly confident pixels
    for r in range(rows):
        for c in range(cols):
            if image[r, c] >= T_H:
                binary_image[r, c] = 255
                seeds.append((r, c))
                
    # Directional array for 8-connected neighbors
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    # Growth phase: Iteratively attach neighbors that meet the relaxed threshold
    while seeds:
        r, c = seeds.pop()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            # Boundary checks
            if 0 <= nr < rows and 0 <= nc < cols:
                # If neighbor is unassigned and passes the low threshold, add it
                if binary_image[nr, nc] == 0 and image[nr, nc] >= T_L:
                    binary_image[nr, nc] = 255
                    seeds.append((nr, nc))
                    
    return binary_image, T_H, T_L