import numpy as np

def compute_chain_and_difference(pixels):
    """
    Computes the 8-directional chain code and first-difference code.
    Directions: 0:E, 1:NE, 2:N, 3:NW, 4:W, 5:SW, 6:S, 7:SE
    """
    direction_map = {
        (0, 1): 0,   # East
        (-1, 1): 1,  # North-East
        (-1, 0): 2,  # North
        (-1, -1): 3, # North-West
        (0, -1): 4,  # West
        (1, -1): 5,  # South-West
        (1, 0): 6,   # South
        (1, 1): 7    # South-East
    }
    
    # 1. Calculate Chain Code
    chain_code = []
    for i in range(len(pixels)):
        curr_p = pixels[i]
        next_p = pixels[(i + 1) % len(pixels)] # Loop back to S at the end
        
        delta = (next_p[0] - curr_p[0], next_p[1] - curr_p[1])
        chain_code.append(direction_map[delta])
        
    # 2. Calculate First-Difference Code (d_i = (c_i - c_{i-1}) mod 8)
    diff_code = []
    n = len(chain_code)
    for i in range(n):
        diff = (chain_code[i] - chain_code[i-1] + 8) % 8
        diff_code.append(diff)
        
    return chain_code, diff_code

# The ordered boundary coordinates clockwise starting from S (Row 2, Col 6)
# Extracted directly from the provided 16x16 grid
contour_S = [
    (2,6), (2,7), (2,8),          # Top edge
    (3,9), (4,10), (5,11),        # Slanted top-right
    (6,11),                       # Vertical drop
    (7,10), (8,9),                # Slanted bottom-right upper
    (8,8), (8,7),                 # Inward horizontal cut
    (9,7), (10,7),                # Vertical drop inward
    (11,6),                       # Slanted bottom-right lower
    (11,5), (11,4),               # Bottom edge
    (10,4),                       # Vertical rise inward
    (9,3),                        # Slanted bottom-left
    (8,3), (7,3), (6,3),          # Left edge
    (5,4), (4,5),                 # Slanted top-left lower
    (3,5)                         # Vertical rise to close shape
]

# Run the algorithm
c_code, d_code = compute_chain_and_difference(contour_S)

print(f"Chain Code:            {c_code}")
print(f"First-Difference Code: {d_code}")