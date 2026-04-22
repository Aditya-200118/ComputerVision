import numpy as np
import matplotlib.pyplot as plt

# The 16x16 grid you provided
grid = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0], # S is at index (2, 6)
    [0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
])

# Create the plot
plt.figure(figsize=(8, 8))

# Display the matrix as an image (0=white, 1=black)
plt.imshow(grid, cmap='binary', interpolation='nearest')

# Mark the starting pixel 'S' (Row 2, Column 6 in 0-indexed coordinates)
plt.text(6, 2, 'S', color='red', fontsize=16, ha='center', va='center', fontweight='bold')

# Add grid lines to clearly see the 16x16 boundaries
plt.xticks(np.arange(-0.5, 16, 1), labels=[])
plt.yticks(np.arange(-0.5, 16, 1), labels=[])
plt.grid(color='gray', linestyle='-', linewidth=1)

# Add standard axis labels for easy coordinate reading
plt.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True, labelbottom=True, labelleft=True)
plt.xticks(np.arange(0, 16, 1))
plt.yticks(np.arange(0, 16, 1))

plt.title("16x16 Contour Grid Verification")
plt.show()