import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import time

# ============================================================
# CONFIGURATION & STYLE
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150
})

# ============================================================
# PIPELINE STEP 1: PREPROCESSING
# ============================================================

def load_and_preprocess(path):
    """Loads raw image, strips header, returns original and binary."""
    try:
        with open(path, 'rb') as f:
            _ = f.read(512) # Strip header
            img = np.fromfile(f, dtype=np.uint8)
        
        original = img.reshape((512, 512))
        
        # [cite_start]Thresholding (T=128) [cite: 3]
        binary = (original > 128).astype(np.uint8)
        
        # Automatic foreground detection (invert if foreground is majority)
        if np.sum(binary) > (binary.size // 2):
            binary = 1 - binary
            
        return original, binary
    except FileNotFoundError:
        print(f"Error: File {path} not found.")
        exit()

# ============================================================
# PIPELINE STEP 2: SEGMENTATION (CCL ALGORITHM)
# ============================================================

def label_components(binary_img):
    """Iterative CCL with 4-connectivity."""
    rows, cols = binary_img.shape
    labels = np.zeros((rows, cols), dtype=int)
    current_label = 1
    equivalences = defaultdict(set)

    # Pass 1: Assign temporary labels
    for i in range(rows):
        for j in range(cols):
            if binary_img[i, j] == 1:
                neighbors = []
                if i > 0 and labels[i-1, j] > 0: neighbors.append(labels[i-1, j])
                if j > 0 and labels[i, j-1] > 0: neighbors.append(labels[i, j-1])

                if not neighbors:
                    labels[i, j] = current_label
                    current_label += 1
                elif len(neighbors) == 1:
                    labels[i, j] = neighbors[0]
                else:
                    min_lab = min(neighbors)
                    labels[i, j] = min_lab
                    for lab in neighbors:
                        if lab != min_lab:
                            equivalences[min_lab].add(lab)
                            equivalences[lab].add(min_lab)

    # Resolve Equivalences
    label_map = {}
    visited = set()
    
    def dfs(node, root):
        stack = [node]
        while stack:
            n = stack.pop()
            if n not in visited:
                visited.add(n)
                label_map[n] = root
                stack.extend(equivalences[n])

    for i in range(1, current_label):
        if i not in visited:
            dfs(i, i)

    # Pass 2: Apply resolved labels
    for i in range(rows):
        for j in range(cols):
            if labels[i, j] > 0:
                labels[i, j] = label_map.get(labels[i, j], labels[i, j])

    return labels

# ============================================================
# PIPELINE STEP 3: POST-PROCESSING (RE-LABELING)
# ============================================================

def filter_and_relabel(labels, min_size):
    """
    Filters small components and re-indexes valid ones to 1, 2, 3...
    Returns: Filtered label map, mapping dict (new_id -> old_id)
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_mask = (unique_labels != 0) & (counts >= min_size)
    valid_labels = unique_labels[valid_mask]
    
    new_labels = np.zeros_like(labels)
    mapping = {}
    
    for new_id, old_id in enumerate(valid_labels, 1):
        new_labels[labels == old_id] = new_id
        mapping[new_id] = old_id
        
    return new_labels, mapping

# ============================================================
# PIPELINE STEP 4: FEATURE EXTRACTION & ANALYSIS
# ============================================================

def extract_features(labels):
    """Calculates geometric properties aligned with Lecture Notes."""
    props = {}
    indices = np.unique(labels)
    indices = indices[indices != 0]

    for lab in indices:
        # np.argwhere returns (row, col)
        coords = np.argwhere(labels == lab)
        y, x = coords[:, 0], coords[:, 1] # y=row, x=col
        area = len(coords)
        
        # [cite_start]1. Centroid [cite: 14]
        # Xc = mean(columns), Yc = mean(rows)
        xc, yc = np.mean(x), np.mean(y)
        
        # [cite_start]Bounding Box [cite: 16]
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        
        # 2. Central Moments
        x_prime = x - xc
        y_prime = y - yc
        
        # a = sum(x'^2), b = 2*sum(x'y'), c = sum(y'^2)
        a = np.sum(x_prime**2)
        b = 2 * np.sum(x_prime * y_prime)
        c = np.sum(y_prime**2)
        
        # 3. Orientation (Theta)
        # tan(2theta) = b / (a - c)
        theta = 0.5 * np.arctan2(b, a - c)
        
        # 4. Principal Moments (Imax, Imin)
        # Using explicit eigenvalues formulation
        term = np.sqrt((a - c)**2 + b**2)
        Imax = (a + c + term) / 2
        Imin = (a + c - term) / 2
        
        # Validation: a+c should equal Imax+Imin (Invariant)
        moment_error = abs((a + c) - (Imax + Imin))
        
        # 5. Eccentricity & Compactness
        elongation = Imax / Imin if Imin > 0 else 0
        
        # [cite_start]Eccentricity formula: sqrt(1 - Imin/Imax) [cite: 20]
        eccentricity = np.sqrt(1 - (Imin / Imax)) if Imax > 0 else 0
        
        # [cite_start]Perimeter (Boundary pixel count) [cite: 20]
        perimeter = 0
        rows, cols = labels.shape
        for py, px in coords:
            # Check 4-neighbors to see if it's an edge
            is_edge = (py==0 or py==rows-1 or px==0 or px==cols-1)
            if not is_edge:
                if (labels[py-1, px]!=lab or labels[py+1, px]!=lab or 
                    labels[py, px-1]!=lab or labels[py, px+1]!=lab):
                    is_edge = True
            if is_edge: perimeter += 1
            
        compactness = (perimeter**2) / area
        
        # 6. Shape Classification
        if eccentricity < 0.6: shape_class = "Compact"
        elif eccentricity < 0.9: shape_class = "Oval"
        else: shape_class = "Elongated"

        props[lab] = {
            "area": area,
            "centroid": (xc, yc),
            "bbox": (min_x, min_y, max_x, max_y),
            "orientation": theta,
            "eccentricity": eccentricity,
            "compactness": compactness,
            "shape_class": shape_class,
            "moment_error": moment_error,
            "principal_axes": (Imax, Imin)
        }
        
    return props

# ============================================================
# PIPELINE STEP 5: VISUALIZATION & REPORTING
# ============================================================

def save_feature_table(props, size_thresh):
    """Generates and saves the APA-style table as an image."""
    if not props:
        return

    table_data = []
    # Sort by ID
    for lab in sorted(props.keys()):
        p = props[lab]
        xc, yc = p["centroid"]
        table_data.append([
            lab,
            p["area"],
            f"({xc:.1f}, {yc:.1f})",
            f"{p['eccentricity']:.3f}",
            f"{p['compactness']:.2f}",
            p['shape_class']
        ])

    column_labels = ["ID", "Area", "Centroid", "Eccentricity", "Compactness", "Shape"]

    # Dynamic height based on row count
    fig_h = len(table_data) * 0.3 + 1.5
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.axis('off')
    
    table = ax.table(
        cellText=table_data,
        colLabels=column_labels,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Styling for APA Look
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f2f2f2') # Light gray header

    plt.title(f"Table 1: Component Features (Size ≥ {size_thresh})", pad=10, fontsize=12, fontweight='bold')
    plt.tight_layout()
    outfile = f"Table_Size_{size_thresh}.png"
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {outfile}")

def visualize_results(original, labels, props, size_thresh):
    """
    Generates Image C (with overlays) and Histograms.
    """
    
    # --- 1. Main Visualization (Image C) ---
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create colored map
    colored = np.zeros((*labels.shape, 3))
    np.random.seed(42) # Deterministic colors
    
    # Black background
    colored[:] = [0, 0, 0]
    
    unique_labs = list(props.keys())
    for lab in unique_labs:
        color = np.random.rand(3)
        # Boost brightness/saturation
        while np.mean(color) < 0.4: color = np.random.rand(3)
        colored[labels == lab] = color

    ax.imshow(colored)
    ax.set_title(f"Image C: Filtered Components (Size ≥ {size_thresh})\n"
                 f"Visualization includes Centroids, Bounding Boxes, and Principal Axes", pad=15)
    ax.axis('off')

    for lab, p in props.items():
        xc, yc = p["centroid"]
        min_x, min_y, max_x, max_y = p["bbox"]
        theta = p["orientation"]
        
        # Bounding Box (Subtle)
        rect = patches.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y,
                                 linewidth=0.8, edgecolor='white', facecolor='none', 
                                 linestyle=':', alpha=0.7)
        ax.add_patch(rect)
        
        # Principal Axes Calculation
        # Length scaled for visibility (not exact eigen-length)
        axis_len = 30 
        
        # Major Axis (Yellow)
        x1 = xc + axis_len * np.cos(theta)
        y1 = yc + axis_len * np.sin(theta)
        x2 = xc - axis_len * np.cos(theta)
        y2 = yc - axis_len * np.sin(theta)
        ax.plot([x1, x2], [y1, y2], color='yellow', linewidth=1.5, alpha=0.9)
        
        # Minor Axis (Red) - Perpendicular
        x3 = xc + (axis_len * 0.6) * np.cos(theta + np.pi/2)
        y3 = yc + (axis_len * 0.6) * np.sin(theta + np.pi/2)
        x4 = xc - (axis_len * 0.6) * np.cos(theta + np.pi/2)
        y4 = yc - (axis_len * 0.6) * np.sin(theta + np.pi/2)
        ax.plot([x3, x4], [y3, y4], color='red', linewidth=1.5, alpha=0.9)

        # Label
        ax.text(xc, yc, str(lab), color='cyan', fontsize=8, fontweight='bold',
                ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5, pad=0))

    # Legend for Axes
    legend_elements = [
        plt.Line2D([0], [0], color='yellow', lw=2, label='Major Axis'),
        plt.Line2D([0], [0], color='red', lw=2, label='Minor Axis')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    outfile = f"Image_C_Size_{size_thresh}.png"
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"[Saved] {outfile}")

    # --- 2. Statistical Histograms ---
    if len(props) > 1:
        fig_hist, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        areas = [p['area'] for p in props.values()]
        eccs = [p['eccentricity'] for p in props.values()]
        comps = [p['compactness'] for p in props.values()]
        
        # Area Hist
        axes[0].hist(areas, bins=10, color='skyblue', edgecolor='black')
        axes[0].set_title("Area Distribution")
        axes[0].set_xlabel("Pixels")
        
        # Eccentricity Hist
        axes[1].hist(eccs, bins=10, range=(0,1), color='salmon', edgecolor='black')
        axes[1].set_title("Eccentricity Distribution")
        axes[1].set_xlabel("0 (Circle) -> 1 (Line)")
        
        # Compactness Hist
        axes[2].hist(comps, bins=10, color='lightgreen', edgecolor='black')
        axes[2].set_title("Compactness Distribution")
        axes[2].set_xlabel("P^2 / Area")
        
        fig_hist.suptitle(f"Feature Statistics (Size ≥ {size_thresh})", fontsize=14)
        hist_file = f"Histograms_Size_{size_thresh}.png"
        plt.savefig(hist_file, bbox_inches='tight')
        plt.close()
        print(f"[Saved] {hist_file}")

def print_statistical_report(props, size_thresh):
    """Prints a professional summary table to console."""
    print(f"\n{'='*80}")
    print(f"ANALYSIS REPORT | Minimum Size Threshold: {size_thresh} pixels")
    print(f"{'='*80}")
    
    if not props:
        print("No components found.")
        return

    # Summary Stats
    areas = [p['area'] for p in props.values()]
    avg_area = np.mean(areas)
    avg_ecc = np.mean([p['eccentricity'] for p in props.values()])
    max_moment_error = max([p['moment_error'] for p in props.values()])
    
    print(f"Total Components: {len(props)}")
    print(f"Average Area:     {avg_area:.1f} pixels")
    print(f"Avg Eccentricity: {avg_ecc:.3f}")
    print(f"Moment Val. Err:  {max_moment_error:.1e} (Should be near 0)")
    
    print(f"\n{'ID':<4} {'Area':<8} {'Centroid':<18} {'Eccentricity':<14} {'Class':<10} {'Moment Check'}")
    print("-" * 80)
    
    for lab in sorted(props.keys()):
        p = props[lab]
        xc, yc = p['centroid']
        err_flag = "OK" if p['moment_error'] < 1e-5 else "FAIL"
        
        print(f"{lab:<4} {p['area']:<8} "
              f"({xc:>6.1f}, {yc:>6.1f})   "
              f"{p['eccentricity']:<14.4f} "
              f"{p['shape_class']:<10} "
              f"{err_flag}")
    print("-" * 80)

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    print(">>> Starting Computer Vision Pipeline...")
    
    # 1. Preprocessing
    original_img, binary_mask = load_and_preprocess("comb.img")
    
    # [cite_start]Save Hardcopies required by assignment [cite: 26]
    plt.imsave("Image_B_Original.png", original_img, cmap='gray')
    plt.imsave("Image_BT_Binary.png", binary_mask, cmap='gray')
    print("[Saved] Image_B_Original.png and Image_BT_Binary.png")

    # 2. Segmentation (CCL)
    print(">>> Running Connected Component Labeling...")
    raw_labels = label_components(binary_mask)
    
    # 3. Analysis Loop
    size_filters = [100, 500, 1000]
    
    for size in size_filters:
        # Re-label and Filter
        clean_labels, _ = filter_and_relabel(raw_labels, size)
        
        # Extract Features
        features = extract_features(clean_labels)
        
        # Generate Professional Report (Console)
        print_statistical_report(features, size)
        
        # Generate Table Image (APA Style) - RESTORED
        save_feature_table(features, size)
        
        # Visualize (Image C + Histograms)
        visualize_results(original_img, clean_labels, features, size)

    print("\n>>> Pipeline Complete. All artifacts generated.")