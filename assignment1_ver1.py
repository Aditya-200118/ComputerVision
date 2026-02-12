import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import math

# ============================================================
# CONFIGURATION & STYLE
# ============================================================
# High DPI for publication quality
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 600,       # CHANGED: 600 DPI for print quality
    "savefig.dpi": 600
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
        
        # Thresholding (T=128)
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
        
        # 1. Centroid
        xc, yc = np.mean(x), np.mean(y)
        
        # Bounding Box
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        
        # 2. Central Moments
        x_prime = x - xc
        y_prime = y - yc
        
        a = np.sum(x_prime**2)
        b = 2 * np.sum(x_prime * y_prime)
        c = np.sum(y_prime**2)
        
        # 3. Orientation (Theta)
        theta = 0.5 * np.arctan2(b, a - c)
        theta_deg = np.degrees(theta) 
        
        # 4. Principal Moments
        term = np.sqrt((a - c)**2 + b**2)
        Imax = (a + c + term) / 2
        Imin = (a + c - term) / 2
        
        moment_error = abs((a + c) - (Imax + Imin))
        
        # 5. Eccentricity & Compactness
        eccentricity = np.sqrt(1 - (Imin / Imax)) if Imax > 0 else 0
        
        # Perimeter (Boundary pixel count)
        perimeter = 0
        rows, cols = labels.shape
        for py, px in coords:
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
            "orientation_deg": theta_deg,
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

def save_latex_table(props, size_thresh):
    """
    Generates a .tex file containing a booktabs-style LaTeX table code.
    This is best for inserting directly into Overleaf/LaTeX docs.
    """
    if not props:
        return

    filename = f"Table_LaTeX_Size_{size_thresh}.tex"
    
    with open(filename, "w") as f:
        # Write LaTeX Header
        f.write("% Copy and paste the following into your LaTeX document.\n")
        f.write("% Requires \\usepackage{booktabs} in preamble.\n\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Component Analysis (Min Size $\\ge$ {size_thresh})}}\n")
        f.write("\\begin{tabular}{l c c c c c l}\n")
        f.write("\\toprule\n")
        
        # Column Headers
        f.write("ID & Area & Centroid $(x, y)$ & Orient ($^\\circ$) & Eccen. & Compact & Class \\\\\n")
        f.write("\\midrule\n")
        
        # Data Rows
        for lab in sorted(props.keys()):
            p = props[lab]
            xc, yc = p['centroid']
            orient = p['orientation_deg']
            # Format row: ID & Area & (x,y) & Orient & Ecc & Comp & Class \\
            line = (f"{lab} & {p['area']} & ({xc:.1f}, {yc:.1f}) & {orient:.1f} & "
                    f"{p['eccentricity']:.3f} & {p['compactness']:.2f} & {p['shape_class']} \\\\\n")
            f.write(line)
            
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(f"\\label{{tab:components_{size_thresh}}}\n")
        f.write("\\end{table}\n")
    
    print(f"[Saved] {filename} (LaTeX code)")

def save_feature_table_image(props, size_thresh):
    """
    Generates a minimalist, publication-ready table image (PNG).
    No colors, no fancy borders.
    """
    if not props:
        return

    table_data = []
    for lab in sorted(props.keys()):
        p = props[lab]
        xc, yc = p["centroid"]
        table_data.append([
            lab,
            p["area"],
            f"({xc:.1f}, {yc:.1f})",
            f"{p['orientation_deg']:.1f}",
            f"{p['eccentricity']:.3f}",
            f"{p['compactness']:.2f}",
            p['shape_class']
        ])

    column_labels = ["ID", "Area", "Centroid", "Orient (deg)", "Eccen.", "Compact", "Class"]

    # Calculate figure size
    fig_h = len(table_data) * 0.3 + 1.0
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.axis('off')
    
    # Create Table
    table = ax.table(
        cellText=table_data,
        colLabels=column_labels,
        loc='center',
        cellLoc='center'
    )
    
    # Minimalist Style Setting
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Iterate over cells to remove colors and set thick lines for top/bottom
    for key, cell in table.get_celld().items():
        row, col = key
        cell.set_linewidth(0)         # Default no border
        cell.set_facecolor('white')   # White background only
        
        # Header Row
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_edgecolor('black')
            cell.set_linewidth(1)     # Thin line under header
            cell.visible_edges = "B"  # Only show bottom edge
        
        # Bottom of last row
        if row == len(table_data):
            cell.set_edgecolor('black')
            cell.set_linewidth(1)
            cell.visible_edges = "B" # Line at bottom of table

    # Save
    outfile = f"Table_Image_Size_{size_thresh}.png"
    plt.savefig(outfile, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"[Saved] {outfile}")

def visualize_results(original, labels, props, size_thresh):
    """Generates Image C (with overlays) and Histograms."""
    
    # --- 1. Main Visualization (Image C) ---
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create colored map
    colored = np.zeros((*labels.shape, 3))
    np.random.seed(42)
    colored[:] = [0, 0, 0] # Black bg
    
    unique_labs = list(props.keys())
    for lab in unique_labs:
        color = np.random.rand(3)
        while np.mean(color) < 0.4: color = np.random.rand(3)
        colored[labels == lab] = color

    ax.imshow(colored)
    # Removing title from image itself for cleaner paper inclusion
    # Titles are usually handled by LaTeX captions
    ax.axis('off')

    for lab, p in props.items():
        xc, yc = p["centroid"]
        min_x, min_y, max_x, max_y = p["bbox"]
        theta = p["orientation"]
        
        # Bounding Box
        rect = patches.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y,
                                 linewidth=0.8, edgecolor='white', facecolor='none', 
                                 linestyle=':', alpha=0.7)
        ax.add_patch(rect)
        
        # Axes
        axis_len = 30 
        # Major
        x1 = xc + axis_len * np.cos(theta)
        y1 = yc + axis_len * np.sin(theta)
        x2 = xc - axis_len * np.cos(theta)
        y2 = yc - axis_len * np.sin(theta)
        ax.plot([x1, x2], [y1, y2], color='yellow', linewidth=1.5, alpha=0.9)
        
        # Minor
        x3 = xc + (axis_len * 0.6) * np.cos(theta + np.pi/2)
        y3 = yc + (axis_len * 0.6) * np.sin(theta + np.pi/2)
        x4 = xc - (axis_len * 0.6) * np.cos(theta + np.pi/2)
        y4 = yc - (axis_len * 0.6) * np.sin(theta + np.pi/2)
        ax.plot([x3, x4], [y3, y4], color='red', linewidth=1.5, alpha=0.9)

        # Label
        ax.text(xc, yc, str(lab), color='cyan', fontsize=8, fontweight='bold',
                ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5, pad=0))

    outfile = f"Image_C_Size_{size_thresh}.png"
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"[Saved] {outfile}")

    # --- 2. Statistical Histograms ---
    if len(props) > 1:
        fig_hist, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        areas = [p['area'] for p in props.values()]
        eccs = [p['eccentricity'] for p in props.values()]
        comps = [p['compactness'] for p in props.values()]
        
        # Use grayscale/patterns for academic look if needed, but color is usually fine for plots
        axes[0].hist(areas, bins=10, color='gray', edgecolor='black', alpha=0.7)
        axes[0].set_title("Area Distribution")
        axes[0].set_xlabel("Pixels")
        
        axes[1].hist(eccs, bins=10, range=(0,1), color='gray', edgecolor='black', alpha=0.7)
        axes[1].set_title("Eccentricity Distribution")
        axes[1].set_xlabel("0 (Circle) -> 1 (Line)")
        
        axes[2].hist(comps, bins=10, color='gray', edgecolor='black', alpha=0.7)
        axes[2].set_title("Compactness Distribution")
        axes[2].set_xlabel("P^2 / Area")
        
        hist_file = f"Histograms_Size_{size_thresh}.png"
        plt.savefig(hist_file, bbox_inches='tight')
        plt.close()
        print(f"[Saved] {hist_file}")

# ============================================================
# NEW: DETAILED TEXT TABLE REPORT
# ============================================================

def print_detailed_report(props, size_thresh):
    """
    Prints a detailed table to console AND saves it to a text file.
    Replaces degree symbol with 'deg'.
    """
    
    # 1. Build the Table String
    lines = []
    lines.append("="*95)
    lines.append(f"DETAILED ANALYSIS REPORT | Minimum Size Threshold: {size_thresh} pixels")
    lines.append("="*95)
    
    if not props:
        lines.append("No components found matching the criteria.")
    else:
        # Header - CHANGED: 'Orient' now implies degrees without symbol
        header = f"| {'ID':<4} | {'Area':<8} | {'Centroid (X, Y)':<18} | {'Orient':<8} | {'Eccen.':<8} | {'Compact':<8} | {'Class':<12} |"
        lines.append(header)
        lines.append("-" * len(header))
        
        # Data Rows
        for lab in sorted(props.keys()):
            p = props[lab]
            xc, yc = p['centroid']
            orient = p['orientation_deg']
            
            # CHANGED: Removed degree symbol, used 'deg' in header or just number
            row = (f"| {lab:<4} | {p['area']:<8} | ({xc:>6.1f}, {yc:>6.1f})    | "
                   f"{orient:>7.1f}  | {p['eccentricity']:<8.4f} | {p['compactness']:<8.2f} | {p['shape_class']:<12} |")
            lines.append(row)
            
        lines.append("-" * len(header))
        
        # Summary Stats
        areas = [p['area'] for p in props.values()]
        avg_area = np.mean(areas)
        avg_ecc = np.mean([p['eccentricity'] for p in props.values()])
        
        lines.append(f"TOTAL COMPONENTS: {len(props)}")
        lines.append(f"AVERAGE AREA:     {avg_area:.2f} pixels")
        lines.append(f"AVG ECCENTRICITY: {avg_ecc:.4f}")
        lines.append("="*95)

    # 2. Join into a single string
    full_report = "\n".join(lines)
    
    # 3. Print to Console
    print(f"\n\n{full_report}\n")
    
    # 4. Save to File
    filename = f"Detailed_Table_Size_{size_thresh}.txt"
    with open(filename, "w") as f:
        f.write(full_report)
    print(f"[Saved] {filename}")

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    print(">>> Starting Computer Vision Pipeline...")
    
    # 1. Preprocessing
    original_img, binary_mask = load_and_preprocess("comb.img")
    
    # Save Hardcopies
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
        
        # --- REPORTING ---
        
        # 1. Console Text Table (Clean characters)
        print_detailed_report(features, size)
        
        # 2. LaTeX Code (.tex file for direct copy-paste)
        save_latex_table(features, size)
        
        # 3. Clean PNG Table (Minimalist, 600 DPI)
        save_feature_table_image(features, size)
        
        # 4. Visualizations (600 DPI)
        visualize_results(original_img, clean_labels, features, size)

    print("\n>>> Pipeline Complete. All artifacts generated.")