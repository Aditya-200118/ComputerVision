import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from matplotlib.ticker import MaxNLocator
import math

# ============================================================
# CONFIGURATION & STYLE (ACADEMIC)
# ============================================================
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 600,  # High Resolution for Print
    }
)


# ============================================================
# 1. PREPROCESSING
# ============================================================
def load_and_preprocess(path):
    """Loads raw image, strips header, returns original and binary."""
    try:
        with open(path, "rb") as f:
            _ = f.read(512)
            img = np.fromfile(f, dtype=np.uint8)

        original = img.reshape((512, 512))
        binary = (original > 128).astype(np.uint8)

        if np.sum(binary) > (binary.size // 2):
            binary = 1 - binary

        return original, binary
    except FileNotFoundError:
        print(f"Error: File {path} not found.")
        exit()


# ============================================================
# 2. SEGMENTATION (CCL)
# ============================================================
def label_components(binary_img):
    """Iterative CCL with 4-connectivity."""
    rows, cols = binary_img.shape
    labels = np.zeros((rows, cols), dtype=int)
    current_label = 1
    equivalences = defaultdict(set)

    # Pass 1
    for i in range(rows):
        for j in range(cols):
            if binary_img[i, j] == 1:
                neighbors = []
                if i > 0 and labels[i - 1, j] > 0:
                    neighbors.append(labels[i - 1, j])
                if j > 0 and labels[i, j - 1] > 0:
                    neighbors.append(labels[i, j - 1])

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

    # Pass 2
    for i in range(rows):
        for j in range(cols):
            if labels[i, j] > 0:
                labels[i, j] = label_map.get(labels[i, j], labels[i, j])

    return labels


# ============================================================
# 3. POST-PROCESSING
# ============================================================
def filter_and_relabel(labels, min_size):
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
# 4. FEATURE EXTRACTION
# ============================================================
def extract_features(labels):
    props = {}
    indices = np.unique(labels)
    indices = indices[indices != 0]

    for lab in indices:
        coords = np.argwhere(labels == lab)
        y, x = coords[:, 0], coords[:, 1]
        area = len(coords)

        xc, yc = np.mean(x), np.mean(y)
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)

        x_prime = x - xc
        y_prime = y - yc

        a = np.sum(x_prime**2)
        b = 2 * np.sum(x_prime * y_prime)
        c = np.sum(y_prime**2)

        theta = 0.5 * np.arctan2(b, a - c)
        theta_deg = np.degrees(theta)

        term = np.sqrt((a - c) ** 2 + b**2)
        Imax = (a + c + term) / 2
        Imin = (a + c - term) / 2

        # Added Elongation Ratio
        elongation = (Imax / Imin) if Imin > 0 else 0

        moment_error = abs((a + c) - (Imax + Imin))
        eccentricity = np.sqrt(1 - (Imin / Imax)) if Imax > 0 else 0

        perimeter = 0
        rows_img, cols_img = labels.shape
        for py, px in coords:
            is_edge = py == 0 or py == rows_img - 1 or px == 0 or px == cols_img - 1
            if not is_edge:
                if (
                    labels[py - 1, px] != lab
                    or labels[py + 1, px] != lab
                    or labels[py, px - 1] != lab
                    or labels[py, px + 1] != lab
                ):
                    is_edge = True
            if is_edge:
                perimeter += 1

        compactness = (perimeter**2) / area

        if eccentricity < 0.6:
            shape_class = "Compact"
        elif eccentricity < 0.9:
            shape_class = "Oval"
        else:
            shape_class = "Elongated"

        props[lab] = {
            "area": area,
            "centroid": (xc, yc),
            "bbox": (min_x, min_y, max_x, max_y),
            "orientation": theta,
            "orientation_deg": theta_deg,
            "elongation": elongation,  # Added Elongation
            "eccentricity": eccentricity,
            "perimeter": perimeter,  # Added for the new table requirement
            "compactness": compactness,
            "shape_class": shape_class,
            "moment_error": moment_error,
        }
    return props


# ============================================================
# 5. TABLES (PDF GENERATION)
# ============================================================


def save_analysis_table(props, size_thresh):
    """
    Saves the 'Analysis Table' (formerly Detailed Table) mirroring console output.
    Cols: ID, Area, Centroid, Eccentricity, Class, Moment Check
    """
    if not props:
        return

    # Added Elongation column
    column_labels = ["ID", "Area", "Centroid", "Eccentricity", "Elongation", "Class", "Moment Check"]
    table_data = []

    for lab in sorted(props.keys()):
        p = props[lab]
        xc, yc = p["centroid"]
        err_flag = "OK" if p["moment_error"] < 1e-5 else "FAIL"

        table_data.append(
            [
                lab,
                p["area"],
                f"({xc:.1f}, {yc:.1f})",
                f"{p['eccentricity']:.4f}",
                f"{p['elongation']:.2f}",  # Added value
                p["shape_class"],
                err_flag,
            ]
        )

    n_rows = len(table_data) + 1
    fig_height = max(2, n_rows * 0.4)

    fig, ax = plt.subplots(figsize=(10, fig_height)) # Slightly wider for new col
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=column_labels,
        loc="center",
        cellLoc="center",
        # Adjusted widths to fit new column
        colWidths=[0.08, 0.12, 0.22, 0.12, 0.20, 0.20, 0.20], 
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eaeaea")
            cell.set_linewidth(1.0)
        else:
            cell.set_facecolor("white")

    plt.title(
        f"Analysis Table (Size ≥ {size_thresh})", pad=10, fontsize=12, fontweight="bold"
    )

    outfile = f"Analysis_Table_Size_{size_thresh}.pdf"
    plt.savefig(outfile, format="pdf", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[Saved PDF] {outfile}")


def save_component_description_table(props, size_thresh):
    """
    Saves the 'Component Description Table' (formerly Standard Table).
    Required Cols: Area, Centroid, BBox, Orientation, Eccentricity, Perimeter, Compactness
    """
    if not props:
        return

    # Note: 'ID' is usually needed to identify rows, I will include it as the first column
    # Added Elong. column
    column_labels = [
        "ID",
        "Area",
        "Centroid",
        "Bounding Box",
        "Orient(deg)",
        "Elongation", # Added
        "Eccentricity",
        "Perimeter",
        "Compactness",
    ]
    table_data = []

    for lab in sorted(props.keys()):
        p = props[lab]
        xc, yc = p["centroid"]
        min_x, min_y, max_x, max_y = p["bbox"]

        table_data.append(
            [
                lab,
                p["area"],
                f"({xc:.1f}, {yc:.1f})",
                f"[{min_x},{min_y},{max_x},{max_y}]",
                f"{p['orientation_deg']:.1f}",
                f"{p['elongation']:.2f}", # Added value
                f"{p['eccentricity']:.3f}",
                f"{p['perimeter']}",
                f"{p['compactness']:.2f}",
            ]
        )

    n_rows = len(table_data) + 1
    fig_height = max(2, n_rows * 0.45)  # Slightly taller for density

    # Wider figure to accommodate BBox and extra columns
    fig, ax = plt.subplots(figsize=(11, fig_height)) # Slightly wider
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=column_labels,
        loc="center",
        cellLoc="center",
        # Adjusted widths for 9 columns
        colWidths=[0.05, 0.08, 0.15, 0.18, 0.12, 0.10, 0.15, 0.15, 0.15],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)  # Slightly smaller font to fit everything
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eaeaea")
            cell.set_linewidth(1.0)
        else:
            cell.set_facecolor("white")

    plt.title(
        f"Component Description Table (Size ≥ {size_thresh})",
        pad=10,
        fontsize=12,
        fontweight="bold",
    )

    outfile = f"Component_Description_Table_Size_{size_thresh}.pdf"
    plt.savefig(outfile, format="pdf", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[Saved PDF] {outfile}")


# ============================================================
# 6. VISUALIZATION & REPORTING (CONSOLE + IMAGES)
# ============================================================


def visualize_results(original, labels, props, size_thresh):
    """
    Generates Image C (with overlays) and Histograms.
    """

    # --- 1. Main Visualization (Image C) ---
    fig, ax = plt.subplots(figsize=(10, 10))

    colored = np.zeros((*labels.shape, 3))
    np.random.seed(42)
    colored[:] = [0, 0, 0]

    unique_labs = list(props.keys())
    for lab in unique_labs:
        color = np.random.rand(3)
        while np.mean(color) < 0.4:
            color = np.random.rand(3)
        colored[labels == lab] = color

    ax.imshow(colored)
    ax.set_title(
        f"Image C: Filtered Components (Size ≥ {size_thresh})\n",
        pad=15,
    )
    ax.axis("off")
    # fig.subplots_adjust(bottom=0.12)
    # fig.text(
    #     0.5, 0.02,
    #     "Visualization includes Centroids, Bounding Boxes, and Principal Axes",
    #     ha="center",
    #     fontsize=11,
    #     color="dimgray"
    # )

    for lab, p in props.items():
        xc, yc = p["centroid"]
        min_x, min_y, max_x, max_y = p["bbox"]
        theta = p["orientation"]

        rect = patches.Rectangle(
            (min_x, min_y),
            max_x - min_x,
            max_y - min_y,
            linewidth=0.8,
            edgecolor="white",
            facecolor="none",
            linestyle=":",
            alpha=0.7,
        )
        ax.add_patch(rect)

        axis_len = 30
        x1 = xc + axis_len * np.cos(theta)
        y1 = yc + axis_len * np.sin(theta)
        x2 = xc - axis_len * np.cos(theta)
        y2 = yc - axis_len * np.sin(theta)
        ax.plot([x1, x2], [y1, y2], color="yellow", linewidth=1.5, alpha=0.9)

        x3 = xc + (axis_len * 0.6) * np.cos(theta + np.pi / 2)
        y3 = yc + (axis_len * 0.6) * np.sin(theta + np.pi / 2)
        x4 = xc - (axis_len * 0.6) * np.cos(theta + np.pi / 2)
        y4 = yc - (axis_len * 0.6) * np.sin(theta + np.pi / 2)
        ax.plot([x3, x4], [y3, y4], color="red", linewidth=1.5, alpha=0.9)

        ax.text(
            xc,
            yc,
            str(lab),
            color="cyan",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.5, pad=0),
        )

    legend_elements = [
        plt.Line2D([0], [0], color="yellow", lw=2, label="Major Axis"),
        plt.Line2D([0], [0], color="red", lw=2, label="Minor Axis"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    outfile = f"Image_C_Size_{size_thresh}.png"
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"[Saved] {outfile}")

    # --- 2. Statistical Histograms ---
    if len(props) > 1:
        fig_hist, axes = plt.subplots(1, 3, figsize=(15, 4))

        areas = [p["area"] for p in props.values()]
        eccs = [p["eccentricity"] for p in props.values()]
        comps = [p["compactness"] for p in props.values()]

        axes[0].hist(areas, bins=10, color="skyblue", edgecolor="black")
        axes[0].set_title("Area Distribution")
        axes[0].set_xlabel("Pixels")
        axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].hist(eccs, bins=10, range=(0, 1), color="salmon", edgecolor="black")
        axes[1].set_title("Eccentricity Distribution")
        axes[1].set_xlabel("0 (Circle) -> 1 (Line)")
        axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[2].hist(comps, bins=10, color="lightgreen", edgecolor="black")
        axes[2].set_title("Compactness Distribution")
        axes[2].set_xlabel("P^2 / Area")
        axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))
        fig_hist.suptitle(f"Feature Statistics (Size ≥ {size_thresh})", fontsize=14)
        hist_file = f"Histograms_Size_{size_thresh}.png"
        plt.savefig(hist_file, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {hist_file}")


def print_statistical_report(props, size_thresh):
    """Prints summary table to console."""
    print(f"\n{'='*80}")
    print(f"ANALYSIS REPORT | Minimum Size Threshold: {size_thresh} pixels")
    print(f"{'='*80}")

    if not props:
        print("No components found.")
        return

    areas = [p["area"] for p in props.values()]
    avg_area = np.mean(areas)
    avg_ecc = np.mean([p["eccentricity"] for p in props.values()])
    max_moment_error = max([p["moment_error"] for p in props.values()])

    print(f"Total Components: {len(props)}")
    print(f"Average Area:     {avg_area:.1f} pixels")
    print(f"Avg Eccentricity: {avg_ecc:.3f}")
    print(f"Moment Val. Err:  {max_moment_error:.1e} (Should be near 0)")

    print(
        f"\n{'ID':<4} {'Area':<8} {'Centroid':<18} {'Eccentricity':<14} {'Elongation':<12} {'Class':<10} {'Moment Check'}"
    )
    print("-" * 80)

    for lab in sorted(props.keys()):
        p = props[lab]
        xc, yc = p["centroid"]
        err_flag = "OK" if p["moment_error"] < 1e-5 else "FAIL"

        print(
            f"{lab:<4} {p['area']:<8} "
            f"({xc:>6.1f}, {yc:>6.1f})    "
            f"{p['eccentricity']:<14.4f} "
            f"{p['elongation']:<12.2f} "
            f"{p['shape_class']:<10} "
            f"{err_flag}"
        )
    print("-" * 80)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(">>> Starting Pipeline...")

    # 1. Preprocess
    original_img, binary_mask = load_and_preprocess("comb.img")
    plt.imsave("Image_B_Original.png", original_img, cmap="gray")
    plt.imsave("Image_BT_Binary.png", binary_mask, cmap="gray")

    # 2. Segment
    raw_labels = label_components(binary_mask)

    # 3. Analyze
    size_filters = [100, 500, 1000]

    for size in size_filters:
        clean_labels, _ = filter_and_relabel(raw_labels, size)
        features = extract_features(clean_labels)

        # A. Console Output
        print_statistical_report(features, size)

        # B. PDF Tables (UPDATED NAMES & CONTENT)
        save_analysis_table(features, size)
        save_component_description_table(features, size)

        # C. Visualization (Image C + Histograms)
        visualize_results(original_img, clean_labels, features, size)

    print("\n>>> Pipeline Complete.")
