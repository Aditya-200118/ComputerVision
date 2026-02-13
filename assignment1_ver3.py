import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

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


def label_components(binary_img):
    """Iterative CCL with 4-connectivity."""
    rows, cols = binary_img.shape
    labels = np.zeros((rows, cols), dtype=int)
    current_label = 1
    equivalences = defaultdict(set)

    # Pass 1 Raster Scan
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

        # Elongation Ratio
        elongation = (Imax / Imin) if Imin > 0 else 0
        
        # Moment Error Check
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

        props[lab] = {
            "area": area,
            "centroid": (xc, yc),
            "bbox": (min_x, min_y, max_x, max_y),
            "orientation": theta,
            "orientation_deg": theta_deg,
            "elongation": elongation,
            "eccentricity": eccentricity,
            "perimeter": perimeter,
            "compactness": compactness,
            "moment_error": moment_error,
        }
    return props


def save_component_description_table(props, size_thresh):

    if not props:
        return

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
    fig_height = max(2, n_rows * 0.45)  

    fig, ax = plt.subplots(figsize=(11, fig_height)) 
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=column_labels,
        loc="center",
        cellLoc="center",
        colWidths=[0.05, 0.08, 0.15, 0.18, 0.12, 0.10, 0.15, 0.15, 0.15],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)  
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


def visualize_results(labels, props, size_thresh):
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


if __name__ == "__main__":
    print("Starting\n")

    # 1. Loading and Preprocess
    original_img, binary_mask = load_and_preprocess("comb.img")
    plt.imsave("Image_B_Original.png", original_img, cmap="gray")
    plt.imsave("Image_BT_Binary.png", binary_mask, cmap="gray")

    # Segment/Label
    raw_labels = label_components(binary_mask)

    # 3. Analyze
    size_filters = [100, 500, 1000]

    for size in size_filters:
        clean_labels, _ = filter_and_relabel(raw_labels, size)
        features = extract_features(clean_labels)

        save_component_description_table(features, size)

        visualize_results(clean_labels, features, size)

    print("Ended")
