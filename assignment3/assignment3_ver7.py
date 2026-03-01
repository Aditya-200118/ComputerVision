import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "figure.dpi": 600,
    }
)


def load_and_preprocess(path):
    try:
        with open(path, "rb") as f:
            _ = f.read(512)
            img_data = np.fromfile(f, dtype=np.uint8)
        return img_data.reshape((512, 512))
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()


def calculate_stats(image):

    mean_val = np.mean(image)
    median_val = np.median(image)
    std_val = np.std(image)

    if std_val > 0:
        skewness = np.mean(((image - mean_val) / std_val) ** 3)
        kurtosis = np.mean(((image - mean_val) / std_val) ** 4) - 3
    else:
        skewness = 0
        kurtosis = 0

    k_norm = np.tanh(kurtosis)

    t0_estimate = (mean_val + median_val) / 2.0

    min_distance = int(max(40, std_val * 0.6))

    min_w, max_w = 3, 15
    window_float = max_w - ((k_norm + 1) / 2.0) * (max_w - min_w)
    window_size = int(round(window_float))

    if window_size % 2 == 0:
        window_size += 1

    s_norm = min(1.0, abs(skewness))
    dynamic_multiplier = max(0.3, 1.0 - s_norm)

    T_temp = t0_estimate
    while True:
        G1 = image[image > T_temp]
        G2 = image[image <= T_temp]
        mu1 = np.mean(G1) if len(G1) > 0 else 0
        mu2 = np.mean(G2) if len(G2) > 0 else 0

        T_new = (mu1 + mu2) / 2.0
        if abs(T_new - T_temp) < 1.0:
            break
        T_temp = T_new

    fg_pixels = image[image > T_temp]
    sigma_fg = np.std(fg_pixels) if len(fg_pixels) > 0 else 0

    raw_prominence = 0.05 + 0.02 * k_norm
    dynamic_prominence = max(0.02, min(0.10, raw_prominence))

    return (
        min_distance,
        t0_estimate,
        sigma_fg,
        window_size,
        dynamic_multiplier,
        dynamic_prominence,
    )


def threshold_peakiness(image, min_distance, win_size, height_ratio=0.05):

    hist = np.zeros(256, dtype=int)
    for val in image.ravel():
        hist[val] += 1

    window = win_size
    smoothed_hist = np.zeros(256, dtype=float)
    for i in range(256):
        start = max(0, i - window)
        end = min(255, i + window)
        smoothed_hist[i] = np.mean(hist[start : end + 1])

    max_hist_val = np.max(smoothed_hist)
    height_thereshold = max_hist_val * height_ratio

    peaks = []
    for i in range(1, 255):
        if (
            smoothed_hist[i] > smoothed_hist[i - 1]
            and smoothed_hist[i] > smoothed_hist[i + 1]
        ):
            if smoothed_hist[i] > height_thereshold:
                peaks.append(i)

    # best_peak_pair = None
    best_valley = 0
    max_peakiness = -1

    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            p1 = peaks[i]
            p2 = peaks[j]

            if abs(p1 - p2) < min_distance:
                continue

            valley_idx = p1 + np.argmin(smoothed_hist[p1 : p2 + 1])
            valley_height = smoothed_hist[valley_idx]

            h_p1 = smoothed_hist[p1]
            h_p2 = smoothed_hist[p2]

            if valley_height == 0:
                peakiness = float("inf")
            else:
                peakiness = min(h_p1, h_p2) / valley_height

            if peakiness > max_peakiness:
                max_peakiness = peakiness
                # best_peak_pair = (p1, p2)
                best_valley = valley_idx

    T = best_valley if best_valley > 0 else int(np.mean(image))
    binary_image = (image >= T).astype(np.uint8) * 255
    return binary_image, T


def threshold_iterative(image, t0_estimate, epsilon=1.0):

    T_old = t0_estimate
    while True:
        G1 = image[image > T_old]
        G2 = image[image <= T_old]
        mu1 = np.mean(G1) if len(G1) > 0 else 0
        mu2 = np.mean(G2) if len(G2) > 0 else 0
        T_new = (mu1 + mu2) / 2.0

        if abs(T_new - T_old) < epsilon:
            break
        T_old = T_new

    T_final = int(T_new)
    return (image >= T_final).astype(np.uint8) * 255, T_final


def threshold_dual_region_growing(image, sigma_fg, multiplier=0.5):
    T_old = np.mean(image)
    while True:
        G1 = image[image > T_old]
        G2 = image[image <= T_old]
        T_new = (np.mean(G1) + np.mean(G2)) / 2.0
        if abs(T_new - T_old) < 1.0:
            break
        T_old = T_new
    T_H = int(T_new)

    T_L = max(0, int(T_H - (sigma_fg * multiplier)))

    rows, cols = image.shape
    binary_image = np.zeros((rows, cols), dtype=np.uint8)

    seed_r, seed_c = np.where(image >= T_H)
    seeds = deque(zip(seed_r, seed_c))

    for r, c in seeds:
        binary_image[r, c] = 255

    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while seeds:
        r, c = seeds.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if binary_image[nr, nc] == 0 and image[nr, nc] >= T_L:
                    binary_image[nr, nc] = 255
                    seeds.append((nr, nc))

    return binary_image, T_H, T_L


if __name__ == "__main__":
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    files = ["test1.img", "test2.img", "test3.img"]

    for file in files:
        if not os.path.exists(file):
            continue

        img = load_and_preprocess(file)
        min_dist, t0_seed, sigma_fg, win_size, dyn_mult, dyn_pro = calculate_stats(img)
        base_name = file.split(".")[0]

        res_peak, t_peak = threshold_peakiness(img, min_dist, win_size, dyn_pro)
        plt.imsave(f"{output_dir}/{base_name}_peakiness.png", res_peak, cmap="gray")

        res_iter, t_iter = threshold_iterative(img, t0_seed)
        plt.imsave(f"{output_dir}/{base_name}_iterative.png", res_iter, cmap="gray")

        res_dual, th, tl = threshold_dual_region_growing(img, sigma_fg, dyn_mult)
        plt.imsave(f"{output_dir}/{base_name}_dual_region.png", res_dual, cmap="gray")