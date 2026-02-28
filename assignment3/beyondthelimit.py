import numpy as np
import matplotlib.pyplot as plt
import os
import math

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
        return img_data.reshape((512, 512)).astype(float)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        exit()

def calculate_gradients(img):
    """Calculates spatial pixel-to-pixel differences using central derivatives."""
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
    gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0
    return np.sqrt(gx**2 + gy**2)

def gaussian_pdf(x, mu, sigma):
    """Calculates the probability density of x for a given normal distribution."""
    # Add epsilon to sigma to prevent division by zero
    sigma = max(sigma, 1e-6)
    variance = sigma ** 2
    return (1.0 / np.sqrt(2 * np.pi * variance)) * np.exp(-((x - mu) ** 2) / (2 * variance))

def threshold_gmm_em(image, iterations=50):
    """
    Task 1 BEYOND THE LIMIT: Fits a 2-component Gaussian Mixture Model 
    using Expectation-Maximization (EM) from scratch.
    """
    # 1. Get Histogram
    hist = np.zeros(256, dtype=float)
    for val in image.ravel():
        hist[int(val)] += 1
    
    N = image.size
    x_vals = np.arange(256)
    
    # 2. Initialize Parameters (Assume one dark cluster, one bright cluster)
    mu1, mu2 = np.percentile(image, 25), np.percentile(image, 75)
    sig1, sig2 = 20.0, 20.0
    pi1, pi2 = 0.5, 0.5
    
    # 3. EM Algorithm Loop
    for _ in range(iterations):
        # E-Step: Calculate Responsibilities (Probabilities)
        pdf1 = pi1 * gaussian_pdf(x_vals, mu1, sig1)
        pdf2 = pi2 * gaussian_pdf(x_vals, mu2, sig2)
        
        total_pdf = pdf1 + pdf2 + 1e-10 # Prevent div by zero
        resp1 = pdf1 / total_pdf
        resp2 = pdf2 / total_pdf
        
        # M-Step: Update Parameters based on weighted histogram
        N1 = np.sum(resp1 * hist)
        N2 = np.sum(resp2 * hist)
        
        pi1 = N1 / N
        pi2 = N2 / N
        
        mu1 = np.sum(resp1 * hist * x_vals) / N1
        mu2 = np.sum(resp2 * hist * x_vals) / N2
        
        sig1 = np.sqrt(np.sum(resp1 * hist * ((x_vals - mu1) ** 2)) / N1)
        sig2 = np.sqrt(np.sum(resp2 * hist * ((x_vals - mu2) ** 2)) / N2)
        
    # 4. Find the threshold: the exact intersection of the two Gaussians
    # We scan between the two means to find where pdf1 crosses pdf2
    T = int(mu1)
    min_mu, max_mu = int(min(mu1, mu2)), int(max(mu1, mu2))
    
    # Calculate final scaled distributions to find intersection
    final_pdf1 = pi1 * gaussian_pdf(x_vals, mu1, sig1)
    final_pdf2 = pi2 * gaussian_pdf(x_vals, mu2, sig2)
    
    for x in range(min_mu, max_mu):
        if (final_pdf1[x] - final_pdf2[x]) * (final_pdf1[x+1] - final_pdf2[x+1]) < 0:
            T = x
            break

    binary_image = (image >= T).astype(np.uint8) * 255
    return binary_image, T, (mu1, mu2, sig1, sig2)

def threshold_iterative(image, t0_estimate):
    """Task 2: Remains unchanged as Isodata is mathematically pure."""
    T_old = t0_estimate
    while True:
        G1 = image[image > T_old]
        G2 = image[image <= T_old]
        mu1 = np.mean(G1) if len(G1) > 0 else 0
        mu2 = np.mean(G2) if len(G2) > 0 else 0
        T_new = (mu1 + mu2) / 2.0
        if abs(T_new - T_old) < 1.0:
            break
        T_old = T_new
    return (image >= int(T_new)).astype(np.uint8) * 255, int(T_new)

def threshold_gradient_region_growing(image, sigma_fg, max_grad_allowance):
    """
    Task 3 BEYOND THE LIMIT: Adds Gradient Magnitude boundaries.
    Will not grow across sharp edges, even if pixel intensity is valid.
    """
    # 1. Mathematical High Threshold
    T_old = np.mean(image)
    for _ in range(50):
        G1 = image[image > T_old]
        G2 = image[image <= T_old]
        mu1 = np.mean(G1) if len(G1) > 0 else 0
        mu2 = np.mean(G2) if len(G2) > 0 else 0
        T_new = (mu1 + mu2) / 2.0
        if abs(T_new - T_old) < 1.0:
            break
        T_old = T_new
        
    T_H = int(T_new)
    T_L = max(0, int(T_H - sigma_fg))
    
    # 2. Calculate Gradient Constraints
    grad_mag = calculate_gradients(image)
    
    # 3. Dual-Constrained Region Growing
    rows, cols = image.shape
    binary_image = np.zeros((rows, cols), dtype=np.uint8)
    seeds = []
    
    for r in range(rows):
        for c in range(cols):
            if image[r, c] >= T_H:
                binary_image[r, c] = 255
                seeds.append((r, c))
                
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    while seeds:
        r, c = seeds.pop()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                # Core Upgrade: Check Intensity AND Ensure we aren't crossing a sharp edge
                if binary_image[nr, nc] == 0:
                    intensity_valid = image[nr, nc] >= T_L
                    edge_valid = grad_mag[nr, nc] < max_grad_allowance
                    
                    if intensity_valid and edge_valid:
                        binary_image[nr, nc] = 255
                        seeds.append((nr, nc))
                        
    return binary_image, T_H, T_L

if __name__ == "__main__":
    output_dir = "assignment3_beyond_limit"
    os.makedirs(output_dir, exist_ok=True)
    
    image_configs = {
        "test1.img": {
            "t0_estimate": 87.4,       
            "sigma_fg": 42.62,
            "max_grad": 25.0  # Stop growing if pixel jump is > 25
        },
        "test2.img": {
            "t0_estimate": 60.3,       
            "sigma_fg": 48.28,
            "max_grad": 30.0 
        },
        "test3.img": {
            "t0_estimate": 110.3,      
            "sigma_fg": 31.98,
            "max_grad": 15.0  # Tighter edge constraint for blurry image
        }
    }
    
    for file, config in image_configs.items():
        if not os.path.exists(file):
            continue
            
        print(f"\nProcessing {file} Beyond Theoretical Limits...")
        img = load_and_preprocess(file)
        base_name = file.split('.')[0]
        
        # GMM Task 1
        res_gmm, t_gmm, stats = threshold_gmm_em(img)
        mu1, mu2, s1, s2 = stats
        print(f"  GMM EM Threshold: {t_gmm} | Gaussians: N1({mu1:.1f}, {s1:.1f}), N2({mu2:.1f}, {s2:.1f})")
        plt.imsave(f"{output_dir}/{base_name}_gmm_peakiness.png", res_gmm, cmap='gray')
        
        # Iterative Task 2
        res_iterative, t_iter = threshold_iterative(img, config["t0_estimate"])
        print(f"  Iterative Threshold: {t_iter}")
        plt.imsave(f"{output_dir}/{base_name}_iterative.png", res_iterative, cmap='gray')
        
        # Gradient Task 3
        res_grad_rg, th, tl = threshold_gradient_region_growing(img, config["sigma_fg"], config["max_grad"])
        print(f"  Gradient Region Growing: High={th}, Low={tl}, Max Grad={config['max_grad']}")
        plt.imsave(f"{output_dir}/{base_name}_gradient_rg.png", res_grad_rg, cmap='gray')

    print(f"\nProcess complete. Check the '{output_dir}' folder.")