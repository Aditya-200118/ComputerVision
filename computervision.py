import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import erf

output_dir = "figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_fig(name):
    plt.savefig(os.path.join(output_dir, f"{name}.png"), dpi=300, bbox_inches='tight')
    print(f"Saved {name}.png to {output_dir}")

def plot_gaussians():
    x = np.linspace(-5, 5, 500)
    def g(x, sigma):
        return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-x**2 / (2 * sigma**2))

    plt.figure(figsize=(8, 5))
    plt.plot(x, g(x, 0.5), 'b', label=r'$\sigma=0.5$ (Narrow)')
    plt.plot(x, g(x, 1.0), 'r', label=r'$\sigma=1.0$ (Medium)')
    plt.plot(x, g(x, 1.5), 'gray', label=r'$\sigma=1.5$ (Wide)', alpha=0.7)
    
    # Adding discrete markers similar to the whiteboard
    x_ticks = [0, 1, 2, 3]
    for xt in x_ticks:
        plt.vlines(xt, 0, g(xt, 0.5), colors='black', linestyles='--', linewidth=0.8)
        
    plt.title("Gaussian Kernels $G(x; \sigma)$")
    plt.xlabel("x")
    plt.ylabel("Weight")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    save_fig("IMG_6511")
    plt.close()

def plot_step_edge():
    x = np.linspace(-5, 5, 1000)
    y = np.where(x < 0, 1, 3) # Intensity I1 to I2
    
    plt.figure(figsize=(8, 4))
    plt.step(x, y, where='post', color='black', linewidth=2)
    plt.axvline(0, color='gray', linestyle='--')
    plt.text(0.2, 2, 'Step Edge', fontsize=12)
    plt.text(-4, 1.2, '$I_1$', fontsize=12)
    plt.text(4, 3.2, '$I_2$', fontsize=12)
    plt.title(r"Input Signal $I(x)$")
    plt.ylim(0, 4)
    save_fig("IMG_6512_top")
    plt.close()

def plot_convolved_edge():
    x = np.linspace(-5, 5, 500)
    def convolved(x, sigma, I1=1, I2=3):
        return I1 + (I2 - I1) * 0.5 * (1 + erf(x / (sigma * np.sqrt(2))))

    plt.figure(figsize=(8, 5))
    plt.plot(x, convolved(x, 0.4), 'r', label='Small $\sigma$ (Sharper)')
    plt.plot(x, convolved(x, 1.2), 'b', label='Large $\sigma$ (Smoother)')
    
    plt.axhline(1, color='black', linestyle=':', alpha=0.3)
    plt.axhline(3, color='black', linestyle=':', alpha=0.3)
    plt.axvline(0, color='gray', alpha=0.2)
    
    plt.title(r"Convolved Signal $I(x) \ast G(x; \sigma)$")
    plt.xlabel("x")
    plt.ylabel("Intensity")
    plt.legend()
    save_fig("IMG_6512_bottom")
    plt.close()

if __name__ == "__main__":
    plot_gaussians()
    plot_step_edge()
    plot_convolved_edge()