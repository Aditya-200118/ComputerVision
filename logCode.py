import numpy as np
import matplotlib.pyplot as plt

def generate_normalized_log_plots():
    # Define the domain (x-axis)
    # Using a high resolution to ensure smooth curves and accurate convolution
    x = np.linspace(-10, 10, 1000)
    dx = x[1] - x[0] # Step size for accurate numerical integration/convolution

    # Create the Step Edge function: 0 for x < 0, 1 for x >= 0
    step_edge = np.zeros_like(x)
    step_edge[x >= 0] = 1.0

    # The standard deviations (widths) requested in the problem
    sigmas = [1, 2, 3]

    # Setup the figure for side-by-side plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for sigma in sigmas:
        # 1. Mathematically define the 1-D Gaussian
        g_x = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-x**2 / (2 * sigma**2))
        
        # 2. Derive the 1-D Normalized LoG Operator
        # Note: We use the scale-normalized version: ((sigma**2 - x**2) / sigma**2) * G(x)
        log_1d = ((sigma**2 - x**2) / sigma**2) * g_x
        
        # 3. Calculate the Response to the Step Edge
        # We convolve the step edge with the LoG operator. 
        # Multiplying by dx scales the discrete convolution to match continuous integration.
        response = np.convolve(step_edge, log_1d, mode='same') * dx
        
        # Plotting the LoG Operator
        ax1.plot(x, log_1d, label=rf'$\sigma = {sigma}$')
        
        # Plotting the Step Edge Response
        ax2.plot(x, response, label=rf'$\sigma = {sigma}$')

    # Formatting for the 1-D LoG Operator Plot
    ax1.set_title('Normalized 1-D LoG Operator', fontsize=14)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.axhline(0, color='black', linewidth=0.8) # x-axis
    ax1.axvline(0, color='black', linewidth=0.8) # y-axis
    ax1.set_xlim([-8, 8])
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Formatting for the Step Edge Response Plot
    ax2.set_title('Response to Step Edge', fontsize=14)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Amplitude', fontsize=12)
    ax2.axhline(0, color='black', linewidth=0.8) # x-axis
    ax2.axvline(0, color='black', linewidth=0.8) # y-axis
    ax2.set_xlim([-8, 8])
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    # Save the plots to a file instead of showing them
    output_filename = 'normalized_log_plots.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots successfully generated and saved as '{output_filename}' in the current directory.")

# Execute the function
if __name__ == "__main__":
    generate_normalized_log_plots()