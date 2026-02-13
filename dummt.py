import numpy as np

def generate_comb_file(filename="image.img"):
    # 1. Define image dimensions
    size = 512
    header_size = 512
    
    # 2. Create the canvas (Foreground: White = 255)
    # Your loader inverts if sum > size/2, so we start with a white background.
    img = np.full((size, size), 255, dtype=np.uint8)
    
    # 3. Draw a perfect black circle (Object: Black = 0)
    center_x, center_y = size // 2, size // 2
    radius = 150
    
    # Create a coordinate grid
    y, x = np.ogrid[:size, :size]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    # Apply the circle
    img[mask] = 0
    
    # 4. Generate a dummy header (512 bytes)
    header = np.zeros(header_size, dtype=np.uint8)
    
    # 5. Write to file
    with open(filename, "wb") as f:
        f.write(header.tobytes())  # Write header
        f.write(img.tobytes())     # Write pixel data
    
    print(f"File '{filename}' created successfully.")

if __name__ == "__main__":
    generate_comb_file()