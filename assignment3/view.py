import numpy as np
import cv2
import os

def load_raw_img(path, size=512, header_size=512):
    """Loads raw .img files with a specific header offset."""
    try:
        with open(path, "rb") as f:
            f.seek(header_size)
            img_data = np.fromfile(f, dtype=np.uint8)
        return img_data.reshape((size, size))
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def save_originals():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "assignment3_figures")
    os.makedirs(output_dir, exist_ok=True)
    
    image_names = ["test1.img", "test2.img", "test3.img"]
    
    for name in image_names:
        path = os.path.join(script_dir, name)
        img = load_raw_img(path)
        if img is not None:
            save_path = os.path.join(output_dir, f"{name.split('.')[0]}_original.png")
            cv2.imwrite(save_path, img)
            print(f"Saved: {save_path}")

if __name__ == "__main__":
    save_originals()