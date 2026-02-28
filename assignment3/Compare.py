import numpy as np
import cv2
import os

OUTPUT_FOLDER = "assignment3_figures"

def load_img(path):
    try:
        with open(path, "rb") as f:
            f.seek(512)
            img_data = np.fromfile(f, dtype=np.uint8)
        return img_data.reshape((512, 512))
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def builtin_dual_threshold_rg(img):
    """Built-in logic for Task 3: Hysteresis-style region growing."""
    # Use Otsu to find a strong T1, and a percentage for T2
    t1, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t2 = t1 * 0.6
    
    _, strong_mask = cv2.threshold(img, t1, 255, cv2.THRESH_BINARY)
    _, weak_mask = cv2.threshold(img, t2, 255, cv2.THRESH_BINARY)
    
    # Keep weak components only if they touch a strong seed
    num_labels, labels = cv2.connectedComponents(weak_mask)
    output = np.zeros_like(img, dtype=np.uint8)
    for i in range(1, num_labels):
        component_mask = (labels == i).astype(np.uint8)
        if cv2.countNonZero(cv2.bitwise_and(strong_mask, strong_mask, mask=component_mask)) > 0:
            output[labels == i] = 255
    return output, (int(t1), int(t2))

def run_comparison():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, OUTPUT_FOLDER)
    os.makedirs(output_dir, exist_ok=True)
    
    image_names = ["test1.img", "test2.img", "test3.img"]

    for name in image_names:
        path = os.path.join(script_dir, name)
        img = load_img(path)
        if img is None: continue
        base_name = name.split('.')[0]

        # TASK 1 & 2 COMPARISON: Otsu's is the library standard for both
        # It serves as the 'ideal' peak-valley finder (Task 1) 
        # and the optimal iterative converger (Task 2).
        val_otsu, res_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # TASK 3 COMPARISON: Hysteresis/Connected Components
        res_dual, (t1, t2) = builtin_dual_threshold_rg(img)

        # ADAPTIVE (Bonus): To see how local thresholds compare to your global tasks
        res_adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # SAVING - Using consistent filenames
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_builtin_otsu_T{int(val_otsu)}.png"), res_otsu)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_builtin_dualRG_T{t1}_{t2}.png"), res_dual)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_builtin_adaptive.png"), res_adaptive)
        
        print(f"Compared {name}: Otsu T={int(val_otsu)}, Dual T1={t1}, T2={t2}")

if __name__ == "__main__":
    run_comparison()