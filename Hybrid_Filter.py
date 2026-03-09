import cv2
import numpy as np
import sys

def apply_clahe(frame):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to the L-channel of the LAB color space.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def apply_pure_red_recovery(frame):
    """
    Applies Logic for Physics-based Red Channel Recovery.
    Optimized for performance using integer math and OpenCV functions.
    """
    # Calculate channel means using optimized OpenCV function
    # cv2.mean returns (b, g, r, a)
    b_mean, g_mean, r_mean, _ = cv2.mean(frame)

    # Avoid division by zero
    if r_mean == 0: r_mean = 0.001
    if b_mean == 0: b_mean = 0.001

    # Calculate gains
    gain_r = g_mean / r_mean
    gain_b = g_mean / b_mean
    
    # Clip Red gain to avoid extreme noise
    gain_r = np.clip(gain_r, 1.0, 4.0)

    # Split channels
    b, g, r = cv2.split(frame)

    # Apply gains using optimized linear transform (src * alpha + beta)
    # This handles scaling and saturation cast to uint8 efficiently
    r = cv2.convertScaleAbs(r, alpha=gain_r)
    b = cv2.convertScaleAbs(b, alpha=gain_b)
    
    return cv2.merge((b, g, r))

def apply_hybrid(frame):
    """
    Applies the Hybrid Method: 
    1. Red Channel Recovery
    2. CLAHE Contrast Enhancement
    """
    # Step 1: Recover Color
    color_recovered = apply_pure_red_recovery(frame)
    # Step 2: Enhance Contrast
    final_output = apply_clahe(color_recovered)
    return final_output

def main():
    source = 0
    # Check if a video file path was passed as an argument
    if len(sys.argv) > 1:
        source = sys.argv[1]
        print(f"[*] Opening source: {source}")
        cap = cv2.VideoCapture(source)
    else:
        print("[*] Opening Webcam (trying index 1 then 0)...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(f"[!] Error: Could not open source: {source}")
        return

    print("[*] Running Hybrid Color Filter. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[*] End of stream or error reading frame.")
            break
        
        # Resize for performance and better screen fit if 4K
        # If it's 4K video, 640x480 might be too small, let's try 1280x720 or keep original aspect ratio if possible
        # But for stability let's stick to a manageable fixed size or percentage.
        # Given the user has a 4K video, let's resize to a nice HD viewing size like 1280x720 
        # to ensure it fits on screen but maintains detail.
        
        # Calculate aspect ratio
        height, width = frame.shape[:2]
        
        # Resize to fit side-by-side on a typical 1920x1080 screen
        # Max width per frame should be 1920 / 2 = 960. Let's use 800 for safety margin.
        target_width = 800
        if width > target_width:
            scale = target_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Apply Hybrid 
        hybrid_frame = apply_hybrid(frame)

        # Describe the images
        cv2.putText(frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(hybrid_frame, "Hybrid Filter", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Stack images horizontally
        comparison = np.hstack((frame, hybrid_frame))
        
        window_name = "Original vs Hybrid Filter"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, comparison)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
