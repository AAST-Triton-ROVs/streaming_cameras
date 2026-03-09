import socket
import struct
import pickle
import cv2
import numpy as np
import time
import threading
import Hybrid_Filter

# Global variable to store the latest frame data
# Dictionary: {cam_name: jpeg_bytes}
latest_frames = {}
frame_lock = threading.Lock()
running = True

# UI state globals
fullscreen_cam_name = None
mouse_click_pos = None

def on_mouse(event, x, y, flags, param):
    global mouse_click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_click_pos = (x, y)

def receive_thread(host, port):
    global latest_frames, running
    
    while running:
        try:
            print(f"[*] Connecting to {host}:{port}...")
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(5)
            client_socket.connect((host, port))
            client_socket.settimeout(None)
            print(f"[*] Connected!")

            data = b""
            payload_size = struct.calcsize(">L")

            while running:
                # Read message size
                while len(data) < payload_size:
                    packet = client_socket.recv(4096)
                    if not packet:
                        break
                    data += packet
                
                if len(data) < payload_size:
                    break

                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack(">L", packed_msg_size)[0]

                # Read full message
                while len(data) < msg_size:
                    packet = client_socket.recv(4096)
                    if not packet:
                        break
                    data += packet
                
                if len(data) < msg_size:
                    break

                msg_data = data[:msg_size]
                data = data[msg_size:]

                try:
                    # Decoding happens in this thread to offload main thread
                    # But we just store the raw bytes or the decoded frames?
                    # Storing raw bytes is faster for network thread.
                    # Let's decode here? No, let's decode in main thread to keep this super fast.
                    # Or decode here to avoid main thread lag?
                    # Best: Receive raw bytes, main thread decodes only latest.
                    frames = pickle.loads(msg_data)
                    
                    with frame_lock:
                        latest_frames = frames
                        
                except Exception as e:
                    print(f"[!] Error decoding frames: {e}")
                    continue

            print("[*] Connection closed by server.")
            client_socket.close()
            
        except (socket.error, ConnectionRefusedError) as e:
            print(f"[!] Connection failed: {e}")
            print("[*] Retrying in 2 seconds...")
            time.sleep(2)
        except Exception as e:
            print(f"[!] Unexpected error: {e}")
            time.sleep(2)

def calculate_grid_size(num_cams):
    if num_cams <= 1:
        return 1, 1
    elif num_cams <= 2:
        return 1, 2
    elif num_cams <= 4:
        # 2x2 grid for 3 or 4 cams
        return 2, 2
    else:
        # 2x3 grid for 5 or 6 cams
        return 2, 3

def main_loop():
    global latest_frames, running
    global fullscreen_cam_name, mouse_click_pos
    
    host = '192.168.1.100'
    port = 8001
    
    # Start receiver thread
    t = threading.Thread(target=receive_thread, args=(host, port))
    t.daemon = True
    t.start()
    
    window_name = "Station Receiver - Dynamic Grid"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, on_mouse)

    enable_filter = True # Default state

    while running:
        current_frames = {}
        with frame_lock:
            # Copy to avoid holding lock while processing
            current_frames = latest_frames.copy()
            
        # Process pending click
        click = None
        if mouse_click_pos:
            click = mouse_click_pos
            mouse_click_pos = None
        
        if not current_frames:
            # Show "Waiting for video..." text
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for signal...", (100, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(window_name, blank)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                running = False
                break
            continue

        frames_dict = {}
        for cam_name, jpeg_bytes in current_frames.items():
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                # Apply filter independently to avoid bounding box issues
                if enable_filter:
                    frame = Hybrid_Filter.apply_hybrid(frame)
                frames_dict[cam_name] = frame

        num_cams = len(frames_dict)
        if num_cams == 0:
            continue

        cam_names = sorted(list(frames_dict.keys()))

        # Get window size, default 1920x1080 
        try:
            win_rect = cv2.getWindowImageRect(window_name)
            screen_w, screen_h = win_rect[2], win_rect[3]
            if screen_w <= 0 or screen_h <= 0:
                screen_w, screen_h = 1920, 1080
        except Exception:
            screen_w, screen_h = 1920, 1080

        grid_image = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        drawn_rects = {} # Label -> (x, y, w, h)

        if fullscreen_cam_name and fullscreen_cam_name in frames_dict:
            # FOCUS LAYOUT
            main_frame = frames_dict[fullscreen_cam_name]
            other_names = [n for n in cam_names if n != fullscreen_cam_name]
            
            main_w = int(screen_w * 0.7)
            main_h = screen_h
            
            side_w = (screen_w - main_w) // 2
            right_side_w = screen_w - main_w - side_w
            
            # Draw main frame centered
            img_h, img_w = main_frame.shape[:2]
            aspect = img_w / img_h
            target_aspect = main_w / main_h
            
            if aspect > target_aspect:
                new_w = main_w
                new_h = int(main_w / aspect)
            else:
                new_h = main_h
                new_w = int(main_h * aspect)
                
            res_main = cv2.resize(main_frame, (new_w, new_h))
            x_m = side_w + (main_w - new_w) // 2
            y_m = (main_h - new_h) // 2
            
            grid_image[y_m:y_m+new_h, x_m:x_m+new_w] = res_main
            drawn_rects[fullscreen_cam_name] = (x_m, y_m, new_w, new_h)
            
            cv2.putText(grid_image, str(fullscreen_cam_name), (x_m + 10, y_m + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw sidebars
            left_names = other_names[::2]
            right_names = other_names[1::2]
            
            # Left Sidebar
            if len(left_names) > 0:
                side_h_per = screen_h // len(left_names)
                for i, name in enumerate(left_names):
                    side_frame = frames_dict[name]
                    img_h, img_w = side_frame.shape[:2]
                    aspect = img_w / img_h
                    target_aspect = side_w / side_h_per
                    
                    if aspect > target_aspect:
                        new_w = side_w
                        new_h = int(side_w / aspect)
                    else:
                        new_h = side_h_per
                        new_w = int(side_h_per * aspect)
                        
                    res_side = cv2.resize(side_frame, (new_w, new_h))
                    x_s = (side_w - new_w) // 2
                    y_s = i * side_h_per + (side_h_per - new_h) // 2
                    
                    grid_image[y_s:y_s+new_h, x_s:x_s+new_w] = res_side
                    drawn_rects[name] = (x_s, y_s, new_w, new_h)
                    cv2.putText(grid_image, str(name), (x_s + 10, y_s + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Right Sidebar
            if len(right_names) > 0:
                side_h_per = screen_h // len(right_names)
                for i, name in enumerate(right_names):
                    side_frame = frames_dict[name]
                    img_h, img_w = side_frame.shape[:2]
                    aspect = img_w / img_h
                    target_aspect = right_side_w / side_h_per
                    
                    if aspect > target_aspect:
                        new_w = right_side_w
                        new_h = int(right_side_w / aspect)
                    else:
                        new_h = side_h_per
                        new_w = int(side_h_per * aspect)
                        
                    res_side = cv2.resize(side_frame, (new_w, new_h))
                    x_s = side_w + main_w + (right_side_w - new_w) // 2
                    y_s = i * side_h_per + (side_h_per - new_h) // 2
                    
                    grid_image[y_s:y_s+new_h, x_s:x_s+new_w] = res_side
                    drawn_rects[name] = (x_s, y_s, new_w, new_h)
                    cv2.putText(grid_image, str(name), (x_s + 10, y_s + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        else:
            # GRID LAYOUT
            rows, cols = calculate_grid_size(num_cams)
            cell_w = screen_w // cols
            cell_h = screen_h // rows
            
            for i, name in enumerate(cam_names):
                r = i // cols
                c = i % cols
                frame = frames_dict[name]
                
                img_h, img_w = frame.shape[:2]
                aspect = img_w / img_h
                target_aspect = cell_w / cell_h
                
                if aspect > target_aspect:
                    new_w = cell_w
                    new_h = int(cell_w / aspect)
                else:
                    new_h = cell_h
                    new_w = int(cell_h * aspect)
                    
                res_cell = cv2.resize(frame, (new_w, new_h))
                x_c = c * cell_w + (cell_w - new_w) // 2
                y_c = r * cell_h + (cell_h - new_h) // 2
                
                grid_image[y_c:y_c+new_h, x_c:x_c+new_w] = res_cell
                drawn_rects[name] = (x_c, y_c, new_w, new_h)
                cv2.putText(grid_image, str(name), (x_c + 10, y_c + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(window_name, grid_image)

        # Handle click resolution
        if click:
            cx, cy = click
            clicked_cam = None
            for name, (rx, ry, rw, rh) in drawn_rects.items():
                if rx <= cx <= rx + rw and ry <= cy <= ry + rh:
                    clicked_cam = name
                    break
            
            if clicked_cam:
                if fullscreen_cam_name == clicked_cam:
                    # Return to normal grid
                    fullscreen_cam_name = None
                else:
                    # Fullscreen this specific cam
                    fullscreen_cam_name = clicked_cam

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
            break
        elif key == ord('c'):
            enable_filter = not enable_filter
            print(f"[*] Filter: {'ON' if enable_filter else 'OFF'}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
