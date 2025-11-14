import cv2
import requests
import time
import numpy as np
from threading import Thread, Lock
from collections import deque

# Server configuration
SERVER_URL = "http://localhost:5000/video_feed"

# Thread-safe result storage
result_lock = Lock()
latest_result = {'hand_detected': False}

# Frame management
latest_frame_data = None
frame_lock = Lock()
should_send = True

def send_frames_worker():
    """Background thread to continuously send latest frame"""
    session = requests.Session()
    global should_send
    
    while should_send:
        # Get latest frame
        with frame_lock:
            frame_data = latest_frame_data
        
        if frame_data is not None:
            try:
                response = session.post(
                    SERVER_URL,
                    data=frame_data,
                    headers={'Content-Type': 'image/jpeg'},
                    timeout=0.5
                )
                
                if response.status_code == 200:
                    result = response.json()
                    with result_lock:
                        global latest_result
                        latest_result = result
            except:
                pass
        else:
            time.sleep(0.01)  # Small delay if no frame available

# Start background sender thread
sender_thread = Thread(target=send_frames_worker, daemon=True)
sender_thread.start()

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution for speed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Starting optimized webcam client...")
print(f"Server: {SERVER_URL}")
print("Press 'q' to quit, 'r' to reset")

frame_count = 0
start_time = time.time()
fps_display = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Encode frame with low quality for speed
    _, img_encoded = cv2.imencode('.jpg', frame, 
                                  [cv2.IMWRITE_JPEG_QUALITY, 50,
                                   cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    
    # Update latest frame for sender thread
    with frame_lock:
        latest_frame_data = img_encoded.tobytes()
    
    # Resize for display (make it bigger for viewing)
    display_frame = cv2.resize(frame, (640, 480))
    
    # Get latest result
    with result_lock:
        result = latest_result.copy()
    
    # Draw result on display frame
    if result.get('hand_detected', False):
        cell = result['cell']
        
        # Draw cell info
        cv2.putText(display_frame, f"Cell: Row {cell['row']}, Col {cell['col']}", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw click indicator
        if result.get('click', False):
            cv2.circle(display_frame, (30, 90), 20, (0, 0, 255), -1)
            cv2.putText(display_frame, "CLICK!", (60, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Draw distance
        distance = result.get('finger_distance', 0)
        cv2.putText(display_frame, f"Distance: {distance}", 
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Status indicator
        cv2.circle(display_frame, (620, 30), 10, (0, 255, 0), -1)
    else:
        cv2.putText(display_frame, "No hand detected", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.circle(display_frame, (620, 30), 10, (0, 0, 255), -1)
    
    # Calculate FPS
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed > 0.3:
        fps_display = frame_count / elapsed
        frame_count = 0
        start_time = time.time()
    
    # Display FPS
    cv2.putText(display_frame, f"FPS: {fps_display:.1f}", 
               (10, display_frame.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Show frame
    cv2.imshow('Hand Detection Client', display_frame)
    
    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        with result_lock:
            latest_result = {'hand_detected': False}
        print("Result reset")

# Cleanup
should_send = False
sender_thread.join(timeout=1)
cap.release()
cv2.destroyAllWindows()
print("Client stopped")