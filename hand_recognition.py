from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import time
from threading import Lock

app = Flask(__name__)

# Configure hand detector
hand_detector = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

drawing_utils = mp.solutions.drawing_utils

# Grid configuration
grid_rows = 6
first_row_cols = 5
other_row_cols = 10

# Thread-safe frame storage
frame_lock = Lock()
latest_frame = None
latest_result = None

# Cell selection tracking
last_selected_cell = None
cell_select_cooldown = 0.3
last_cell_select_time = 0

last_click_time = 0
click_cooldown = 0.5
click_threshold = 20

def get_grid_cell(x, y, width, height):
    """Determine which grid cell contains the given coordinates"""
    row_height = height / grid_rows
    
    row = int(y / row_height)
    if row >= grid_rows:
        row = grid_rows - 1
    
    if row == 0:
        col_width = width / first_row_cols
        col = int(x / col_width)
        if col >= first_row_cols:
            col = first_row_cols - 1
        return (row, col, first_row_cols)
    else:
        col_width = width / other_row_cols
        col = int(x / col_width)
        if col >= other_row_cols:
            col = other_row_cols - 1
        return (row, col, other_row_cols)

def process_frame(frame):
    """Process frame and detect hand pointing"""
    global last_selected_cell, last_cell_select_time, last_click_time

    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    
    result = {
        'hand_detected': False,
        'cell': None,
        'click': False,
        'timestamp': time.time()
    }
    
    if hands:
        hand = hands[0]
        landmarks = hand.landmark
        
        # Get index finger tip (landmark 8)
        index_finger = landmarks[8]
        frame_index_x = int(index_finger.x * frame_width)
        frame_index_y = int(index_finger.y * frame_height)
        
        # Get current cell
        current_cell = get_grid_cell(frame_index_x, frame_index_y, frame_width, frame_height)
        current_time = time.time()
        if current_cell != last_selected_cell and (current_time - last_cell_select_time) > cell_select_cooldown:
            print(f"Selected cell: Row {current_cell[0]}, Column {current_cell[1]} (Total cols in row: {current_cell[2]})")
            last_selected_cell = current_cell
            last_cell_select_time = current_time
        
        # Get thumb tip (landmark 4)
        thumb = landmarks[4]
        frame_thumb_x = int(thumb.x * frame_width)
        frame_thumb_y = int(thumb.y * frame_height)
        
        # Calculate distance between thumb and index finger
        distance = ((frame_index_x - frame_thumb_x)**2 + 
                   (frame_index_y - frame_thumb_y)**2)**0.5
        
        current_time = time.time()
        if distance < click_threshold and (current_time - last_click_time) > click_cooldown:
            print(f"Click detected at cell: Row {current_cell[0]}, Column {current_cell[1]}")
            last_click_time = current_time
        
        result['hand_detected'] = True
        result['cell'] = {
            'row': current_cell[0],
            'col': current_cell[1],
            'total_cols': current_cell[2]
        }
        result['click'] = distance < click_threshold
        result['finger_distance'] = int(distance)
        
        # Draw on frame for visualization
        cv2.circle(frame, (frame_index_x, frame_index_y), 10, (0, 255, 255), -1)
        cv2.circle(frame, (frame_thumb_x, frame_thumb_y), 10, (0, 255, 255), -1)
        
        if result['click']:
            cv2.putText(frame, "CLICK!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 3)
    
    return result, frame

@app.route('/video_feed', methods=['POST'])
def video_feed():
    """Receive frame from ESP32 and process it"""
    global latest_frame, latest_result
    
    try:
        # Get image data from request
        nparr = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Process frame
        result, processed_frame = process_frame(frame)
        
        # Store latest results
        with frame_lock:
            latest_frame = processed_frame
            latest_result = result
        
        # Log detection
        if result['hand_detected']:
            cell = result['cell']
            print(f"Hand detected at Row {cell['row']}, Col {cell['col']} "
                  f"{'[CLICK]' if result['click'] else ''}")
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/latest_result', methods=['GET'])
def get_latest_result():
    """Get the latest detection result"""
    with frame_lock:
        if latest_result is None:
            return jsonify({'error': 'No data available'}), 404
        return jsonify(latest_result), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': time.time()}), 200

if __name__ == '__main__':
    # Run on all interfaces so ESP32 can access it
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)