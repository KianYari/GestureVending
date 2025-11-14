import cv2
import mediapipe as mp
import pyautogui
import time

# Disable PyAutoGUI failsafe for smoother operation
pyautogui.FAILSAFE = False

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Configure hand detector for single hand and better performance
hand_detector = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

# Smoothing variables
smooth_x, smooth_y = 0, 0
smoothing_factor = 0.5

# Click detection variables
last_click_time = 0
click_cooldown = 0.5
click_threshold = 40

# Grid configuration
grid_rows = 6
first_row_cols = 5
other_row_cols = 10

# Cell selection tracking
last_selected_cell = None
cell_select_cooldown = 0.3
last_cell_select_time = 0

def get_grid_cell(x, y, width, height):
    """Determine which grid cell contains the given coordinates"""
    row_height = height / grid_rows
    
    # Determine row
    row = int(y / row_height)
    if row >= grid_rows:
        row = grid_rows - 1
    
    # Determine column based on row
    if row == 0:
        # First row has 5 columns
        col_width = width / first_row_cols
        col = int(x / col_width)
        if col >= first_row_cols:
            col = first_row_cols - 1
        return (row, col, first_row_cols)
    else:
        # Other rows have 10 columns
        col_width = width / other_row_cols
        col = int(x / col_width)
        if col >= other_row_cols:
            col = other_row_cols - 1
        return (row, col, other_row_cols)

def draw_grid(frame, frame_width, frame_height, selected_cell=None):
    """Draw the grid overlay on the frame"""
    row_height = frame_height / grid_rows
    
    # Draw horizontal lines
    for i in range(grid_rows + 1):
        y = int(i * row_height)
        cv2.line(frame, (0, y), (frame_width, y), (0, 255, 0), 2)
    
    # Draw vertical lines for first row (5 columns)
    col_width_first = frame_width / first_row_cols
    for i in range(first_row_cols + 1):
        x = int(i * col_width_first)
        y_end = int(row_height)
        cv2.line(frame, (x, 0), (x, y_end), (0, 255, 0), 2)
    
    # Draw vertical lines for other rows (10 columns)
    col_width_other = frame_width / other_row_cols
    for i in range(other_row_cols + 1):
        x = int(i * col_width_other)
        y_start = int(row_height)
        cv2.line(frame, (x, y_start), (x, frame_height), (0, 255, 0), 2)
    
    # Highlight selected cell
    if selected_cell:
        row, col, num_cols = selected_cell
        
        if row == 0:
            col_width = frame_width / first_row_cols
        else:
            col_width = frame_width / other_row_cols
        
        x1 = int(col * col_width)
        y1 = int(row * row_height)
        x2 = int((col + 1) * col_width)
        y2 = int((row + 1) * row_height)
        
        # Draw highlighted rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw cell label
        cell_label = f"R{row}C{col}"
        cv2.putText(frame, cell_label, (x1 + 10, y1 + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    current_cell = None
    
    if hands:
        hand = hands[0]
        drawing_utils.draw_landmarks(frame, hand)
        landmarks = hand.landmark
        
        # Get index finger tip (landmark 8)
        index_finger = landmarks[8]
        index_x = index_finger.x * screen_width
        index_y = index_finger.y * screen_height
        
        # Draw circle on index finger
        frame_index_x = int(index_finger.x * frame_width)
        frame_index_y = int(index_finger.y * frame_height)
        cv2.circle(frame, (frame_index_x, frame_index_y), 10, (0, 255, 255), -1)
        
        # Get current cell for index finger position
        current_cell = get_grid_cell(frame_index_x, frame_index_y, frame_width, frame_height)
        
        # # Smooth cursor movement
        # smooth_x = smooth_x + (index_x - smooth_x) * smoothing_factor
        # smooth_y = smooth_y + (index_y - smooth_y) * smoothing_factor
        
        # Move cursor with boundary checking
        # try:
        #     pyautogui.moveTo(smooth_x, smooth_y, duration=0)
        # except:
        #     pass
        
        # Get thumb tip (landmark 4)
        thumb = landmarks[4]
        thumb_x = thumb.x * screen_width
        thumb_y = thumb.y * screen_height
        
        # Draw circle on thumb
        frame_thumb_x = int(thumb.x * frame_width)
        frame_thumb_y = int(thumb.y * frame_height)
        cv2.circle(frame, (frame_thumb_x, frame_thumb_y), 10, (0, 255, 255), -1)
        
        # Calculate distance between thumb and index finger
        distance = ((index_x - thumb_x)**2 + (index_y - thumb_y)**2)**0.5
        
        # Click detection with cooldown
        current_time = time.time()
        if distance < click_threshold and (current_time - last_click_time) > click_cooldown:
            print(f"Click detected at cell: Row {current_cell[0]}, Column {current_cell[1]}")
            # pyautogui.click()
            last_click_time = current_time
            
            # Visual feedback
            cv2.putText(frame, "CLICK!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 3)
        
        # Cell selection logging with cooldown
        if current_cell != last_selected_cell and (current_time - last_cell_select_time) > cell_select_cooldown:
            print(f"Selected cell: Row {current_cell[0]}, Column {current_cell[1]} (Total cols in row: {current_cell[2]})")
            last_selected_cell = current_cell
            last_cell_select_time = current_time
        
        # Display distance for debugging
        cv2.putText(frame, f"Distance: {int(distance)}", (10, frame_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw the grid overlay
    draw_grid(frame, frame_width, frame_height, current_cell)
    
    cv2.imshow('Hand Mouse with Grid', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()