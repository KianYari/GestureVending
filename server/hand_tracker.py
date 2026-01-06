import cv2
import mediapipe as mp
import time
import config
import grid_manager
import mqtt_handler
import json

class HandProcessor:
    def __init__(self):
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0  
        )
        self.drawing_utils = mp.solutions.drawing_utils
        
        # Interaction States
        self.last_selected_cell = None
        self.last_cell_select_time = 0
        self.last_click_time = 0
        
        # FPS statistics
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0

        # Initialize grid with default or empty config initially
        # (Ideally, mqtt_handler calls grid_manager.update_grid_layout when config arrives)
        grid_manager.update_grid_layout({"NumOfSlots": 60, "NumOfDoubleSlots": 5})

    def process(self, frame):
        # FPS Calculation
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

        frame_height, frame_width, _ = frame.shape

        # Resize to 640px width if larger
        if frame_width > 640:
            scale = 640 / frame_width
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
            frame_height, frame_width, _ = frame.shape

        # MediaPipe Processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = self.hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        result = {
            'hand_detected': False,
            'cell': None,
            'click': False,
            'timestamp': time.time(),
            'fps': round(self.fps, 1)
        }

        grid_manager.draw_grid(frame)

        if hands:
            hand = hands[0]
            landmarks = hand.landmark

            # --- Logic: Index Finger (Pointer) ---
            index_finger = landmarks[8]
            
            # 1. BOUNDARY CHECK: Ignore interaction if finger is off-screen
            # This prevents "wrapping" (top selecting bottom) and IndexErrors
            if 0.0 <= index_finger.x <= 1.0 and 0.0 <= index_finger.y <= 1.0:
                
                # 2. CLAMPING: Ensure pixel coordinates are strictly within frame
                frame_index_x = int(index_finger.x * frame_width)
                frame_index_y = int(index_finger.y * frame_height)
                frame_index_x = max(0, min(frame_width - 1, frame_index_x))
                frame_index_y = max(0, min(frame_height - 1, frame_index_y))

                current_cell = grid_manager.get_grid_cell(frame_index_x, frame_index_y, frame_width, frame_height)
                
                # Verify we actually got a valid cell back
                if current_cell:
                    slot_id = grid_manager.calculate_slot_id(current_cell[0], current_cell[1])

                    # Selection Publishing (Hover)
                    if current_cell != self.last_selected_cell and (current_time - self.last_cell_select_time) > config.CELL_SELECT_COOLDOWN:
                        payload = json.dumps({
                            "SelectItem": slot_id
                        })
                        print(f"Cell Select: R{current_cell[0]} C{current_cell[1]} (Slot {slot_id}) -> MQTT Payload: {payload}")
                        mqtt_handler.publish(config.MQTT_TOPIC_SELECTING, payload)

                        self.last_selected_cell = current_cell
                        self.last_cell_select_time = current_time

                    # --- Logic: Thumb (For Click) ---
                    thumb = landmarks[4]
                    
                    mcp_index = landmarks[5]
                    mcp_pinky = landmarks[17]
                    
                    pinch_distance = ((index_finger.x - thumb.x)**2 + (index_finger.y - thumb.y)**2)**0.5
                    palm_size = ((mcp_index.x - mcp_pinky.x)**2 + (mcp_index.y - mcp_pinky.y)**2)**0.5
                    normalized_distance = pinch_distance / palm_size if palm_size > 0 else 1.0
                    
                    
                    # 3. CLAMPING THUMB: Prevent math errors if thumb goes wild
                    frame_thumb_x = int(thumb.x * frame_width)
                    frame_thumb_y = int(thumb.y * frame_height)
                    frame_thumb_x = max(0, min(frame_width - 1, frame_thumb_x))
                    frame_thumb_y = max(0, min(frame_height - 1, frame_thumb_y))

                    distance = ((frame_index_x - frame_thumb_x)**2 + (frame_index_y - frame_thumb_y)**2)**0.5
                    is_clicking = normalized_distance < config.CLICK_THRESHOLD

                    # Click Publishing
                    if is_clicking and (current_time - self.last_click_time) > config.CLICK_COOLDOWN:
                        payload = json.dumps({
                            "Goto_Payment": True
                        })
                        print(f"CLICK at Slot {slot_id} -> MQTT Payload: {payload}")
                        mqtt_handler.publish(config.MQTT_TOPIC_SELECTING, payload)
                        self.last_click_time = current_time

                    # Populate Result
                    result['hand_detected'] = True
                    result['cell'] = {'row': current_cell[0], 'col': current_cell[1], 'total_cols': current_cell[2], 'slot_id': slot_id}
                    result['click'] = is_clicking
                    result['finger_distance'] = int(distance)

                    # --- Drawing Visuals ---
                    cv2.circle(frame, (frame_index_x, frame_index_y), 8, (0, 255, 255), -1)
                    cv2.circle(frame, (frame_thumb_x, frame_thumb_y), 8, (0, 255, 255), -1)
                    cv2.line(frame, (frame_index_x, frame_index_y), (frame_thumb_x, frame_thumb_y), (255, 0, 255), 2)

                    # Highlight Cell
                    row, col, total_cols = current_cell
                    num_rows = len(grid_manager.grid_layout)
                    row_height = frame_height / num_rows
                    col_width = frame_width / total_cols
                    
                    x1 = int(col * col_width)
                    y1 = int(row * row_height)
                    x2 = int((col + 1) * col_width)
                    y2 = int((row + 1) * row_height)

                    color = (0, 255, 0) if is_clicking else (255, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                    if is_clicking:
                        cv2.putText(frame, "CLICK!", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Finger is out of bounds - Reset selection or handle gracefully
                self.last_selected_cell = None

        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (frame_width - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return result, frame