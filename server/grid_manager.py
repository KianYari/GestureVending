# grid_manager.py
import cv2

# Default: 6 Rows. Row 0 is 5 cols, Rows 1-5 are 10 cols
grid_layout = [5, 10, 10, 10, 10, 10] 

def update_grid_layout(config):
    """Regenerates the grid_layout list based on MQTT config."""
    global grid_layout
    try:
        total_rows = int(config.get('rows', 6))
        cols_double = int(config.get('cols_double', 5))
        cols_single = int(config.get('cols_single', 10))
        
        double_indices = config.get('double_row_indices', [0])
        
        new_layout = []
        for r in range(total_rows):
            if r in double_indices:
                new_layout.append(cols_double)
            else:
                new_layout.append(cols_single)
        
        grid_layout = new_layout
        print(f"Grid Layout Updated: {grid_layout}")
        
    except Exception as e:
        print(f"Error parsing grid config: {e}")

def get_grid_cell(x, y, width, height):
    global grid_layout
    num_rows = len(grid_layout)
    if num_rows == 0: return (0, 0, 1)
    
    row_height = height / num_rows
    
    # 1. Determine Row
    row = int(y / row_height)
    if row >= num_rows: row = num_rows - 1
    if row < 0: row = 0

    # 2. Determine Column based on specific configuration of THIS row
    cols_in_this_row = grid_layout[row]
    col_width = width / cols_in_this_row
    
    col = int(x / col_width)
    if col >= cols_in_this_row: col = cols_in_this_row - 1
    if col < 0: col = 0
    
    return (row, col, cols_in_this_row)

def calculate_slot_id(row, col):
    global grid_layout
    slot_id = 0
    # Sum of all slots in previous rows
    for r in range(row):
        slot_id += grid_layout[r]
    # Add current column (1-based index)
    slot_id += (col + 1)
    return slot_id - 1

def draw_grid(frame):
    global grid_layout
    height, width, _ = frame.shape
    num_rows = len(grid_layout)
    if num_rows == 0: return

    row_height = height / num_rows
    
    # Draw Horizontal lines
    for i in range(1, num_rows):
        y = int(i * row_height)
        cv2.line(frame, (0, y), (width, y), (255, 255, 255), 1)
    
    # Draw Vertical lines
    for r in range(num_rows):
        cols = grid_layout[r]
        col_width = width / cols
        y_start = int(r * row_height)
        y_end = int((r + 1) * row_height)
        
        for c in range(1, cols):
            x = int(c * col_width)
            cv2.line(frame, (x, y_start), (x, y_end), (255, 255, 255), 1)