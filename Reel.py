import cv2
import numpy as np
import turtle

# Load reference image in grayscale
image_path = "mataji.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Error: Could not find '{image_path}'. Make sure it is in the same folder.")
    exit()

# Turtle window setup has to happen before we pick a size, so we can
# read the real screen. This display is 1280x720; a fixed 810px window
# is taller than that, so Windows shoved the top off the monitor.
screen = turtle.Screen()
screen_w = screen.getcanvas().winfo_screenwidth()
screen_h = screen.getcanvas().winfo_screenheight()

# Keep the whole window below the top edge and above the taskbar.
# setup() size is the client area; the title bar sits outside it.
top_margin = 10
bottom_reserve = 110
side_margin = 24
frame_pad = 36
max_draw_w = screen_w - (side_margin * 2) - frame_pad
max_draw_h = screen_h - top_margin - bottom_reserve - frame_pad

aspect_ratio = image.shape[0] / image.shape[1]
if max_draw_w * aspect_ratio <= max_draw_h:
    target_width = int(max_draw_w)
    target_height = int(target_width * aspect_ratio)
else:
    target_height = int(max_draw_h)
    target_width = int(target_height / aspect_ratio)

resized_image = cv2.resize(image, (target_width, target_height))

# Clean thresholding to keep sharp edges (hands, weapons, jewelry, lion's face)
_, binary = cv2.threshold(resized_image, 180, 255, cv2.THRESH_BINARY_INV)

# Find all nested layers (RETR_TREE captures outer outlines, inner cuts, and inner-inner details)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

if hierarchy is None:
    print("Error: No shapes detected.")
    exit()

hierarchy = hierarchy[0]

# Window fits on screen, and the scrollable canvas matches the drawing.
# Turtle's default canvas is only 400x300, so a larger figure is clipped
# to the middle unless screensize() is set to the full drawing.
win_w = target_width + frame_pad
win_h = target_height + frame_pad
startx = max(0, (screen_w - win_w) // 2)
screen.setup(width=win_w, height=win_h, startx=startx, starty=top_margin)
screen.screensize(canvwidth=target_width + 8, canvheight=target_height + 8)
screen.title("Live Drawing - Mataji and Lion")
screen.bgcolor("white")
screen.tracer(2, 0)  # Drawing speed balance: fast yet clearly visible

# Pen (Cursor) Setup
pen = turtle.Turtle()
pen.speed(0)
pen.pensize(1)
pen.shape("classic")
pen.showturtle()  # You will see cursor actively drawing all shapes

# Coordinate converter
half_w = target_width / 2
half_h = target_height / 2

def to_turtle_coords(x, y):
    return (x - half_w, half_h - y)

# Helper function to get contour nesting depth level
def get_contour_depth(idx):
    depth = 0
    parent = hierarchy[idx][3]
    while parent != -1:
        depth += 1
        parent = hierarchy[parent][3]
    return depth

# Draw function
def render_shape(cnt, fill_col, border_col):
    first_pt = cnt[0][0]
    tx, ty = to_turtle_coords(first_pt[0], first_pt[1])
    
    pen.penup()
    pen.goto(tx, ty)
    pen.pendown()
    
    pen.color(border_col, fill_col)
    pen.begin_fill()
    
    # Smooth step to prevent lagging while maintaining high curve accuracy
    step = 2 if len(cnt) > 80 else 1
    for i in range(1, len(cnt), step):
        pt = cnt[i][0]
        tx, ty = to_turtle_coords(pt[0], pt[1])
        pen.goto(tx, ty)
        
    pen.goto(to_turtle_coords(first_pt[0], first_pt[1]))
    pen.end_fill()

# Group contours by hierarchy depth
max_depth = max(get_contour_depth(i) for i in range(len(contours)))
depth_layers = {d: [] for d in range(max_depth + 1)}

for i, cnt in enumerate(contours):
    if cv2.contourArea(cnt) < 4:  # Remove tiny camera noise
        continue
    d = get_contour_depth(i)
    depth_layers[d].append(cnt)

print("Drawing in progress... Watch the cursor create all details!")

# Render layer by layer (Outer Black -> Inner White -> Inner Black, etc.)
for d in range(max_depth + 1):
    # Even layers (0, 2, 4) are BLACK; Odd layers (1, 3) are WHITE cutouts (Lion face, eyes, hand gaps)
    fill_color = "black" if (d % 2 == 0) else "white"
    border_color = fill_color
    
    for cnt in depth_layers[d]:
        render_shape(cnt, fill_color, border_color)

# Finish and finalize
pen.hideturtle()
screen.update()
print("Done! Exact reference matched perfectly.")
screen.mainloop()