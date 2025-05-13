import cv2
import numpy as np
from rembg import remove
from PIL import Image
import math

# Camera parameters
CAMERA_DISTANCE = 300  # mm
IMAGE_WIDTH_PIXELS = 1280
SENSOR_WIDTH_MM = 4.6  
FOCAL_LENGTH_MM = 35  

# Load image
image = cv2.imread('img_book.jpg')
if image is None:
    print("Error: Image not found or cannot be opened.")
    exit()

# Store original dimensions
orig_height, orig_width = image.shape[:2]

# Resize image for processing
scale_percent = 80
resized_width = int(orig_width * scale_percent / 100)
resized_height = int(orig_height * scale_percent / 100)
dim = (resized_width, resized_height)
image_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Convert to PIL and remove background
image_pil = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
output_pil = remove(image_pil)

# Convert back to OpenCV
image_np = np.array(output_pil)
image_no_bg = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

# Convert to grayscale and apply thresholding
gray_image = cv2.cvtColor(image_no_bg, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (11, 11), 0)
_, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Function to calculate real-world size
def calculate_real_size(pixel_size):
    sensor_size = (pixel_size * SENSOR_WIDTH_MM) / IMAGE_WIDTH_PIXELS
    return (CAMERA_DISTANCE * sensor_size) / FOCAL_LENGTH_MM

# Function to calculate Euclidean distance
def euclidean_distance(pt1, pt2):
    return math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

# Process contours
shape_image = image.copy()
for contour in contours:
    if cv2.contourArea(contour) > 500:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Scale box points to original size
        box = np.array([(int(p[0] * orig_width / resized_width), int(p[1] * orig_height / resized_height)) for p in box])
        
        # Draw rectangle
        cv2.drawContours(shape_image, [box], 0, (0, 255, 0), 2)

        # Calculate pixel dimensions
        side1_px = euclidean_distance(box[0], box[1])
        side2_px = euclidean_distance(box[1], box[2])

        # Convert to real-world size
        side1_mm = calculate_real_size(side1_px)
        side2_mm = calculate_real_size(side2_px)
        
        # Get center
        cx, cy = np.mean(box, axis=0).astype(int)

        # Display measurements
        cv2.putText(shape_image, f"L: {side1_mm:.2f} mm", (cx - 50, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(shape_image, f"B: {side2_mm:.2f} mm", (cx - 50, cy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# Resize output image for display
display_width = 600  
display_height = int((display_width / orig_width) * orig_height)  
shape_image_resized = cv2.resize(shape_image, (display_width, display_height))

# Display result in a small window
cv2.namedWindow('Shape Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Shape Detection', 600, 400)  # Set a smaller window size
cv2.imshow('Shape Detection', shape_image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
