#Colour Detection
import cv2
import numpy as np
from rembg import remove
from PIL import Image
import pandas as pd
from collections import Counter

# Load the Excel file with colors
df = pd.read_excel('colors.xlsx')  # Ensure correct path

# Convert the data into a dictionary with Color_name
color_dict = {}
for index, row in df.iterrows():
  color_name = row['Color_name']
  hex_value = row['Hex']
  decimal_value = (int(row['D2']), int(row['D1']), int(row['D0']))  # Convert to int
  color_dict[color_name] = {'Hex': hex_value, 'Decimal': decimal_value}

# Function to find the closest color name
def closest_color(requested_color):
  min_distance = float("inf")
  closest_match = "Unknown"

  for color_name, data in color_dict.items():
    r_c, g_c, b_c = map(int, data["Decimal"])  # Convert stored values to int
    r, g, b = map(int, requested_color)  # Convert input to int
    
    # Use float subtraction to avoid overflow
    distance = ((float(r_c) - float(r)) ** 2 +
          (float(g_c) - float(g)) ** 2 +
          (float(b_c) - float(b)) ** 2)

    if distance < min_distance:
      min_distance = distance
      closest_match = color_name

  return closest_match

# Function to get average color inside a contour
def get_avg_color(c, image):
  mask = np.zeros(image.shape[:2], dtype=np.uint8)
  cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)

  # Extract only the contour pixels
  pixels = cv2.bitwise_and(image, image, mask=mask)  # Apply mask on the image
  pixels = pixels[mask == 255]  # Get only non-black pixels inside contour

  if len(pixels) == 0:
    return (0, 0, 0)  # Return black if no pixels found

  # Compute mean color inside contour
  mean_color = np.mean(pixels, axis=0)
  return tuple(map(int, mean_color[::-1]))  # Convert to (R, G, B)

# Function to detect shape
def detect_shape(c):
  approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
  sides = len(approx)
  if sides == 3:
    return "Triangle"
  elif sides == 4:
    return "Rectangle"
  elif sides == 5:
    return "Pentagon"
  elif sides > 5:
    return "Circle"
  return "Unknown"

# Load the original image
image = cv2.imread('image_25_30.jpg')  
if image is None:
  print("Error: Image not found or cannot be opened.")
  exit()

# Resize the image to 30% of original size
scale_percent = 30
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize original image
image_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Convert OpenCV image to PIL for background removal
image_pil = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))

# Remove background using rembg
output_pil = remove(image_pil)

# Convert PIL image back to OpenCV format
image_np = np.array(output_pil)
image_no_bg = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)  # Remove alpha channel

# Convert to grayscale
gray_image = cv2.cvtColor(image_no_bg, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred_image = cv2.GaussianBlur(gray_image, (11, 11), 0)

# Apply Otsu's thresholding
_, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours and label shapes
shape_image = image_resized.copy()
for contour in contours:
  if cv2.contourArea(contour) > 500:
    shape = detect_shape(contour)
    avg_color = get_avg_color(contour, image_resized)
    color_name = closest_color(avg_color)

    # Draw contour
    cv2.drawContours(shape_image, [contour], -1, (0, 255, 0), 2)

    # Get center of the contour
    M = cv2.moments(contour)
    if M["m00"] != 0:
      cx = int(M["m10"] / M["m00"])
      cy = int(M["m01"] / M["m00"])
      
      # Compute adaptive font size based on image width
      font_scale = max(0.3, image_resized.shape[1] / 500)  # Adjust for better clarity


      # Draw shape name above the contour
      cv2.putText(shape_image, shape, (cx - 30, cy - 20),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 0, cv2.LINE_AA)

      
      # Draw color name below the contour
      
      cv2.putText(shape_image, color_name, (cx - 30, cy + 20),
            cv2.FONT_HERSHEY_TRIPLEX, font_scale, (255, 0, 0), 0, cv2.LINE_AA)

# Resize the output window
cv2.namedWindow('Shape Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Shape Detection', 1000, 700)

# Display the images
cv2.imshow('Shape Detection', shape_image)
cv2.waitKey(0)
cv2.destroyAllWindows()