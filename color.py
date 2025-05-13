import cv2
import numpy as np
from rembg import remove
from PIL import Image
import pandas as pd
from collections import Counter

# Load the Excel file with colors
df = pd.read_excel('colour2.xlsx')  # Make sure the file is in the same folder or provide the correct path

# Convert the data into a dictionary with Color_name
color_dict = {}
for index, row in df.iterrows():
    color_family = row['Color_name']  # Updated to 'Color_name'
    hex_value = row['Hex']
    # Now D2 (R), D1 (G), D0 (B) are in separate columns
    decimal_value = (int(row['D2']), int(row['D1']), int(row['D0']))
    color_dict[color_family] = {'Hex': hex_value, 'Decimal': decimal_value}


# Function to find the closest color family
def closest_color(requested_color):
    min_colors = {}
    for family, data in color_dict.items():
        r_c, g_c, b_c = map(int, data['Decimal'])  # <-- convert to int here
        distance = ((r_c - int(requested_color[0])) ** 2 +
                    (g_c - int(requested_color[1])) ** 2 +
                    (b_c - int(requested_color[2])) ** 2)
        min_colors[distance] = family
    return min_colors[min(min_colors.keys())]



# Function to get color family (exact match or closest match)
def get_color_family(rgb):
    try:
        # Find closest family match
        return closest_color(rgb)  # Find closest match from the Family column
    except ValueError:
        return "Unknown"  # Default in case something goes wrong


# Load the original image
image = cv2.imread("image_10.jpg")
if image is None:
    print("Error: Image not found or cannot be opened.")
    exit()

# Resize the image to a smaller size (30% of original size)
scale_percent = 70
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

# Function to detect shape
def detect_shape(c):
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    sides = len(approx)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    
    if sides == 3:
        return "Triangle"
    elif sides == 4:
        # Check if rectangle or square
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05:
            return "Square"
        else:
            return "Rectangle"
    elif sides == 5:
        return "Pentagon"
    else:
        # Check circularity
        if perimeter == 0:
            return "Unknown"
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if 0.7 < circularity < 1.2:
            return "Circle"
        return "Unknown"\

def get_dominant_color(c, image):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, (255), thickness=cv2.FILLED)

    masked_image = cv2.bitwise_and(image, image, mask=mask)
    pixels = masked_image[mask == 255]

    if len(pixels) == 0:
        return (0, 0, 0)

    # print(f"\nPixels inside contour (Total: {len(pixels)}):")
    # for i, pixel in enumerate(pixels[:50]):  # Just printing the first 50 for sanity
    #     b, g, r = pixel  # OpenCV is BGR
    #     print(f"Pixel {i}: R={r}, G={g}, B={b}")
    # if len(pixels) > 50:
    #     print("...")  # Skip printing all pixels for large shapes

    # Filter out very bright pixels
    filtered_pixels = []
    for pixel in pixels:
        if not (pixel[0] > 220 and pixel[1] > 220 and pixel[2] > 220):  # Not almost white
            filtered_pixels.append(tuple(pixel))

    if len(filtered_pixels) == 0:
        filtered_pixels = [tuple(p) for p in pixels]  # fallback if all were filtered

    # Find most common pixel
    most_common_pixel = Counter(filtered_pixels).most_common(1)[0][0]

    return most_common_pixel




# Draw contours and label shapes
shape_image = image_resized.copy()
for contour in contours:
    if cv2.contourArea(contour) > 500:  # Filter out small contours
        shape = detect_shape(contour)
        color = get_dominant_color(contour, image_no_bg)
        color_rgb = tuple(reversed(color))
        color_family = get_color_family(color_rgb)  
        print(f"Shape: {shape} -> Matched Colour: {color_family },{color_rgb}")


        cv2.drawContours(shape_image, [contour], -1, (0, 255, 0), 2)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Draw shape name above the contour
            #cv2.putText(shape_image, shape, (cx - 20, cy - 10),
             #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            # Draw family name below the contour
            cv2.putText(shape_image, color_family, (cx - 80, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Resize the output window (to be larger)
height, width = image_resized.shape[:2]
cv2.namedWindow('Shape Detection', cv2.WINDOW_NORMAL)  # Create window that can be resized
cv2.resizeWindow('Shape Detection', width, height)  # Set custom window size (width x height)

# Display the images
cv2.imshow('Shape Detection', shape_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
