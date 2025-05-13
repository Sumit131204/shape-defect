import requests
import cv2
import numpy as np

API_KEY = '8jhEqypwqfFY1C88e7ETneA8'
input_path = r"bluecap.jpg"
output_path = "removed_bg.png"

# --- Background Removal ---
with open(input_path, 'rb') as image_file:
    response = requests.post(
        'https://api.remove.bg/v1.0/removebg',
        files={'image_file': image_file},
        data={'size': 'auto'},
        headers={'X-Api-Key': API_KEY}
    )

if response.status_code == requests.codes.ok:
    with open(output_path, 'wb') as out:
        out.write(response.content)
    print(f"Image saved to {output_path}")
else:
    print("Error:", response.status_code, response.text)
    exit()

# --- Shape Detection Function ---
def detect_shape(contour):
    shape = "unknown"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

    if len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        shape = "square" if 0.95 <= aspect_ratio <= 1.05 else "rectangle"
    elif len(approx) > 4:
        area = cv2.contourArea(contour)
        circularity = (4 * np.pi * area) / (peri * peri)
        if circularity > 0.8:
            shape = "circle"
        else:
            shape = "ellipse/rounded"
    return shape

# --- Load and Process Image ---
image = cv2.imread("removed_bg.png", cv2.IMREAD_UNCHANGED)
img_height, img_width = image.shape[:2]

# Calibration: Assume visible field is approx 32 cm high
real_world_height_cm = 32.0
pixels_per_cm = img_height / real_world_height_cm

if image.shape[2] == 4:
    alpha_channel = image[:, :, 3]
    _, mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
else:
    print("No alpha channel found! Contour detection may not work properly.")
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

# --- Contour Detection ---
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = image[:, :, :3].copy()

# --- Analyze and Draw ---
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    shape = detect_shape(contour)

    # Convert pixel-based measurements to cm
    area_cm2 = area / (pixels_per_cm ** 2)
    x, y, w, h = cv2.boundingRect(contour)
    width_cm = w / pixels_per_cm
    height_cm = h / pixels_per_cm

    # Draw contour
    cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)

    # Label shape and size
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        if shape == "circle":
            diameter_px = (w + h) / 2
            diameter_cm = diameter_px / pixels_per_cm
            label = f"{shape}, diameter: {diameter_cm:.1f} cm"
        else:
            label = f"{shape}, {width_cm:.1f}x{height_cm:.1f} cm"

        cv2.putText(output, label, (cX - 50, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    print(f"Contour {i+1}: Shape = {shape}, Area = {area_cm2:.2f} cm², Size = {width_cm:.2f} x {height_cm:.2f} cm")

# --- Save and Show Output ---
cv2.imwrite("size_shape_detected.png", output)
cv2.imshow("Detected Shapes & Sizes", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
