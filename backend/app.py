from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
import math
from rembg import remove
import uuid
import traceback
import imutils
import pandas as pd  # Add import for pandas to read Excel file
import time
import json
from sklearn.cluster import KMeans
from enhanced_color_detection import EnhancedColorDetector
from collections import Counter
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the enhanced color detector at app startup
enhanced_color_detector = EnhancedColorDetector()

# Global settings dictionary
app_settings = {
    'object_distance': 300.0,  # Default value for distance-based calculations
    'field_height_cm': 30.0    # Default value for field height in cm (used in pixels_per_cm calculations)
}

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Camera parameters (calibrated for accurate size measurement)
CAMERA_DISTANCE = 300.0  # mm (distance from camera to object)
IMAGE_WIDTH_PIXELS = 1280  # Standard width for processing
SENSOR_WIDTH_MM = 4.8  # Typical sensor width for smartphone cameras
FOCAL_LENGTH_MM = 35.0  # Typical focal length

# Calibration factor to improve accuracy (adjust based on testing)
# This factor compensates for systematic errors in the measurement
CALIBRATION_FACTOR = 1.02  # Slightly adjusted based on testing with known objects

# Load color data from Excel file
try:
    color_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'colour2.xlsx')
    color_df = pd.read_excel(color_data_path)
    
    # Print debug info
    print(f"Excel file columns: {color_df.columns.tolist()}")
    
    # Convert the data into a dictionary for faster lookup
    color_dict = {}
    for index, row in color_df.iterrows():
        try:
            # Use the correct column names found in the Excel file
            color_name = row['Color_name'] if 'Color_name' in row else f"Color_{index}"
            hex_value = row['Hex'] if 'Hex' in row else ""
            
            # The color components are in D2 (R), D1 (G), D0 (B) columns
            r = int(row['D2']) if 'D2' in row and pd.notna(row['D2']) else 0
            g = int(row['D1']) if 'D1' in row and pd.notna(row['D1']) else 0 
            b = int(row['D0']) if 'D0' in row and pd.notna(row['D0']) else 0
            
            decimal_value = (r, g, b)
            color_dict[color_name] = {'Hex': hex_value, 'Decimal': decimal_value}
        except Exception as ex:
            print(f"Error loading color row {index}: {ex}")
            continue
    
    print(f"Successfully loaded {len(color_dict)} colors from Excel file")
    
    # Print a few sample color entries for debugging
    sample_colors = list(color_dict.items())[:5]
    print("Sample colors:")
    for name, data in sample_colors:
        print(f"  {name}: {data}")
    
except Exception as e:
    print(f"Error loading color data: {str(e)}")
    # Provide basic colors as fallback
    color_dict = {
        'Red': {'Hex': '#FF0000', 'Decimal': (255, 0, 0)},
        'Green': {'Hex': '#00FF00', 'Decimal': (0, 255, 0)},
        'Blue': {'Hex': '#0000FF', 'Decimal': (0, 0, 255)},
        'Yellow': {'Hex': '#FFFF00', 'Decimal': (255, 255, 0)},
        'Orange': {'Hex': '#FFA500', 'Decimal': (255, 165, 0)},
        'Purple': {'Hex': '#800080', 'Decimal': (128, 0, 128)},
        'Brown': {'Hex': '#A52A2A', 'Decimal': (165, 42, 42)},
        'Black': {'Hex': '#000000', 'Decimal': (0, 0, 0)},
        'White': {'Hex': '#FFFFFF', 'Decimal': (255, 255, 255)},
        'Gray': {'Hex': '#808080', 'Decimal': (128, 128, 128)}
    }
    print(f"Using {len(color_dict)} basic colors as fallback")

# Function to get dominant color inside a contour
def get_avg_color(c, image):
    """
    Calculate the dominant color inside a contour using collection counting with improved natural color detection.
    
    Parameters:
    c (numpy.ndarray): Contour points
    image (numpy.ndarray): Image in BGR format
    
    Returns:
    tuple: RGB color tuple
    """
    # Create mask for the contour
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
    
    # Extract only the contour pixels
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    pixels = masked_image[mask == 255]
    
    if len(pixels) == 0:
        return (0, 0, 0)  # Return black if no pixels found
    
    # Convert to HSV for better filtering
    hsv_pixels = cv2.cvtColor(np.array([pixels]), cv2.COLOR_BGR2HSV)[0]
    
    # Filter out very bright pixels (likely background or reflections)
    # and very dark pixels (likely shadows)
    filtered_indices = []
    for i, (pixel, hsv) in enumerate(zip(pixels, hsv_pixels)):
        # Filter out white/very bright pixels
        if all(val > 220 for val in pixel):
            continue
        
        # Filter out very dark pixels (potential shadows)
        # V in HSV represents brightness (0-255)
        if hsv[2] < 20:  # Very dark
            continue
            
        filtered_indices.append(i)
    
    if not filtered_indices:
        # Fallback: if all pixels were filtered, use all non-white pixels
        filtered_indices = [i for i, p in enumerate(pixels) if not all(val > 220 for val in p)]
        
    if not filtered_indices:
        # Last resort: use all pixels
        filtered_indices = range(len(pixels))
    
    filtered_pixels = [tuple(pixels[i]) for i in filtered_indices]
    
    # Get most common colors using Counter
    color_counts = Counter(filtered_pixels)
    common_colors = color_counts.most_common(5)  # Get top 5 most common colors
    
    if len(common_colors) > 1:
        # For natural tones (like wood), sometimes the most common color might be too dark
        # due to shadows. Try to select a good representative color from the top common ones.
        
        # Convert to HSV for better color comparison
        hsv_common = []
        for color, count in common_colors:
            color_array = np.uint8([[color]])
            hsv = cv2.cvtColor(color_array, cv2.COLOR_BGR2HSV)[0][0]
            hsv_common.append((color, hsv, count))
        
        # Sort by value (brightness) descending, but weighted by count
        # This helps find a color that is both common and not too dark
        brightness_weighted = sorted(hsv_common, 
                                    key=lambda x: (x[1][2] * min(1.0, x[2]/common_colors[0][1])), 
                                    reverse=True)
        
        # Pick the dominant color (not too dark, not too saturated)
        for color, hsv, _ in brightness_weighted:
            # Skip overly saturated colors (usually not natural wood tones)
            if hsv[1] > 240 and hsv[2] > 100:
                continue
            # Convert BGR to RGB (OpenCV uses BGR by default)
            return tuple(map(int, color[::-1]))  # Reverse BGR to RGB
        
        # Fallback to most common
        return tuple(map(int, common_colors[0][0][::-1]))  # Convert BGR to RGB
    else:
        # If there's only one common color, use it
        # Convert BGR to RGB (OpenCV uses BGR by default)
        return tuple(map(int, common_colors[0][0][::-1]))  # Reverse BGR to RGB

# Function to find the closest color name
def closest_color(requested_color):
    """
    Find the closest color name for a given RGB value using improved matching algorithm
    for natural tones like wood.
    
    Parameters:
    requested_color (tuple): RGB color tuple
    
    Returns:
    str: Closest matching color name
    """
    if not color_dict:
        return "Color data unavailable"
    
    r, g, b = requested_color
    
    # Handle common color ranges directly before database lookup
    # This ensures more intuitive color names for certain RGB ranges
    
    # Pink/Red/Magenta detection for values like (196, 34, 79)
    if r > 150 and g < 100 and b < 120 and r > g and r > b:
        if r > 200 and g < 50 and b < 50:
            return "Red"
        elif r > 180 and g < 100 and b > 50 and b < 120:
            return "Crimson"
        elif r > 180 and g < 80 and b > 100:
            return "Magenta"
        elif r > 170 and g < 70 and b < 100:
            return "Ruby"
        else:
            return "Pink"
    
    # Specific handling for (196, 34, 79) - the color in the screenshot
    if 180 <= r <= 210 and 20 <= g <= 50 and 60 <= b <= 100:
        return "Ruby"
    
    # Special handling for wood-like colors (browns and beiges)
    # Check if it's potentially a brown/wood tone
    is_brownish = (r > g > b or r > b > g) and r > 60 and g > 30 and r-b > 20
    
    if is_brownish:
        # Find wood tones or browns in our color dictionary
        wood_colors = {}
        for family, data in color_dict.items():
            if "wood" in family.lower() or "brown" in family.lower() or "beige" in family.lower() or "tan" in family.lower():
                r_c, g_c, b_c = map(int, data['Decimal'])
                distance = ((r_c - r) ** 2 + (g_c - g) ** 2 + (b_c - b) ** 2)
                wood_colors[distance] = family
        
        # If we found some wood/brown colors, use the closest one
        if wood_colors:
            return wood_colors[min(wood_colors.keys())]
    
    # For other colors or as fallback, use our standard approach
    
    # Convert requested color to HSV for better comparison
    rgb_array = np.uint8([[[r, g, b]]])
    hsv_requested = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)[0][0]
    
    # Prepare to store weighted distances
    weighted_distances = {}
    
    # Compare with each color in our dictionary
    for family, data in color_dict.items():
        try:
            r_c, g_c, b_c = map(int, data['Decimal'])
            
            # Simple RGB distance for quick filtering
            rgb_distance = ((r_c - r) ** 2 + (g_c - g) ** 2 + (b_c - b) ** 2)
            
            # Only compute HSV distance for colors that are reasonably close in RGB space
            if rgb_distance < 30000:  # Threshold to avoid unnecessary calculations
                # Convert reference color to HSV
                rgb_ref = np.uint8([[[r_c, g_c, b_c]]])
                hsv_ref = cv2.cvtColor(rgb_ref, cv2.COLOR_RGB2HSV)[0][0]
                
                # Calculate HSV components' differences
                h_diff = min(abs(hsv_ref[0] - hsv_requested[0]), 180 - abs(hsv_ref[0] - hsv_requested[0])) / 180.0
                s_diff = abs(hsv_ref[1] - hsv_requested[1]) / 255.0
                v_diff = abs(hsv_ref[2] - hsv_requested[2]) / 255.0
                
                # HSV distance with more weight on hue for better color distinction
                if hsv_requested[1] > 50:  # If it has significant saturation, hue is important
                    hsv_distance = (h_diff * 3) + s_diff + v_diff
                else:
                    # For less saturated colors, value (brightness) is more important
                    hsv_distance = h_diff + s_diff + (v_diff * 3)
                
                # Combined distance (weighted mix of RGB and HSV)
                weighted_distance = (rgb_distance / 30000) * 0.5 + hsv_distance * 0.5
                weighted_distances[weighted_distance] = family
            else:
                # For distant colors, just store the RGB distance
                weighted_distances[rgb_distance / 30000 + 1] = family  # Adding 1 to ensure it's higher than HSV distances
                
        except Exception as e:
            print(f"Error processing color {family}: {e}")
            continue
            
    # Return the closest color
    if weighted_distances:
        return weighted_distances[min(weighted_distances.keys())]
    
    # Fallback to pure RGB distance if HSV comparison fails
    min_colors = {}
    for family, data in color_dict.items():
        r_c, g_c, b_c = map(int, data['Decimal'])
        distance = ((r_c - r) ** 2 + (g_c - g) ** 2 + (b_c - b) ** 2)
        min_colors[distance] = family
    
    return min_colors[min(min_colors.keys())]

# Function to remove background
def remove_background(input_path, output_path):
    """Removes background from the input image and saves the output using rembg library."""
    try:
        print(f"[DEBUG] Starting background removal for {input_path}")
        
        # Use rembg library
        print(f"[DEBUG] Using rembg library for {input_path}")
        input_img = Image.open(input_path)
        
        # Add error handling for corrupt images
        try:
            # Verify the image can be processed
            input_img.verify()
            # Need to reopen after verify
            input_img = Image.open(input_path)
        except Exception as img_error:
            print(f"[ERROR] Input image verification failed: {str(img_error)}")
            # Try to repair by converting to RGB first
            try:
                input_img = Image.open(input_path).convert('RGB')
                print("[DEBUG] Successfully repaired image by converting to RGB")
            except:
                raise Exception(f"Image is corrupt and could not be repaired: {str(img_error)}")
        
        # Process with rembg
        output_img = remove(input_img)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert RGBA to RGB if saving as JPEG
        if output_path.lower().endswith(('.jpg', '.jpeg')):
            print("[DEBUG] Converting RGBA to RGB for JPEG output")
            bg = Image.new("RGB", output_img.size, (255, 255, 255))
            bg.paste(output_img, mask=output_img.split()[3])  # Use alpha channel as mask
            bg.save(output_path)
        else:
            output_img.save(output_path)
        
        print(f"[DEBUG] Background removed successfully, saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"[ERROR] Background removal failed: {str(e)}")
        print(f"[DEBUG] Copying original image as fallback")
        try:
            # If background removal fails, make a copy of the original image
            input_img = Image.open(input_path)
            if input_img.mode == 'RGBA':
                input_img = input_img.convert('RGB')
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as copy
            input_img.save(output_path)
            print(f"[DEBUG] Original image copied to {output_path}")
            return output_path
        except Exception as copy_error:
            print(f"[ERROR] Failed to copy original image: {str(copy_error)}")
            raise Exception(f"Background removal failed and could not copy original: {str(e)}")

# Function to calculate Euclidean distance between two points
def euclidean_distance(pt1, pt2):
    """
    Calculate the Euclidean distance between two points.
    
    Parameters:
    pt1 (tuple): First point (x1, y1)
    pt2 (tuple): Second point (x2, y2)
    
    Returns:
    float: Euclidean distance
    """
    return math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

# Enhanced function to calculate real-world size
def calculate_real_size(pixel_size, image_width_px, camera_distance=None):
    """
    Calculate the real-world size of an object based on camera parameters
    
    Parameters:
    pixel_size (float): Size in pixels
    image_width_px (int): Image width in pixels
    camera_distance (float, optional): Override distance from camera to object
    
    Returns:
    float: Size in millimeters
    """
    # Use provided camera distance or default from app settings
    distance = camera_distance if camera_distance is not None else app_settings.get('object_distance', CAMERA_DISTANCE)
    
    # Calculate using pinhole camera model
    sensor_size = (pixel_size * SENSOR_WIDTH_MM) / image_width_px
    return (distance * sensor_size) / FOCAL_LENGTH_MM * CALIBRATION_FACTOR

# Function to calculate accurate sizes for different shapes
def calculate_shape_size(shape, contour, image_width_px):
    """
    Calculate real-world size of a shape from its contour, 
    using the simplified approach from apibg.py.
    
    Parameters:
    shape (str): The detected shape name
    contour (numpy.ndarray): Contour points
    image_width_px (int): Width of the image in pixels
    
    Returns:
    dict: Dictionary of measurements
    """
    # Calculate area in pixels
    pixel_area = cv2.contourArea(contour)
    
    # Calculate the perimeter of the contour
    perimeter_px = cv2.arcLength(contour, True)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calculate pixels per cm using the camera parameters
    # Calibrate based on the object distance from app settings
    object_distance_mm = app_settings.get('object_distance', CAMERA_DISTANCE)
    
    # Using the pinhole camera model for calculating real-world sizes
    # Calculate field of view width in mm at the object's distance
    field_of_view_width_mm = (object_distance_mm * SENSOR_WIDTH_MM) / FOCAL_LENGTH_MM
    
    # Calculate pixels per mm
    pixels_per_mm = image_width_px / field_of_view_width_mm
    
    # Convert to pixels per cm for easier understanding
    pixels_per_cm = pixels_per_mm * 10  # 10mm = 1cm
    
    # Initialize result dictionary
    measurements = {}
    
    # Calculate shape-specific measurements
    if shape == "circle":
        # For circles, use the formula based on area for more accuracy
        radius_px = math.sqrt(pixel_area / math.pi)
        
        # Alternative using bounding box (average of width and height)
        diameter_bbox_px = (w + h) / 2
        radius_bbox_px = diameter_bbox_px / 2
        
        # Use the average of the two methods for better accuracy
        radius_px = (radius_px + radius_bbox_px) / 2
        
        # Convert to real-world measurements
        radius_cm = radius_px / pixels_per_cm
        diameter_cm = radius_cm * 2
        area_cm2 = math.pi * (radius_cm ** 2)
        
        # Add circle-specific measurements
        measurements['radius_mm'] = round(radius_cm * 10, 1)  # Convert to mm
        measurements['diameter_mm'] = round(diameter_cm * 10, 1)  # Convert to mm
        measurements['dimension_text'] = f"Diameter: {diameter_cm:.1f} cm"
        
    elif shape in ["rectangle", "square"]:
        # Convert to real-world dimensions
        width_cm = w / pixels_per_cm
        height_cm = h / pixels_per_cm
            
        # Add rectangle-specific measurements
        measurements['width_mm'] = round(width_cm * 10, 1)  # Convert to mm
        measurements['height_mm'] = round(height_cm * 10, 1)  # Convert to mm
        
        if shape == "square":
            # For squares, average the width and height
            side_cm = (width_cm + height_cm) / 2
            measurements['side_mm'] = round(side_cm * 10, 1)  # Convert to mm
            measurements['dimension_text'] = f"Side: {side_cm:.1f} cm"
            area_cm2 = side_cm ** 2
        else:
            measurements['dimension_text'] = f"{width_cm:.1f} x {height_cm:.1f} cm"
            area_cm2 = width_cm * height_cm
            
    elif shape == "triangle":
        # For triangles, calculate the sides
        # Get actual perimeter in cm
        perimeter_cm = perimeter_px / pixels_per_cm
        
        # Convert pixel area to cm²
        area_cm2 = pixel_area / (pixels_per_cm ** 2)
        
        # Get width and height from bounding box
        width_cm = w / pixels_per_cm
        height_cm = h / pixels_per_cm
        
        measurements['dimension_text'] = f"{width_cm:.1f} x {height_cm:.1f} cm"
    else:
        # For other shapes, use width and height of bounding box
        width_cm = w / pixels_per_cm
        height_cm = h / pixels_per_cm
        
        # Convert pixel area to cm²
        area_cm2 = pixel_area / (pixels_per_cm ** 2)
        
        measurements['dimension_text'] = f"{width_cm:.1f} x {height_cm:.1f} cm"
    
    # Add common measurements to all shapes
    measurements['area_pixels'] = round(pixel_area, 1)
    measurements['area_mm2'] = round(area_cm2 * 100, 1)  # Convert cm² to mm²
    measurements['perimeter_px'] = round(perimeter_px, 1)
    measurements['perimeter_mm'] = round((perimeter_px / pixels_per_cm) * 10, 1)  # Convert to mm
    
    # Apply calibration factor to all measurements
    for key in measurements:
        if key not in ['dimension_text'] and isinstance(measurements[key], (int, float)):
            measurements[key] = round(measurements[key] * CALIBRATION_FACTOR, 1)
    
    # Update dimension text with calibrated values
    if shape == "circle":
        measurements['dimension_text'] = f"Diameter: {measurements['diameter_mm']/10:.1f} cm"
    elif shape == "square":
        measurements['dimension_text'] = f"Side: {measurements['side_mm']/10:.1f} cm"
    elif shape == "rectangle":
        measurements['dimension_text'] = f"{measurements['width_mm']/10:.1f} x {measurements['height_mm']/10:.1f} cm"
    elif shape == "triangle" and 'sides_mm' in measurements:
        sides = measurements['sides_mm']
        measurements['dimension_text'] = f"Sides: {sides[0]/10:.1f}, {sides[1]/10:.1f}, {sides[2]/10:.1f} cm"
    else:
        measurements['dimension_text'] = f"Area: {measurements['area_mm2']/100:.1f} cm²"
    
    return measurements

# Improved preprocessing function for more accurate contour detection
def preprocess_image_for_contours(image):
    """
    Preprocess an image for contour detection.
    Apply multiple preprocessing techniques for better contour quality.
    
    Parameters:
    image (numpy.ndarray): Input image
    
    Returns:
    numpy.ndarray: Binary image ready for contour detection
    """
    # Check if image has alpha channel (RGBA)
    if len(image.shape) > 2 and image.shape[2] == 4:
        # Extract alpha channel
        alpha = image[:, :, 3]
        # Threshold the alpha channel to get a binary mask
        _, binary = cv2.threshold(alpha, 128, 255, cv2.THRESH_BINARY)
    else:
        # Convert to grayscale if not already
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
        
        # Use Otsu's thresholding for better results
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations for cleaner contours
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Ensure binary is uint8 type
    binary = binary.astype(np.uint8)
    
    return binary


def find_main_shape(contours):
    if not contours:
        return None
    min_area = 50
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid_contours:
        return None
    return max(valid_contours, key=cv2.contourArea)


def optimize_circular_contour(contour):
    try:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return contour, False

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > 0.75:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                axes = ellipse[1]
                axes_ratio = min(axes) / max(axes) if max(axes) > 0 else 0
                if axes_ratio > 0.85:
                    perfect_circle = [
                        [[int(center[0] + radius * np.cos(i * 2 * np.pi / 180)),
                          int(center[1] + radius * np.sin(i * 2 * np.pi / 180))]]
                        for i in range(180)
                    ]
                    return np.array(perfect_circle, dtype=np.int32), True

            if circularity > 0.85:
                perfect_circle = [
                    [[int(center[0] + radius * np.cos(i * 2 * np.pi / 180)),
                      int(center[1] + radius * np.sin(i * 2 * np.pi / 180))]]
                    for i in range(180)
                ]
                return np.array(perfect_circle, dtype=np.int32), True
    except:
        pass
    return contour, False


def preprocess_for_circle_detection(image):
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return binary


def smooth_contour(contour, factor=0.005):
    epsilon = factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        shape = "unknown"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            shape = "square" if 0.95 <= aspect_ratio <= 1.05 else "rectangle"
        elif len(approx) > 4:
            area = cv2.contourArea(c)
            circularity = (4 * np.pi * area) / (peri * peri)
            shape = "circle" if circularity > 0.8 else "ellipse/rounded"
        return shape


# Standalone function for detecting shape from contour
def detect_shape_from_contour(contour):
    """
    Detect the shape of a contour.
    
    Parameters:
    contour (numpy.ndarray): The contour to detect
    
    Returns:
    str: The detected shape name
    """
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
        shape = "circle" if circularity > 0.8 else "ellipse/rounded"
    return shape

    
    try:
        # Generate output paths
        bg_removed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"nobg_{filename}")
        if not bg_removed_path.lower().endswith('.png'):
            bg_removed_path = bg_removed_path.rsplit('.', 1)[0] + '.png'
        
        # Step 1: Remove background with priority on API if key is available
        api_key = app_settings.get('api_key')
        print(f"[DEBUG] Using API key for background removal: {'Yes' if api_key else 'No'}")
        remove_background(filepath, bg_removed_path, api_key=api_key)
        
        # Step 2: Load image with alpha channel preserved
        img = cv2.imread(bg_removed_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise Exception(f"Failed to load image after background removal: {bg_removed_path}")
        
        # Get original dimensions
        img_height, img_width = img.shape[:2]
        print(f"[DEBUG] Original image dimensions: {img_width}x{img_height}")
        
        # Step 3: Calibration approach from apibg.py
        field_height_cm = app_settings.get('field_height_cm', 30.0)
        pixels_per_cm = img_height / field_height_cm
        print(f"[DEBUG] Calibration: {pixels_per_cm} pixels per cm")
        
        # Step 4: Create a clean binary mask for contour detection
        if len(img.shape) > 2 and img.shape[2] == 4:
            # Use alpha channel for perfect mask
            alpha = img[:, :, 3]
            _, mask = cv2.threshold(alpha, 128, 255, cv2.THRESH_BINARY)
        else:
            # No alpha channel, create mask through color thresholding
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Ensure mask is 8-bit for contour detection
        mask = mask.astype(np.uint8)
        
        # Step 5: Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"[DEBUG] Found {len(contours)} contours")
        
        # Filter small contours (noise)
        min_contour_area = 500  # Minimum area in pixels
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
        print(f"[DEBUG] After filtering: {len(valid_contours)} valid contours")
        
        # Create output image
        if len(img.shape) > 2 and img.shape[2] == 4:
            output_img = img[:, :, :3].copy()  # Remove alpha for output
        else:
            output_img = img.copy()
        
        # Process contours and calculate sizes
        results = []
        
        for i, contour in enumerate(valid_contours):
            # Get shape
            shape = detect_shape_from_contour(contour)
            
            # Calculate measurements
            area_pixels = cv2.contourArea(contour)
            perimeter_pixels = cv2.arcLength(contour, True)
            
            # Convert to real-world measurements using pixels_per_cm
            area_cm2 = area_pixels / (pixels_per_cm ** 2)
            perimeter_cm = perimeter_pixels / pixels_per_cm
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            width_cm = w / pixels_per_cm
            height_cm = h / pixels_per_cm
            
            # Get center of contour
            M = cv2.moments(contour)
            if M["m00"] != 0:  # Avoid division by zero
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX = x + w // 2
                cY = y + h // 2
            
            # Draw contour with unique color for each shape
            color = (0, 255, 0)  # Default green
            if shape == "circle":
                color = (0, 255, 0)  # Green for circles
            elif shape == "square":
                color = (0, 255, 255)  # Yellow for squares
            elif shape == "rectangle":
                color = (0, 165, 255)  # Orange for rectangles
            elif shape == "triangle":
                color = (255, 0, 0)  # Blue for triangles
                
            cv2.drawContours(output_img, [contour], -1, color, 2)
            
            # Create measurement result object
            result = {
                'shape': shape,
                'area_cm2': round(area_cm2, 2),
                'perimeter_cm': round(perimeter_cm, 2),
                'width_cm': round(width_cm, 2),
                'height_cm': round(height_cm, 2)
            }
            
            # Add shape-specific measurements
            if shape == "circle":
                # For circles, calculate diameter more accurately using area
                area = cv2.contourArea(contour)
                radius_pixels = np.sqrt(area / np.pi)
                radius_cm = radius_pixels / pixels_per_cm
                diameter_cm = 2 * radius_cm
                
                result['diameter_cm'] = round(diameter_cm, 2)
                result['radius_cm'] = round(radius_cm, 2)
                
                # Add diameter measurement to image
                cv2.putText(output_img, f"Diameter: {diameter_cm:.2f} cm", 
                           (cX - 70, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Draw circle center and radius
                cv2.circle(output_img, (cX, cY), 5, (0, 0, 255), -1)
                cv2.circle(output_img, (cX, cY), int(radius_pixels), (255, 0, 0), 2)
            else:
                # For other shapes, add width/height to image
                cv2.putText(output_img, f"{width_cm:.2f} x {height_cm:.2f} cm", 
                           (cX - 70, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Draw bounding box
                cv2.rectangle(output_img, (x, y), (x+w, y+h), (255, 0, 0), 1)
            
            # Add shape label
            cv2.putText(output_img, shape, (cX - 30, cY - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Add area measurement
            cv2.putText(output_img, f"Area: {area_cm2:.2f} cm²", 
                       (cX - 70, cY + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Add the result to our list
            results.append(result)
        
        # If we found multiple contours, sort by area (largest first)
        if len(results) > 1:
            results.sort(key=lambda x: x['area_cm2'], reverse=True)
        
        # Save the processed image
        output_filename = f"size_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, output_img)
        
        print(f"[DEBUG] Saved output image to {output_path}")
        print(f"[DEBUG] Detected {len(results)} valid shapes")
        
        return jsonify({
            'success': True,
            'measurements': results,
            'processedImage': '/static/uploads/' + output_filename
        })
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[ERROR] Error in size detection: {str(e)}\n{error_trace}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect-color', methods=['POST'])
def detect_color():
    data = request.json
    
    if 'filename' not in data:
        return jsonify({'error': 'Filename not provided'}), 400
    
    filename = data['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    # Process image with enhanced color detection
    try:
        # Use our enhanced color detector
        output_image, color_results = enhanced_color_detector.detect_colors(filepath)
        
        # Save the processed image
        output_filename = f"color_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, output_image)
        
        # Add shape info to each color (using the existing shape detector)
        sd = ShapeDetector()
        
        # Get binary image for contour detection
        binary = preprocess_image_for_contours(cv2.imread(filepath))
        # Find contours
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Match shapes to colors
        for i, contour in enumerate(cnts):
            if cv2.contourArea(contour) < 500:
                continue
                
            # Detect shape
            shape = sd.detect(contour)
            
            # If we have a corresponding color result, add the shape
            if i < len(color_results):
                color_results[i]['shape'] = shape
        
        return jsonify({
            'success': True,
            'colors': color_results,
            'processedImage': '/static/uploads/' + output_filename
        })
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in enhanced color detection: {str(e)}\n{error_trace}")
        
        # Fall back to the original color detection if enhanced detection fails
        try:
            # Original implementation starts here
            # Generate output paths
            bg_removed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"nobg_{filename}")
            if not bg_removed_path.lower().endswith('.png'):
                bg_removed_path = bg_removed_path.rsplit('.', 1)[0] + '.png'
            
            # Remove background and save as PNG
            remove_background(filepath, bg_removed_path)
            
            # Load image
            image = cv2.imread(bg_removed_path)
            if image is None:
                raise Exception("Failed to load image after background removal")
            
            # Get image dimensions for size calculations
            img_height, img_width = image.shape[:2]
            
            # Resize for processing while maintaining aspect ratio
            max_dim = 1000
            scale_factor = min(max_dim / img_width, max_dim / img_height)
            if scale_factor < 1:
                resized_width = int(img_width * scale_factor)
                resized_height = int(img_height * scale_factor)
                resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
                ratio = img_height / float(resized_height)
            else:
                resized = image.copy()
                ratio = 1.0
            
            # Apply improved edge detection and preprocessing
            binary = preprocess_image_for_contours(resized)
            
            # Find contours
            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process results
            results = []
            output_image = image.copy()
            
            for c in cnts:
                # Filter small contours
                if cv2.contourArea(c) < 50:
                    continue
                    
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                
                # Scale contour back to original image size    
                c_orig = c.astype("float") * ratio
                c_orig = c_orig.astype("int")
                
                # Get center coordinates
                cX = int((M["m10"] / M["m00"]) * ratio)
                cY = int((M["m01"] / M["m00"]) * ratio)
                
                # Detect shape
                shape = sd.detect(c)
                
                # Get dominant color of the shape
                avg_color = get_avg_color(c_orig, image)
                color_name = closest_color(avg_color)
                
                # Convert RGB to hex for frontend display
                hex_color = '#{:02x}{:02x}{:02x}'.format(avg_color[0], avg_color[1], avg_color[2])
                
                # Save contour details with shape and color information
                results.append({
                    'shape': shape,
                    'color': color_name,
                    'rgb': avg_color,
                    'hex': hex_color
                })
                
                # Draw contours and labels on the output image
                cv2.drawContours(output_image, [c_orig], -1, (0, 255, 0), 2)
                
                # Draw shape name and color name
                cv2.putText(output_image, f"{shape}", (cX, cY - 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(output_image, f"{color_name}", (cX, cY + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add a small rectangle with the detected color
                color_block_size = 30
                color_bg = np.zeros((color_block_size, color_block_size, 3), dtype=np.uint8)
                color_bg[:, :] = (avg_color[2], avg_color[1], avg_color[0])  # BGR for OpenCV
                
                # Overlay the color rectangle
                x_pos = cX + 60
                y_pos = cY - 15
                output_image[y_pos:y_pos+color_block_size, x_pos:x_pos+color_block_size] = color_bg
                
                # Add a black border around the color square
                cv2.rectangle(output_image, 
                             (x_pos, y_pos), 
                             (x_pos + color_block_size, y_pos + color_block_size), 
                             (0, 0, 0), 1)
            
            # Save the processed image
            output_filename = f"color_{filename}"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            cv2.imwrite(output_path, output_image)
            
            return jsonify({
                'success': True,
                'colors': results,
                'processedImage': '/static/uploads/' + output_filename
            })
            
        except Exception as e2:
            error_trace2 = traceback.format_exc()
            print(f"Error in fallback color detection: {str(e2)}\n{error_trace2}")
            return jsonify({'error': f"Error detecting colors: {str(e)}, fallback failed: {str(e2)}"}), 500

@app.route('/detect-shape-color', methods=['POST'])
def detect_shape_color():
    data = request.json
    
    if 'filename' not in data:
        return jsonify({'error': 'Filename not provided'}), 400
    
    filename = data['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    # Process image for combined shape and color detection
    try:
        # First try with the enhanced color detector
        try:
            # Process the image with our enhanced detector first
            output_image, color_results = enhanced_color_detector.detect_colors(filepath)
            
            # We still need to detect shapes and get measurements
            # Generate output paths for background removal
            bg_removed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"nobg_{filename}")
            if not bg_removed_path.lower().endswith('.png'):
                bg_removed_path = bg_removed_path.rsplit('.', 1)[0] + '.png'
            
            # Remove background and save as PNG
            remove_background(filepath, bg_removed_path)
            
            # Load image
            image = cv2.imread(bg_removed_path)
            if image is None:
                raise Exception("Failed to load image after background removal")
            
            # Get image dimensions for size calculations
            img_height, img_width = image.shape[:2]
            
            # Resize for processing while maintaining aspect ratio
            max_dim = 1000
            scale_factor = min(max_dim / img_width, max_dim / img_height)
            if scale_factor < 1:
                resized_width = int(img_width * scale_factor)
                resized_height = int(img_height * scale_factor)
                resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
                ratio = img_height / float(resized_height)
            else:
                resized = image.copy()
                ratio = 1.0
            
            # Apply improved edge detection and preprocessing
            binary = preprocess_image_for_contours(resized)
            
            # Find contours
            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Initialize shape detector
            sd = ShapeDetector()
            
            # Create combined results
            combined_results = []
            
            # Process each contour
            for i, c in enumerate(cnts):
                # Filter small contours
                if cv2.contourArea(c) < 500:
                    continue
                
                # Scale contour to original size
                c_orig = c.astype("float") * ratio
                c_orig = c_orig.astype("int")
                
                # Detect shape
                shape = detect_shape_from_contour(c)
                
                # Calculate measurements
                measurements = calculate_shape_size(shape, c, img_width)
                
                # Find matching color result
                color_info = None
                if i < len(color_results):
                    color_info = color_results[i]
                else:
                    # Fallback to original method if no match
                    avg_color = get_avg_color(c_orig, image)
                    color_name = closest_color(avg_color)
                    hex_color = '#{:02x}{:02x}{:02x}'.format(avg_color[0], avg_color[1], avg_color[2])
                    color_info = {
                        'color': color_name,
                        'rgb': avg_color,
                        'hex': hex_color
                    }
                
                # Create combined result
                combined_result = {
                    'shape': shape,
                    'color': color_info['color'],
                    'hex': color_info['hex'],
                    'dimensions': measurements['dimension_text'],
                    'area_mm2': measurements['area_mm2']
                }
                
                combined_results.append(combined_result)
            
            # Save the processed image
            output_filename = f"combined_{filename}"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            cv2.imwrite(output_path, output_image)
            
            return jsonify({
                'success': True,
                'results': combined_results,
                'processedImage': '/static/uploads/' + output_filename
            })
            
        except Exception as enhanced_error:
            # Log the error but proceed with original method
            print(f"Enhanced detection failed: {str(enhanced_error)}")
            print(f"Falling back to original method...")
            
            # Original implementation starts here
            # Generate output paths
            bg_removed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"nobg_{filename}")
            if not bg_removed_path.lower().endswith('.png'):
                bg_removed_path = bg_removed_path.rsplit('.', 1)[0] + '.png'
            
            # Remove background and save as PNG
            remove_background(filepath, bg_removed_path)
            
            # Load image
            image = cv2.imread(bg_removed_path)
            if image is None:
                raise Exception("Failed to load image after background removal")
            
            # Get image dimensions for size calculations
            img_height, img_width = image.shape[:2]
            
            # Resize for processing while maintaining aspect ratio
            max_dim = 1000
            scale_factor = min(max_dim / img_width, max_dim / img_height)
            if scale_factor < 1:
                resized_width = int(img_width * scale_factor)
                resized_height = int(img_height * scale_factor)
                resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
                ratio = img_height / float(resized_height)
            else:
                resized = image.copy()
                ratio = 1.0
            
            # Apply improved edge detection and preprocessing
            binary = preprocess_image_for_contours(resized)
            
            # Find contours - using RETR_EXTERNAL to get only the outermost contours
            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the main shape (largest contour)
            main_contour = find_main_shape(cnts)
            
            if main_contour is None:
                return jsonify({'error': 'No shapes detected in the image'}), 400
                
            # Try specialized circle detection
            # First, check if it looks like a circle
            peri = cv2.arcLength(main_contour, True)
            area = cv2.contourArea(main_contour)
            circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
            
            if circularity > 0.75:  # Potentially a circle
                # Try specialized circle detection preprocessing
                circle_binary = preprocess_for_circle_detection(resized)
                circle_contours, _ = cv2.findContours(circle_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if circle_contours:
                    # Find the largest contour
                    circle_main_contour = max(circle_contours, key=cv2.contourArea)
                    if cv2.contourArea(circle_main_contour) > 100:  # Ensure it's a significant contour
                        main_contour = circle_main_contour
            
            # Process all contours
            output = image.copy()
            results = []
            sd = ShapeDetector()
            
            for c in cnts:
                # Skip small contours
                if cv2.contourArea(c) < 500:  # Min area threshold
                    continue
                
                # Get moments and compute centroid
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                
                # Scale back to original size
                c_orig = c.astype("float") * ratio
                c_orig = c_orig.astype("int")
                
                cX = int((M["m10"] / M["m00"]) * ratio)
                cY = int((M["m01"] / M["m00"]) * ratio)
                
                # Detect shape
                shape = detect_shape_from_contour(c)
                
                # Get color
                avg_color = get_avg_color(c_orig, image)
                color_name = closest_color(avg_color)
                hex_color = '#{:02x}{:02x}{:02x}'.format(avg_color[0], avg_color[1], avg_color[2])
                
                # Calculate size
                measurements = calculate_shape_size(shape, c, img_width)
                
                # Create result object
                result = {
                    'shape': shape,
                    'color': color_name,
                    'hex': hex_color,
                    'dimensions': measurements['dimension_text'],
                    'area_mm2': measurements['area_mm2']
                }
                
                results.append(result)
                
                # Draw on output image
                cv2.drawContours(output, [c_orig], -1, (0, 255, 0), 2)
                cv2.putText(output, shape, (cX, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(output, color_name, (cX, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Add dimension text
                cv2.putText(output, measurements['dimension_text'], (cX, cY + 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw a small color swatch
                color_swatch_size = 20
                color_swatch = np.zeros((color_swatch_size, color_swatch_size, 3), dtype=np.uint8)
                color_swatch[:, :] = (avg_color[2], avg_color[1], avg_color[0])  # BGR format
                
                # Place color swatch near the shape label
                swatch_x = cX + 60
                swatch_y = cY - 10
                
                # Make sure the swatch fits within the image
                if (swatch_y >= 0 and swatch_y + color_swatch_size < output.shape[0] and 
                    swatch_x >= 0 and swatch_x + color_swatch_size < output.shape[1]):
                    output[swatch_y:swatch_y+color_swatch_size, 
                           swatch_x:swatch_x+color_swatch_size] = color_swatch
                    
                    # Add border
                    cv2.rectangle(output, 
                                (swatch_x, swatch_y), 
                                (swatch_x + color_swatch_size, swatch_y + color_swatch_size), 
                                (0, 0, 0), 1)
            
            # Save the result
            output_filename = f"combined_{filename}"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            cv2.imwrite(output_path, output)
            
            return jsonify({
                'success': True,
                'results': results,
                'processedImage': '/static/uploads/' + output_filename
            })
            
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in combined detection: {str(e)}\n{error_trace}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/uploads/<filename>')
def serve_file(filename):
    """Serve files from the upload directory."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<filename>')
def download_file(filename):
    """Download a file with the correct headers for browser download."""
    return send_from_directory(
        app.config['UPLOAD_FOLDER'], 
        filename, 
        as_attachment=True,
        mimetype='image/png'
    )

@app.route('/settings', methods=['GET'])
def get_settings():
    """Get current application settings"""
    return jsonify(app_settings)

@app.route('/settings', methods=['POST'])
def update_settings():
    """Update application settings"""
    data = request.get_json()
    
    if data and isinstance(data, dict):
        # Update only provided settings
        for key in data:
            if key in app_settings:
                # For numeric settings, ensure we have proper values
                if key == 'object_distance' or key == 'field_height_cm':
                    try:
                        value = float(data[key])
                        if value <= 0:
                            return jsonify({'error': f'Invalid value for {key}: must be positive'}), 400
                        app_settings[key] = value
                    except ValueError:
                        return jsonify({'error': f'Invalid value for {key}: must be a number'}), 400
                else:
                    app_settings[key] = data[key]
    
    return jsonify({'success': True, 'settings': app_settings})

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload from the frontend."""
    if 'image' not in request.files:
        print("[ERROR] No image part in the request")
        return jsonify({'error': 'No image part in the request'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        print("[ERROR] No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Generate a unique filename
            filename = secure_filename(str(uuid.uuid4()) + '_' + file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Ensure upload directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save the file
            file.save(filepath)
            print(f"[DEBUG] File saved successfully to {filepath}")
            
            return jsonify({
                'success': True,
                'filename': filename,
                'filepath': '/static/uploads/' + filename
            })
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"[ERROR] Error saving uploaded file: {str(e)}\n{error_trace}")
            return jsonify({'error': f'Error saving file: {str(e)}'}), 500
    
    print("[ERROR] File type not allowed")
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/detect-shape', methods=['POST'])
def detect_shape():
    """Detect shapes in the uploaded image."""
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({'error': 'No filename provided'}), 400
    
    filename = data['filename']
    print(f"[DEBUG] Shape detection requested for file: {filename}")
    
    try:
        # Original image path
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"[DEBUG] Original path: {original_path}")
        
        # Generate output paths for background-removed image
        bg_removed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"nobg_{filename}")
        if not bg_removed_path.lower().endswith('.png'):
            bg_removed_path = bg_removed_path.rsplit('.', 1)[0] + '.png'
        print(f"[DEBUG] Background removed path: {bg_removed_path}")
        
        # Check if original file exists
        if not os.path.exists(original_path):
            print(f"[ERROR] Original file not found: {original_path}")
            return jsonify({'error': f'File not found: {original_path}'}), 404
        
        # Remove background and save as PNG using API key from settings if available
        print(f"[DEBUG] Removing background from {original_path}")
        try:
            remove_background(original_path, bg_removed_path)
        except Exception as bg_error:
            print(f"[ERROR] Background removal failed: {str(bg_error)}")
            return jsonify({'error': f'Background removal failed: {str(bg_error)}'}), 500
        
        # Check if background removal was successful
        if not os.path.exists(bg_removed_path):
            print(f"[ERROR] Failed to remove background: {bg_removed_path}")
            return jsonify({'error': 'Failed to process image'}), 500
            
        # Read the background-removed image
        try:
            img = cv2.imread(bg_removed_path, cv2.IMREAD_UNCHANGED)
        except Exception as img_error:
            print(f"[ERROR] Failed to read image: {str(img_error)}")
            return jsonify({'error': f'Failed to read image: {str(img_error)}'}), 500
        
        if img is None:
            print(f"[ERROR] Failed to read image after background removal: {bg_removed_path}")
            return jsonify({'error': 'Failed to read processed image'}), 500
        
        print(f"[DEBUG] Successfully loaded image with shape: {img.shape}")
        
        # Resize image while maintaining aspect ratio
        max_dimension = 1000
        h, w = img.shape[:2]
        
        # Calculate new dimensions
        if h > w:
            new_h = max_dimension
            new_w = int(w * (max_dimension / h))
        else:
            new_w = max_dimension
            new_h = int(h * (max_dimension / w))
            
        # Resize image
        print(f"[DEBUG] Resizing image from {w}x{h} to {new_w}x{new_h}")
        try:
            img_resized = cv2.resize(img, (new_w, new_h))
        except Exception as resize_error:
            print(f"[ERROR] Failed to resize image: {str(resize_error)}")
            return jsonify({'error': f'Failed to resize image: {str(resize_error)}'}), 500
        
        # Create a preprocessed image for contour detection
        print("[DEBUG] Preprocessing image for contour detection")
        try:
            binary = preprocess_image_for_contours(img_resized)
        except Exception as preprocess_error:
            print(f"[ERROR] Failed to preprocess image: {str(preprocess_error)}")
            return jsonify({'error': f'Failed to preprocess image: {str(preprocess_error)}'}), 500
        
        # Find contours in the processed image
        print("[DEBUG] Finding contours")
        try:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"[DEBUG] Found {len(contours)} contours")
        except Exception as contour_error:
            print(f"[ERROR] Failed to find contours: {str(contour_error)}")
            return jsonify({'error': f'Failed to find contours: {str(contour_error)}'}), 500
        
        # Find the main shape (largest contour)
        try:
            main_contour = find_main_shape(contours)
        except Exception as main_shape_error:
            print(f"[ERROR] Failed to find main shape: {str(main_shape_error)}")
            return jsonify({'error': f'Failed to find main shape: {str(main_shape_error)}'}), 500
        
        if main_contour is None:
            print("[ERROR] No main shape detected")
            return jsonify({'error': 'No shape detected'}), 400
        
        # Apply contour smoothing
        try:
            smoothed_contour = smooth_contour(main_contour)
        except Exception as smooth_error:
            print(f"[ERROR] Failed to smooth contour: {str(smooth_error)}")
            return jsonify({'error': f'Failed to smooth contour: {str(smooth_error)}'}), 500
        
        # Check if it's a circle for special processing
        try:
            circular_contour, is_circle = optimize_circular_contour(smoothed_contour)
        except Exception as circle_error:
            print(f"[ERROR] Failed to check if circle: {str(circle_error)}")
            return jsonify({'error': f'Failed to check if circle: {str(circle_error)}'}), 500
        
        if is_circle:
            main_contour = circular_contour
            print("[DEBUG] Detected a circular shape, optimized the contour")
        else:
            main_contour = smoothed_contour
        
        # Create a visualization of the detected shape
        try:
            viz_img = img_resized.copy()
            if len(viz_img.shape) == 2:  # If grayscale, convert to color
                viz_img = cv2.cvtColor(viz_img, cv2.COLOR_GRAY2BGR)
            elif viz_img.shape[2] == 4:  # If RGBA, convert to BGR
                viz_img = cv2.cvtColor(viz_img[:, :, :3], cv2.COLOR_BGRA2BGR)
        except Exception as viz_error:
            print(f"[ERROR] Failed to prepare visualization image: {str(viz_error)}")
            return jsonify({'error': f'Failed to prepare visualization image: {str(viz_error)}'}), 500
        
        # Draw the contour on the image
        try:
            cv2.drawContours(viz_img, [main_contour], -1, (0, 255, 0), 2)
        except Exception as draw_error:
            print(f"[ERROR] Failed to draw contours: {str(draw_error)}")
            return jsonify({'error': f'Failed to draw contours: {str(draw_error)}'}), 500
        
        # Detect shape
        try:
            shape_name = detect_shape_from_contour(main_contour)
            print(f"[DEBUG] Detected shape: {shape_name}")
        except Exception as shape_error:
            print(f"[ERROR] Failed to detect shape: {str(shape_error)}")
            return jsonify({'error': f'Failed to detect shape: {str(shape_error)}'}), 500
        
        # Calculate measurements for the shape
        try:
            measurements = calculate_shape_size(shape_name, main_contour, new_w)
        except Exception as measure_error:
            print(f"[ERROR] Failed to calculate measurements: {str(measure_error)}")
            return jsonify({'error': f'Failed to calculate measurements: {str(measure_error)}'}), 500
        
        # Calculate bounding rectangle for the contour
        try:
            x, y, w, h = cv2.boundingRect(main_contour)
        except Exception as bound_error:
            print(f"[ERROR] Failed to calculate bounding rectangle: {str(bound_error)}")
            return jsonify({'error': f'Failed to calculate bounding rectangle: {str(bound_error)}'}), 500
        
        # Calculate center of the shape
        try:
            M = cv2.moments(main_contour)
            if M["m00"] != 0:  # Avoid division by zero
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX = x + w // 2
                cY = y + h // 2
        except Exception as center_error:
            print(f"[ERROR] Failed to calculate center: {str(center_error)}")
            return jsonify({'error': f'Failed to calculate center: {str(center_error)}'}), 500
        
        # Add shape label to the visualization
        try:
            cv2.putText(viz_img, shape_name, (cX - 20, cY), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        except Exception as text_error:
            print(f"[ERROR] Failed to add text to image: {str(text_error)}")
            return jsonify({'error': f'Failed to add text to image: {str(text_error)}'}), 500
        
        # Save the visualization image
        try:
            viz_path = os.path.join(app.config['UPLOAD_FOLDER'], f"viz_{filename}")
            if not viz_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                viz_path = viz_path.rsplit('.', 1)[0] + '.png'
            
            cv2.imwrite(viz_path, viz_img)
            print(f"[DEBUG] Visualization saved to: {viz_path}")
        except Exception as save_error:
            print(f"[ERROR] Failed to save visualization: {str(save_error)}")
            return jsonify({'error': f'Failed to save visualization: {str(save_error)}'}), 500
        
        # Prepare the result in a format the frontend is expecting
        try:
            result = {
                'shape': shape_name,
                'area_mm2': measurements['area_mm2'],
                'dimensions': measurements['dimension_text'],
                'contour': main_contour.tolist(),  # Convert to list for JSON serialization
                'boundingBox': {'x': x, 'y': y, 'width': w, 'height': h},
                'center': {'x': cX, 'y': cY}
            }
            
            print(f"[DEBUG] Returning success response with shape: {shape_name}")
            return jsonify({
                'success': True,
                'shapes': [result],  # Return in shapes array for frontend compatibility
                'processedImage': '/static/uploads/' + os.path.basename(viz_path)
            })
        except Exception as result_error:
            print(f"[ERROR] Failed to prepare result: {str(result_error)}")
            return jsonify({'error': f'Failed to prepare result: {str(result_error)}'}), 500
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[ERROR] Error in shape detection: {str(e)}\n{error_trace}")
        return jsonify({'error': str(e)}), 500

# Initialize application resources
def initialize_app():
    # Ensure all required directories exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Check if color data file exists and is readable
    try:
        color_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'color_data.xlsx')
        if os.path.exists(color_data_path):
            print(f"Color data file found at: {color_data_path}")
        else:
            print(f"Warning: Color data file not found at: {color_data_path}")
            print("Using fallback colors for detection")
    except Exception as e:
        print(f"Error checking color data: {str(e)}")
    
    print(f"Shape detection server initialized. Upload directory: {UPLOAD_FOLDER}")
    print("Available endpoints:")
    print("  - /upload: POST endpoint for uploading images")
    print("  - /detect-shape: POST endpoint for shape and color detection")
    print("  - /static/uploads/<filename>: GET endpoint for accessing processed images")

@app.route('/detect-size', methods=['POST'])
def detect_size():
    """Detect the size of objects in the uploaded image using the apibg.py approach."""
    data = request.json
    
    if 'filename' not in data:
        return jsonify({'error': 'Filename not provided'}), 400
    
    filename = data['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': f'File not found: {filepath}'}), 404
    
    try:
        print(f"[DEBUG] Size detection for file: {filename}")
        
        # Step 1: Generate paths for processing
        bg_removed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"nobg_{filename}")
        if not bg_removed_path.lower().endswith('.png'):
            bg_removed_path = bg_removed_path.rsplit('.', 1)[0] + '.png'
        print(f"[DEBUG] Background removal path: {bg_removed_path}")
        
        # Step 2: Remove background with rembg library
        print(f"[DEBUG] Using rembg for background removal")
        remove_background(filepath, bg_removed_path)
        
        # Step 3: Load the background-removed image
        try:
            image = cv2.imread(bg_removed_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"[ERROR] Failed to load image: {bg_removed_path}")
                # Try loading with regular imread as fallback
                image = cv2.imread(bg_removed_path)
                if image is None:
                    return jsonify({'error': f'Failed to load processed image: {bg_removed_path}'}), 500
                print(f"[DEBUG] Fallback image loading succeeded without alpha channel")
            print(f"[DEBUG] Image loaded successfully: shape={image.shape}")
        except Exception as load_error:
            print(f"[ERROR] Image loading error: {str(load_error)}")
            return jsonify({'error': f'Failed to load image: {str(load_error)}'}), 500
        
        # Step 4: Get image dimensions
        img_height, img_width = image.shape[:2]
        print(f"[DEBUG] Image dimensions: {img_width}x{img_height}")
        
        # Step 5: Setup calibration from apibg.py approach
        field_height_cm = app_settings.get('field_height_cm', 30.0)
        pixels_per_cm = img_height / field_height_cm
        print(f"[DEBUG] Calibration: {pixels_per_cm:.2f} pixels per cm (assuming {field_height_cm} cm field height)")
        
        # Prepare the output image
        output_image = None
        if len(image.shape) > 2 and image.shape[2] == 4:
            # Has alpha channel, extract RGB
            output_image = image[:, :, :3].copy()
        else:
            # Already RGB
            output_image = image.copy()
        
        # Step 6: Create mask for contour detection
        try:
            # Check if we have an alpha channel
            if len(image.shape) > 2 and image.shape[2] == 4:
                # Use alpha channel for mask
                alpha_channel = image[:, :, 3]
                _, mask = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)
                print(f"[DEBUG] Created mask from alpha channel")
            else:
                # Convert to grayscale for mask
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                print(f"[DEBUG] Created mask from grayscale conversion")
            
            # Clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Ensure mask is 8-bit for contour detection
            mask = mask.astype(np.uint8)
            
            # Save mask for debugging
            debug_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], f"mask_{filename}")
            cv2.imwrite(debug_mask_path, mask)
            print(f"[DEBUG] Saved mask image to {debug_mask_path}")
        except Exception as mask_error:
            print(f"[ERROR] Mask creation failed: {str(mask_error)}")
            return jsonify({'error': f'Error creating mask: {str(mask_error)}'}), 500
        
        # Step 7: Find contours in the mask
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"[DEBUG] Found {len(contours)} contours")
            
            if len(contours) == 0:
                # Fallback: Try inverting the mask
                print(f"[DEBUG] No contours found, trying inverted mask")
                inverted_mask = cv2.bitwise_not(mask)
                contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print(f"[DEBUG] Found {len(contours)} contours with inverted mask")
            
            # Filter small contours (noise)
            min_area = 100  # Reduced from 500 to be more lenient
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            print(f"[DEBUG] After filtering: {len(valid_contours)} valid contours")
            
            if len(valid_contours) == 0:
                print(f"[ERROR] No valid contours found after filtering")
                # Try again with even smaller minimum area
                valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
                print(f"[DEBUG] After reducing minimum area: {len(valid_contours)} valid contours")
                
                if len(valid_contours) == 0:
                    return jsonify({'error': 'No shapes detected in the image'}), 400
        except Exception as contour_error:
            print(f"[ERROR] Contour detection failed: {str(contour_error)}")
            return jsonify({'error': f'Error detecting contours: {str(contour_error)}'}), 500
        
        # Step 8: Process contours and calculate sizes
        results = []
        for i, contour in enumerate(valid_contours):
            try:
                # Calculate basic measurements
                area_pixels = cv2.contourArea(contour)
                perimeter_pixels = cv2.arcLength(contour, True)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Get shape
                shape = detect_shape_from_contour(contour)
                print(f"[DEBUG] Contour {i+1}: Shape={shape}, Area={area_pixels} px")
                
                # Calculate real-world measurements using the calibration
                area_cm2 = area_pixels / (pixels_per_cm ** 2)
                perimeter_cm = perimeter_pixels / pixels_per_cm
                width_cm = w / pixels_per_cm
                height_cm = h / pixels_per_cm
                
                # Get center of contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX = x + w // 2
                    cY = y + h // 2
                
                # Draw contour and info on the output image
                cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)
                
                # Add result
                result = {
                    'shape': shape,
                    'area_cm2': round(area_cm2, 2),
                    'perimeter_cm': round(perimeter_cm, 2),
                    'width_cm': round(width_cm, 2),
                    'height_cm': round(height_cm, 2)
                }
                
                # Add shape-specific measurements
                if shape == "circle":
                    # Calculate diameter using area formula
                    radius_pixels = np.sqrt(area_pixels / np.pi)
                    radius_cm = radius_pixels / pixels_per_cm
                    diameter_cm = 2 * radius_cm
                    
                    result['diameter_cm'] = round(diameter_cm, 2)
                    result['radius_cm'] = round(radius_cm, 2)
                    
                    # Draw circle outline
                    cv2.circle(output_image, (cX, cY), int(radius_pixels), (255, 0, 0), 2)
                    
                    # Add text labels
                    cv2.putText(output_image, f"circle", (cX - 20, cY - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(output_image, f"d: {diameter_cm:.2f} cm", (cX - 50, cY), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # Draw bounding rectangle
                    cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # Add text labels
                    cv2.putText(output_image, f"{shape}", (cX - 30, cY - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(output_image, f"{width_cm:.2f} x {height_cm:.2f} cm", (cX - 50, cY), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add area text
                cv2.putText(output_image, f"area: {area_cm2:.2f} cm²", (cX - 50, cY + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                results.append(result)
            except Exception as contour_process_error:
                print(f"[ERROR] Error processing contour {i+1}: {str(contour_process_error)}")
                # Continue with next contour
                continue
        
        # Check if we found any valid results
        if not results:
            return jsonify({'error': 'Failed to calculate measurements for any shapes'}), 400
        
        # Sort by area (largest first)
        results.sort(key=lambda x: x['area_cm2'], reverse=True)
        
        # Save the output image
        output_filename = f"size_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, output_image)
        print(f"[DEBUG] Saved output image to {output_path}")
        
        # Return success response
        return jsonify({
            'success': True,
            'measurements': results,
            'processedImage': '/static/uploads/' + output_filename
        })
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[ERROR] Error in size detection: {str(e)}\n{error_trace}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Initialize application resources
        initialize_app()
        
        # Run the Flask application
        print("Starting shape detection server on http://0.0.0.0:5000")
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        print(f"Error starting the server: {str(e)}")
        traceback.print_exc() 