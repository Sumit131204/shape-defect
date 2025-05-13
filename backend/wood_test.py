import cv2
import numpy as np
from enhanced_color_detection import EnhancedColorDetector
import os

def test_wood_tones():
    """Test the color detection system with wood and brown tones"""
    print("\n===== Wood Tone Detection Test =====")
    
    # Create a detector
    detector = EnhancedColorDetector()
    
    # Define some common wood RGB values to test
    wood_tones = [
        # Light wood tones
        (222, 184, 135),  # Tan / Light wood
        (210, 180, 140),  # Light beige / maple
        (245, 222, 179),  # Light tan / birch
        
        # Medium wood tones
        (205, 133, 63),   # Peru / medium oak
        (184, 134, 11),   # Dark goldenrod / medium wood
        (160, 120, 90),   # Medium brown / walnut
        
        # Dark wood tones
        (139, 69, 19),    # Saddle brown / dark wood
        (101, 67, 33),    # Dark brown / mahogany
        (85, 60, 42),     # Dark walnut
        
        # Reddish wood
        (165, 42, 42),    # Brown with red tint / cherry
        (128, 70, 27),    # Brown with slight red / rosewood
    ]
    
    print("Testing standard wood RGB values:")
    for rgb in wood_tones:
        color_name = detector.get_color_family(rgb)
        # Format RGB values to be evenly spaced
        rgb_str = f"({rgb[0]:3d}, {rgb[1]:3d}, {rgb[2]:3d})"
        print(f"  RGB {rgb_str} -> {color_name}")
    
    # Test with the modified dominant color extraction
    print("\nTesting dominant color extraction from sample images:")
    
    # Create a sample wooden circle image for testing
    circle_img = np.ones((300, 300, 3), dtype=np.uint8) * 255  # White background
    # Draw a wooden-colored circle
    wood_color = (90, 120, 160)  # BGR format (medium brown)
    cv2.circle(circle_img, (150, 150), 100, wood_color, -1)
    
    # Create a contour for the circle
    gray = cv2.cvtColor(circle_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Test the dominant color extraction
    dominant_color_bgr = detector.get_dominant_color(contours[0], circle_img)
    dominant_color_rgb = tuple(reversed(dominant_color_bgr))
    color_name = detector.get_color_family(dominant_color_rgb)
    print(f"  Synthetic circle (BGR {wood_color}): detected as {color_name}, RGB {dominant_color_rgb}")
    
    # Try to find and process images in the test directory
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_images')
    if os.path.exists(test_dir):
        for filename in os.listdir(test_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(test_dir, filename)
                try:
                    # Process the image
                    result_img, colors = detector.detect_colors(image_path)
                    print(f"  Image '{filename}':")
                    for color_info in colors:
                        print(f"    - {color_info['color']}, RGB: {color_info['rgb']}")
                except Exception as e:
                    print(f"  Error processing image '{filename}': {e}")
    else:
        print(f"  No test_images directory found at {test_dir}")
    
    print("\nWood tone detection test complete!")

if __name__ == "__main__":
    test_wood_tones() 