import cv2
import numpy as np
from enhanced_color_detection import EnhancedColorDetector

def test_color_naming():
    """Test the color naming system with various RGB values"""
    print("\n===== Color Naming Test =====")
    
    # Create a detector
    detector = EnhancedColorDetector()
    
    # Define test colors as RGB values
    test_colors = [
        # The problematic color from the screenshot
        (196, 34, 79),  # Should be Ruby/Crimson/Pink, not Pastel Brown
        
        # Reds and pinks
        (255, 0, 0),    # Pure Red
        (220, 20, 60),  # Crimson
        (255, 105, 180), # Hot Pink
        (219, 112, 147), # Pale Violet Red
        
        # Browns for comparison
        (139, 69, 19),  # Saddle Brown
        (160, 82, 45),  # Sienna
        (210, 105, 30), # Chocolate
        
        # Other colors to test general naming
        (0, 128, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 0, 0),      # Black
        (255, 255, 255), # White
    ]
    
    # Test the color naming for each RGB value
    print("Testing color naming:")
    print("-" * 50)
    print("RGB Value".ljust(20) + "| Color Name")
    print("-" * 50)
    for rgb in test_colors:
        color_name = detector.get_color_family(rgb)
        # Format RGB values and display
        rgb_str = f"({rgb[0]:3d}, {rgb[1]:3d}, {rgb[2]:3d})"
        print(f"{rgb_str.ljust(20)}| {color_name}")
    
    # Test with a visual representation
    print("\nCreating visual test image...")
    
    # Create an image with color swatches and their names
    swatch_size = 60
    padding = 10
    swatches_per_row = 5
    rows = (len(test_colors) + swatches_per_row - 1) // swatches_per_row
    
    # Calculate image dimensions
    img_width = (swatch_size + padding) * swatches_per_row + padding
    img_height = (swatch_size + padding + 20) * rows + padding
    
    # Create white image
    test_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    # Draw color swatches with names
    for i, rgb in enumerate(test_colors):
        row = i // swatches_per_row
        col = i % swatches_per_row
        
        # Calculate swatch position
        x1 = padding + col * (swatch_size + padding)
        y1 = padding + row * (swatch_size + padding + 20)
        x2 = x1 + swatch_size
        y2 = y1 + swatch_size
        
        # Draw colored swatch
        test_img[y1:y2, x1:x2] = rgb[::-1]  # RGB to BGR for OpenCV
        
        # Add black border
        cv2.rectangle(test_img, (x1, y1), (x2, y2), (0, 0, 0), 1)
        
        # Get color name
        color_name = detector.get_color_family(rgb)
        
        # Add color name text
        cv2.putText(test_img, color_name, (x1, y2 + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Save the test image
    output_path = 'color_naming_test.png'
    cv2.imwrite(output_path, test_img)
    print(f"Visual test image saved to: {output_path}")
    
    # Display the test image if not in CI environment
    try:
        cv2.imshow("Color Naming Test", test_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("Could not display image (likely running in a headless environment).")
    
    print("\nColor naming test complete!")

if __name__ == "__main__":
    test_color_naming() 