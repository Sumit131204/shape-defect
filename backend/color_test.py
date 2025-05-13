import os
import cv2
import numpy as np
from enhanced_color_detection import EnhancedColorDetector
import pandas as pd

def test_color_detection():
    # Test 1: Check if colour2.xlsx file exists and can be loaded
    print("\nTest 1: Testing colour2.xlsx loading")
    excel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'colour2.xlsx')
    if os.path.exists(excel_path):
        print(f"✅ File found at: {excel_path}")
        
        try:
            df = pd.read_excel(excel_path)
            print(f"✅ Successfully loaded Excel file with {len(df)} rows")
            print(f"Excel columns: {df.columns.tolist()}")
            
            # Check for required columns
            required_columns = ['Color_name', 'Hex', 'D2', 'D1', 'D0']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if not missing_columns:
                print(f"✅ All required columns found in the Excel file")
            else:
                print(f"❌ Missing columns in Excel file: {missing_columns}")
                
            # Print first few rows for inspection
            print("\nSample data from Excel:")
            sample = df.head(3)
            for _, row in sample.iterrows():
                print(f"Color: {row['Color_name']}, Hex: {row['Hex']}, RGB: ({row['D2']}, {row['D1']}, {row['D0']})")
                
        except Exception as e:
            print(f"❌ Error loading Excel file: {e}")
    else:
        print(f"❌ Excel file not found at expected path: {excel_path}")
    
    # Test 2: Initialize EnhancedColorDetector
    print("\nTest 2: Testing EnhancedColorDetector initialization")
    try:
        detector = EnhancedColorDetector()
        print(f"✅ Successfully initialized EnhancedColorDetector")
        print(f"Loaded {len(detector.color_dict)} colors")
        
        # Print sample colors from the detector
        print("\nSample colors from detector:")
        sample_colors = list(detector.color_dict.items())[:3]
        for name, data in sample_colors:
            print(f"  {name}: {data}")
            
    except Exception as e:
        print(f"❌ Error initializing EnhancedColorDetector: {e}")
        
    # Test 3: Test closest color matching
    print("\nTest 3: Testing color matching")
    try:
        if 'detector' in locals():
            test_colors = [
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Yellow
                (128, 128, 128) # Gray
            ]
            
            print("Testing color matching with sample RGB values:")
            for rgb in test_colors:
                color_name = detector.get_color_family(rgb)
                print(f"RGB {rgb} -> Matched color: {color_name}")
                
    except Exception as e:
        print(f"❌ Error testing color matching: {e}")
    
    print("\nColor detection testing complete!")

if __name__ == "__main__":
    test_color_detection() 