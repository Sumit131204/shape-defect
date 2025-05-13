import pandas as pd
import os

try:
    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(script_dir, 'colors.xlsx')
    
    print(f"Reading Excel file: {excel_path}")
    print(f"File exists: {os.path.exists(excel_path)}")
    
    # Read the Excel file
    df = pd.read_excel(excel_path)
    
    # Print information about the dataframe
    print(f"\nDataFrame info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Print the first few rows
    print("\nFirst 3 rows:")
    print(df.head(3))
    
    # Check for specific columns
    expected_columns = ['Color_name', 'Hex', 'D0', 'D1', 'D2']
    for col in expected_columns:
        print(f"Column '{col}' exists: {col in df.columns}")
    
    # If Color_name exists, show some sample color names
    if 'Color_name' in df.columns:
        print("\nSample color names:")
        for name in df['Color_name'].head(5):
            print(f"  {name}")
    
except Exception as e:
    print(f"Error: {str(e)}") 