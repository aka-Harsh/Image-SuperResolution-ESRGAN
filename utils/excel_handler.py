import os
import pandas as pd
from datetime import datetime

EXCEL_FILE = 'user_ratings.xlsx'

def initialize_excel():
    """Initialize the Excel file if it doesn't exist"""
    if not os.path.exists(EXCEL_FILE):
        # Create DataFrame with columns
        df = pd.DataFrame(columns=[
            'ID', 
            'Timestamp', 
            'Rating',
            'Original_Size_KB',
            'Enhanced_Size_KB',
            'Enhancement_Model'
        ])
        
        # Save to Excel
        df.to_excel(EXCEL_FILE, index=False)
        print(f"Created new Excel file: {EXCEL_FILE}")

def update_excel(image_id, rating, original_size_kb=0, enhanced_size_kb=0, model="ESRGAN"):
    """Update the Excel file with new rating data"""
    # Make sure the Excel file exists
    initialize_excel()
    
    # Read existing data
    df = pd.read_excel(EXCEL_FILE)
    
    # Create new row
    new_row = {
        'ID': image_id,
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Rating': rating,
        'Original_Size_KB': round(original_size_kb, 2),
        'Enhanced_Size_KB': round(enhanced_size_kb, 2),
        'Enhancement_Model': model
    }
    
    # Append the new row
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save back to Excel
    df.to_excel(EXCEL_FILE, index=False)
    print(f"Updated Excel file with rating for image ID: {image_id}")
    
    return True

def get_all_ratings():
    """Get all ratings from the Excel file"""
    # Make sure the Excel file exists
    initialize_excel()
    
    # Read data
    df = pd.read_excel(EXCEL_FILE)
    
    return df.to_dict('records')