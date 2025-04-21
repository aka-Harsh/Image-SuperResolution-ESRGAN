import os
from werkzeug.utils import secure_filename
import glob

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Check if the filename has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file, upload_folder, image_id):
    """Save the uploaded file to the uploads folder with a unique name"""
    if file and allowed_file(file.filename):
        # Get the file extension
        ext = file.filename.rsplit('.', 1)[1].lower()
        
        # Create a new filename with the image ID
        filename = f"{image_id}.{ext}"
        
        # Save the file
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        
        return filename
    return None

def get_image_path(folder, image_id):
    """Find an image in the folder by its ID part of the filename"""
    # Look for any file starting with the image_id
    pattern = os.path.join(folder, f"{image_id}.*")
    matching_files = glob.glob(pattern)
    
    # Return the first match if any found
    if matching_files:
        return matching_files[0]
    return None