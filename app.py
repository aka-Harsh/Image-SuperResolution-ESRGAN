from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import uuid
from datetime import datetime
import torch
import gc  # Garbage collector for managing memory
from utils.excel_handler import update_excel
from model.esrgan import enhance_image, check_cuda
from utils.image_utils import save_uploaded_file, get_image_path
from utils.batch_processor import batch_process_folder

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ENHANCED_FOLDER'] = 'static/enhanced'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ENHANCED_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Generate unique ID
    image_id = str(uuid.uuid4())[:8]
    
    # Save the uploaded file
    filename = save_uploaded_file(file, app.config['UPLOAD_FOLDER'], image_id)
    
    # Return the image ID and filename
    return jsonify({
        'id': image_id,
        'filename': filename,
        'upload_path': f"{app.config['UPLOAD_FOLDER']}/{filename}"
    })

@app.route('/enhance', methods=['POST'])
def enhance():
    data = request.json
    image_id = data.get('image_id')
    
    if not image_id:
        return jsonify({'error': 'No image ID provided'}), 400
    
    # Find the uploaded image
    uploaded_image_path = get_image_path(app.config['UPLOAD_FOLDER'], image_id)
    if not uploaded_image_path:
        return jsonify({'error': 'Image not found'}), 404
    
    # Generate enhanced image filename
    enhanced_filename = f"{image_id}_enhanced.png"
    enhanced_path = f"{app.config['ENHANCED_FOLDER']}/{enhanced_filename}"
    
    # Enhance image using ESRGAN
    success = enhance_image(uploaded_image_path, enhanced_path)
    
    # Clean up GPU memory after processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    if success:
        # Calculate file sizes for comparison
        original_size = os.path.getsize(uploaded_image_path)
        enhanced_size = os.path.getsize(enhanced_path)
        
        return jsonify({
            'id': image_id,
            'enhanced_path': f"{app.config['ENHANCED_FOLDER']}/{enhanced_filename}",
            'original_size': original_size,
            'enhanced_size': enhanced_size
        })
    else:
        return jsonify({'error': 'Enhancement failed'}), 500

@app.route('/rate', methods=['POST'])
def rate_image():
    data = request.json
    image_id = data.get('image_id')
    rating = data.get('rating')
    
    if not image_id or rating is None:
        return jsonify({'error': 'Missing image ID or rating'}), 400
    
    # Get file paths
    uploaded_image_path = get_image_path(app.config['UPLOAD_FOLDER'], image_id)
    enhanced_image_path = get_image_path(app.config['ENHANCED_FOLDER'], image_id + '_enhanced')
    
    # Get file sizes in KB
    original_size_kb = os.path.getsize(uploaded_image_path) / 1024 if uploaded_image_path else 0
    enhanced_size_kb = os.path.getsize(enhanced_image_path) / 1024 if enhanced_image_path else 0
    
    # Update Excel file with rating and additional information
    update_excel(
        image_id, 
        rating, 
        original_size_kb=original_size_kb, 
        enhanced_size_kb=enhanced_size_kb,
        model="ESRGAN"
    )
    
    return jsonify({'success': True, 'message': 'Rating saved'})

@app.route('/batch', methods=['GET'])
def batch_page():
    return render_template('batch.html')

@app.route('/batch-process', methods=['POST'])
def batch_process():
    # Get the input and output folders from the request
    data = request.json
    input_folder = data.get('input_folder')
    output_folder = data.get('output_folder', 'static/batch_output')
    
    # Make sure the input folder exists
    if not os.path.exists(input_folder):
        return jsonify({'error': f'Input folder {input_folder} does not exist'}), 400
    
    # Process all images in the folder
    weights_path = 'model/weights/RRDB_ESRGAN_x4.pth'
    result = batch_process_folder(input_folder, output_folder, weights_path)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)