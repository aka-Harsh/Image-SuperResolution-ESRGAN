import os
import glob
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from model.esrgan import enhance_image

def process_image(input_path, output_folder, weights_path):
    """Process a single image with ESRGAN"""
    # Generate a unique ID
    image_id = str(uuid.uuid4())[:8]
    
    # Get the filename and extension
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    
    # Create output path
    output_path = os.path.join(output_folder, f"{name}_enhanced{ext}")
    
    # Enhance the image
    start_time = time.time()
    success = enhance_image(input_path, output_path, weights_path)
    end_time = time.time()
    
    if success:
        processing_time = end_time - start_time
        return {
            'id': image_id,
            'input_path': input_path,
            'output_path': output_path,
            'success': True,
            'processing_time': processing_time
        }
    else:
        return {
            'id': image_id,
            'input_path': input_path,
            'output_path': None,
            'success': False,
            'processing_time': 0
        }

def batch_process_folder(input_folder, output_folder, weights_path, max_workers=2):
    """Process all images in a folder using ESRGAN"""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    if not image_files:
        return {'success': False, 'message': 'No image files found in the folder', 'results': []}
    
    # Process images in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image, img_path, output_folder, weights_path): img_path for img_path in image_files}
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                img_path = futures[future]
                results.append({
                    'id': str(uuid.uuid4())[:8],
                    'input_path': img_path,
                    'output_path': None,
                    'success': False,
                    'error': str(e),
                    'processing_time': 0
                })
    
    # Calculate statistics
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    total_time = sum(r['processing_time'] for r in results)
    avg_time = total_time / len(results) if results else 0
    
    return {
        'success': True,
        'total_images': len(results),
        'successful': successful,
        'failed': failed,
        'total_processing_time': total_time,
        'average_processing_time': avg_time,
        'results': results
    }