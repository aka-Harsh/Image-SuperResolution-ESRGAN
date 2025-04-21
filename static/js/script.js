document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const uploadForm = document.getElementById('upload-form');
    const uploadArea = document.getElementById('upload-area');
    const imageInput = document.getElementById('image-input');
    const previewSection = document.getElementById('preview-section');
    const originalImage = document.getElementById('original-image');
    const originalDetails = document.getElementById('original-details');
    const enhancedContainer = document.getElementById('enhanced-container');
    const enhancedImage = document.getElementById('enhanced-image');
    const enhancedDetails = document.getElementById('enhanced-details');
    const enhanceButton = document.getElementById('enhance-button');
    const compareBtn = document.getElementById('compare-btn');
    const ratingSection = document.getElementById('rating-section');
    const stars = document.querySelectorAll('.star');
    const submitRatingButton = document.getElementById('submit-rating');
    const downloadLink = document.getElementById('download-link');
    const statusMessage = document.getElementById('status-message');
    const loadingSpinner = document.getElementById('loading-spinner');
    
    // Store the current image ID and selected rating
    let currentImageId = null;
    let selectedRating = null;
    
    // Initially hide the preview section
    previewSection.style.display = 'none';
    enhancedContainer.style.display = 'none';
    
    // Define setupDragAndDrop BEFORE calling it
    function setupDragAndDrop() {
        console.log("Setting up drag and drop"); // Debug message
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            uploadArea.classList.add('dragover');
        }
        
        function unhighlight() {
            uploadArea.classList.remove('dragover');
        }
        
        uploadArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                imageInput.files = files; // Update the file input
                uploadImage(files[0]);
            }
        }
        
        // Also add click handler to the upload area to trigger file input
        uploadArea.addEventListener('click', function(e) {
            // Don't trigger if they clicked on the actual button
            if (e.target.tagName !== 'LABEL' && !e.target.closest('label')) {
                imageInput.click();
            }
        });
    }
    
    // Now call setupDragAndDrop after it's defined
    setupDragAndDrop();
    
    // Handle file selection change
    imageInput.addEventListener('change', function() {
        console.log("File input change detected"); // Debug message
        if (this.files && this.files[0]) {
            const file = this.files[0];
            console.log("Selected file:", file.name); // Debug message
            uploadImage(file);
        }
    });
    
    // Handle file upload form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Check if a file is selected
        const file = imageInput.files[0];
        if (!file) {
            showStatusMessage('Please select an image file.', 'error');
            return;
        }
        
        uploadImage(file);
    });
    
    // Handle enhance button click
    enhanceButton.addEventListener('click', function() {
        // Check if an image is uploaded
        if (!currentImageId) {
            showStatusMessage('Please upload an image first.', 'error');
            return;
        }
        
        // Show loading spinner
        loadingSpinner.style.display = 'flex';
        
        // Send the enhance request
        fetch('/enhance', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image_id: currentImageId
            })
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading spinner
            loadingSpinner.style.display = 'none';
            
            if (data.error) {
                showStatusMessage(data.error, 'error');
                return;
            }
            
            // Show the enhanced image with fade-in animation
            enhancedContainer.style.display = 'block';
            enhancedContainer.style.opacity = 0;
            setTimeout(() => {
                enhancedContainer.style.opacity = 1;
            }, 100);
            
            // Set the enhanced image source with timestamp to prevent caching
            enhancedImage.src = data.enhanced_path + '?t=' + new Date().getTime();
            
            // Set the download link
            downloadLink.href = data.enhanced_path;
            
            // Format file sizes for display
            const originalSizeFormatted = formatFileSize(data.original_size);
            const enhancedSizeFormatted = formatFileSize(data.enhanced_size);
            
            // Update details
            enhancedDetails.innerHTML = `
                <strong>Enhanced Size:</strong> ${enhancedSizeFormatted}<br>
                <strong>Upscaling Factor:</strong> 4x<br>
                <strong>Model:</strong> ESRGAN
            `;
            
            // Show the rating section with animation
            ratingSection.style.display = 'block';
            ratingSection.style.opacity = 0;
            setTimeout(() => {
                ratingSection.style.opacity = 1;
            }, 300);
            
            // Show success message with file size info
            showStatusMessage(`Image enhanced successfully! Original: ${originalSizeFormatted}, Enhanced: ${enhancedSizeFormatted}`, 'success');
            
            // Apply pulse animation to the enhanced image
            enhancedImage.style.animation = 'pulse 2s';
            
            // Reset animation after it completes
            setTimeout(() => {
                enhancedImage.style.animation = '';
            }, 2000);
        })
        .catch(error => {
            // Hide loading spinner
            loadingSpinner.style.display = 'none';
            
            console.error('Error:', error);
            showStatusMessage('An error occurred while enhancing the image.', 'error');
        });
    });
    
    // Handle star rating selection
    stars.forEach(star => {
        star.addEventListener('click', function() {
            // Get the rating value
            selectedRating = this.dataset.rating;
            
            // Remove selected class from all stars
            stars.forEach(s => s.classList.remove('selected'));
            
            // Add selected class to this star and all previous stars
            stars.forEach(s => {
                if (s.dataset.rating <= selectedRating) {
                    s.classList.add('selected');
                }
            });
        });
    });
    
    // Handle rating submission
    submitRatingButton.addEventListener('click', function() {
        // Check if a rating is selected
        if (!selectedRating) {
            showStatusMessage('Please select a rating first.', 'error');
            return;
        }
        
        // Show loading spinner
        loadingSpinner.style.display = 'flex';
        
        // Send the rating request
        fetch('/rate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image_id: currentImageId,
                rating: selectedRating
            })
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading spinner
            loadingSpinner.style.display = 'none';
            
            if (data.error) {
                showStatusMessage(data.error, 'error');
                return;
            }
            
            // Show success message
            showStatusMessage('Thank you for your feedback! Your rating has been saved.', 'success');
            
            // Hide the rating section with fade-out animation
            ratingSection.style.opacity = 0;
            setTimeout(() => {
                ratingSection.style.display = 'none';
            }, 500);
        })
        .catch(error => {
            // Hide loading spinner
            loadingSpinner.style.display = 'none';
            
            console.error('Error:', error);
            showStatusMessage('An error occurred while submitting the rating.', 'error');
        });
    });
    
    // Handle compare button click for before/after comparison
    compareBtn.addEventListener('click', function() {
        // Simplified comparison toggle
        if (this.innerHTML.includes('Show Original')) {
            // Currently showing enhanced, switch to original
            this.innerHTML = '<i class="fas fa-exchange-alt"></i> Show Enhanced';
            enhancedImage.style.display = 'none';
            originalImage.style.display = 'block';
        } else {
            // Currently showing original, switch to enhanced
            this.innerHTML = '<i class="fas fa-exchange-alt"></i> Show Original';
            originalImage.style.display = 'none';
            enhancedImage.style.display = 'block';
        }
    });
    
    // Function to upload an image
    function uploadImage(file) {
        console.log("Uploading file:", file.name); // Debug message
        
        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/gif'];
        if (!validTypes.includes(file.type)) {
            showStatusMessage('Please select a valid image file (JPG, PNG, GIF).', 'error');
            return;
        }
        
        // Create form data
        const formData = new FormData();
        formData.append('image', file);
        
        // Show loading spinner
        loadingSpinner.style.display = 'flex';
        
        // Create a preview immediately
        const reader = new FileReader();
        reader.onload = function(e) {
            console.log("File preview loaded"); // Debug message
            originalImage.src = e.target.result;
            
            // Show the preview section immediately
            previewSection.style.display = 'block';
            previewSection.style.opacity = 1;
            
            // Get image dimensions
            const img = new Image();
            img.onload = function() {
                console.log("Image dimensions:", this.width, "x", this.height); // Debug message
                originalDetails.innerHTML = `
                    <strong>Dimensions:</strong> ${this.width}x${this.height}<br>
                    <strong>File Type:</strong> ${file.type.split('/')[1].toUpperCase()}<br>
                    <strong>File Size:</strong> ${formatFileSize(file.size)}
                `;
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
        
        // Send the upload request
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading spinner
            loadingSpinner.style.display = 'none';
            
            if (data.error) {
                showStatusMessage(data.error, 'error');
                return;
            }
            
            // Store the image ID
            currentImageId = data.id;
            
            // Make sure preview is visible
            previewSection.style.display = 'block';
            previewSection.style.opacity = 1;
            
            // Hide the enhanced image container
            enhancedContainer.style.display = 'none';
            
            // Hide the rating section
            ratingSection.style.display = 'none';
            
            // Reset stars
            stars.forEach(s => s.classList.remove('selected'));
            selectedRating = null;
            
            // Scroll to the preview section
            previewSection.scrollIntoView({ behavior: 'smooth' });
            
            // Show success message
            showStatusMessage('Image uploaded successfully! Click "Enhance Image" to process it.', 'success');
        })
        .catch(error => {
            // Hide loading spinner
            loadingSpinner.style.display = 'none';
            
            console.error('Error:', error);
            showStatusMessage('An error occurred while uploading the image.', 'error');
        });
    }
    
    // Helper function to show status messages
    function showStatusMessage(message, type) {
        statusMessage.textContent = message;
        statusMessage.className = 'status-message ' + type;
        statusMessage.style.display = 'block';
        
        // Add animation
        statusMessage.style.animation = 'fadeIn 0.3s ease-in-out';
        
        // Reset animation after it completes
        setTimeout(() => {
            statusMessage.style.animation = '';
        }, 300);
        
        // Hide the message after 5 seconds for success/info messages
        if (type === 'success' || type === 'info') {
            setTimeout(() => {
                fadeOut(statusMessage);
            }, 5000);
        }
    }
    
    // Helper function to fade out an element
    function fadeOut(element) {
        element.style.opacity = '1';
        
        (function fade() {
            if ((element.style.opacity -= 0.1) < 0) {
                element.style.display = 'none';
            } else {
                requestAnimationFrame(fade);
            }
        })();
    }
    
    // Helper function to format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
});