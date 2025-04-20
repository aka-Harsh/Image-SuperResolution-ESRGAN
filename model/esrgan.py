import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import urllib.request
import ssl

# Check for CUDA availability and print GPU info
def check_cuda():
    """Check if CUDA is available and print GPU info"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA is available! Found {device_count} GPU(s).")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {device_name}")
        return True
    else:
        print("CUDA is not available. Running on CPU.")
        return False

# Run the CUDA check on import
cuda_available = check_cuda()

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # Upsampling layers
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

def download_weights(model_path):
    """Download pre-trained ESRGAN weights if they don't exist"""
    if os.path.exists(model_path):
        print(f"Weights file already exists at {model_path}")
        return True
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # ESRGAN weights URL (this is an example URL - replace with the actual URL)
    weights_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    
    print(f"Downloading weights from {weights_url}...")
    try:
        # Create SSL context that ignores certificate validation
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(weights_url, context=context) as response, open(model_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Weights downloaded to {model_path}")
        return True
    except Exception as e:
        print(f"Error downloading weights: {e}")
        return False

# Load model function
def load_model(weights_path):
    """Load the ESRGAN model with pre-trained weights"""
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = RRDBNet().to(device)
    
    # First try to download the weights if they don't exist
    if not os.path.exists(weights_path):
        success = download_weights(weights_path)
        if not success:
            print(f"Failed to download weights to {weights_path}")
            return None, device
        
    try:
        # Load state dict
        state_dict = torch.load(weights_path, map_location=device)
        
        # Handle different state dict structures
        if 'params_ema' in state_dict:
            # Real-ESRGAN format
            state_dict = state_dict['params_ema']
        
        # Create new OrderedDict without 'module.' prefix if it exists
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        # Try to load state dict - this might fail if the model architecture doesn't match
        try:
            model.load_state_dict(new_state_dict)
        except Exception as e:
            print(f"Error loading state dict directly: {e}")
            print("Trying to load with flexible keys...")
            
            # More flexible loading - load only the keys that match
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
        model.eval()
        return model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, device

def preprocess_image(img, device):
    """Preprocess the image for input to the model"""
    # Resize if the image is too large
    # CPU can handle smaller images well, GPU can handle larger ones
    max_size = 800 if device.type == 'cpu' else 1500
    print(f"Using max image size: {max_size} for {device.type}")
    
    if img.width > max_size or img.height > max_size:
        ratio = max(img.width / max_size, img.height / max_size)
        new_width = int(img.width / ratio)
        new_height = int(img.height / ratio)
        print(f"Resizing image from {img.width}x{img.height} to {new_width}x{new_height}")
        img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Convert to tensor
    img = np.array(img).astype(np.float32) / 255.
    img = torch.from_numpy(img).float()
    
    # HWC to NCHW
    img = img.permute(2, 0, 1).unsqueeze(0)
    
    # Move tensor to the same device as model
    img = img.to(device)
    
    return img

def postprocess_tensor(tensor):
    """Convert the output tensor to a PIL Image"""
    # Make sure tensor is on CPU for numpy conversion
    tensor = tensor.cpu()
    
    # NCHW to HWC
    img = tensor.squeeze().float().clamp_(0, 1).permute(1, 2, 0).numpy()
    
    # Convert to uint8 and create PIL Image
    img = (img * 255.0).round().astype(np.uint8)
    output_img = Image.fromarray(img)
    
    return output_img

def fallback_enhance(input_path, output_path):
    """
    Fallback enhancement using PIL's built-in resizing when ESRGAN model fails to load
    """
    try:
        print("Using fallback image enhancement method")
        # Load image
        img = Image.open(input_path).convert('RGB')
        
        # Get dimensions
        width, height = img.size
        
        # Resize using Lanczos filtering (high quality)
        enhanced_img = img.resize((width*4, height*4), Image.LANCZOS)
        
        # Apply light sharpening to improve details
        from PIL import ImageFilter, ImageEnhance
        
        # Sharpen
        enhanced_img = enhanced_img.filter(ImageFilter.SHARPEN)
        
        # Increase contrast slightly
        enhancer = ImageEnhance.Contrast(enhanced_img)
        enhanced_img = enhancer.enhance(1.2)
        
        # Save output
        enhanced_img.save(output_path)
        print("Fallback enhancement completed successfully")
        return True
    except Exception as e:
        print(f"Error in fallback enhancement: {e}")
        return False

def enhance_image(input_path, output_path, weights_path='model/weights/RRDB_ESRGAN_x4.pth'):
    """
    Enhance an image using ESRGAN model
    """
    try:
        # Load the model
        model, device = load_model(weights_path)
        if model is None:
            print("Model failed to load, using fallback enhancement")
            return fallback_enhance(input_path, output_path)
            
        # Load and preprocess the image
        img = Image.open(input_path).convert('RGB')
        img_tensor = preprocess_image(img, device)
        
        # Perform inference
        print(f"Running inference on {device.type}...")
        with torch.no_grad():
            output = model(img_tensor)
        
        # Postprocess the output tensor
        output_img = postprocess_tensor(output)
        
        # Save the enhanced image
        output_img.save(output_path)
        print("Enhancement completed successfully")
        
        # Clean up memory if using CUDA
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        print(f"Error enhancing image: {e}")
        print("Trying fallback enhancement...")
        return fallback_enhance(input_path, output_path)