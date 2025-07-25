import os
import sys
import torch
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from utils.visualization_utils import plot_cam, concat_images
from pytorch_grad_cam.utils.image import preprocess_image
import torchvision.transforms as transforms

# Add the HECTNet directory to Python path correctly
hectnet_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, hectnet_dir)

# Import HECTNet modules
import models.classification
from models.classification import ehfr_net
from models import get_model
from options.opts import get_eval_arguments
from utils import logger
from common import DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT

def create_model_with_config():
    """Create model with proper configuration - same as cvnets_evaluation.py"""
    
    # Get base arguments
    opts = get_eval_arguments()
    
    # Apply the EXACT same configuration as HECTNet cvnets_evaluation.py
    # Common settings
    setattr(opts, "common.auto_resume", True)
    setattr(opts, "common.mixed_precision", True)
    setattr(opts, "common.channels_last", False)
    setattr(opts, "common.tensorboard_logging", False)
    setattr(opts, "common.grad_clip", 10.0)
    setattr(opts, "common.accum_freq", 2)
    setattr(opts, "common.results_loc", "hectnet_results")
    setattr(opts, "common.run_label", "hectnet_multiscale_width050_100epochs")
    setattr(opts, "common.log_freq", 100)
    
    # Sampler settings
    setattr(opts, "sampler.name", "batch_sampler")
    setattr(opts, "sampler.bs.crop_size_width", 256)
    setattr(opts, "sampler.bs.crop_size_height", 256)
    
    # Dataset settings
    setattr(opts, "dataset.root_train", "C:/Users/User/OneDrive/Desktop/DATASETS/malaysian_food_processed/train")
    setattr(opts, "dataset.root_val", "C:/Users/User/OneDrive/Desktop/DATASETS/malaysian_food_processed/validation")
    setattr(opts, "dataset.name", "food_another")
    setattr(opts, "dataset.category", "classification")
    setattr(opts, "dataset.train_batch_size0", 8)
    setattr(opts, "dataset.val_batch_size0", 8)
    setattr(opts, "dataset.eval_batch_size0", 8)
    setattr(opts, "dataset.workers", 2)
    setattr(opts, "dataset.prefetch_factor", 2)
    setattr(opts, "dataset.persistent_workers", True)
    setattr(opts, "dataset.pin_memory", True)
    
    # Model settings - EXACT match with HECTNet configuration
    setattr(opts, "model.classification.name", "hectnet_multiscale")
    setattr(opts, "model.classification.n_classes", 5)
    
    # HECTNet multiscale specific parameters
    setattr(opts, "model.classification.hectnet_multiscale.width_multiplier", 0.5)
    setattr(opts, "model.classification.hectnet_multiscale.attn_norm_layer", "layer_norm_2d")
    setattr(opts, "model.classification.hectnet_multiscale.no_fuse_local_global_features", False)
    setattr(opts, "model.classification.hectnet_multiscale.no_ca", False)
    
    # Multiscale branch parameters
    setattr(opts, "model.classification.hectnet_multiscale.aux_dim", 32)
    setattr(opts, "model.classification.hectnet_multiscale.gabor_channels", 16)
    setattr(opts, "model.classification.hectnet_multiscale.fused_dim", 160)
    
    setattr(opts, "model.classification.activation.name", "hard_swish")
    setattr(opts, "model.normalization.name", "batch_norm")
    setattr(opts, "model.normalization.momentum", 0.1)
    setattr(opts, "model.activation.name", "hard_swish")
    setattr(opts, "model.layer.global_pool", "mean")
    setattr(opts, "model.layer.conv_init", "kaiming_normal")
    setattr(opts, "model.layer.conv_init_std_dev", 0.02)
    setattr(opts, "model.layer.linear_init", "trunc_normal")
    setattr(opts, "model.layer.linear_init_std_dev", 0.02)
    
    # Loss function settings
    setattr(opts, "loss.classification.name", "cross_entropy")
    setattr(opts, "loss.classification.label_smoothing", 0.1)
    setattr(opts, "loss.category", "classification")
    
    # GPU/CUDA settings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        setattr(opts, "dev.device", torch.device("cuda"))
        setattr(opts, "dev.num_gpus", torch.cuda.device_count())
        setattr(opts, "dev.device_id", 0)
    else:
        setattr(opts, "dev.device", torch.device("cpu"))
        setattr(opts, "dev.num_gpus", 0)
    
    # Initialize model
    model = get_model(opts)
    
    # Load trained weights - HECTNet checkpoint
    model_path = r'C:\Users\User\OneDrive\Desktop\Capstone Project\CODE\Snackly App\HECTNet\hectnet_results\hectnet_multiscale_width050_100epochs\checkpoint_best.pt'
    
    try:
        model_state_dict = torch.load(model_path, map_location=device)
        if 'model' in model_state_dict:
            model.load_state_dict(model_state_dict['model'])
        elif 'state_dict' in model_state_dict:
            model.load_state_dict(model_state_dict['state_dict'])
        else:
            model.load_state_dict(model_state_dict)
        print(f"Successfully loaded HECTNet model weights from {model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise
    
    model.to(device)
    model.eval()
    
    return model

def get_confidence_scores_hectnet_method(model, class_path, target_class_idx):
    """Calculate confidence scores using EXACT same method as hectnet_evaluation.py"""
    device = next(model.parameters()).device
    all_images = sorted(glob.glob(os.path.join(class_path, '*.jpg')))
    
    # Use EXACT same transform as hectnet_evaluation.py
    transform = transforms.Compose([
        transforms.Resize((DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)),  # 256x256
        transforms.ToTensor(),
        # NO normalization - this is commented out in hectnet_evaluation.py
    ])
    
    image_scores = []
    
    for image_path in all_images:
        try:
            # Use EXACT same preprocessing as hectnet_evaluation.py predict_image method
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Get model prediction - EXACT same as hectnet_evaluation.py
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence = probabilities[0][target_class_idx].item()
            
            image_scores.append((image_path, confidence))
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Sort by confidence score (highest first)
    image_scores.sort(key=lambda x: x[1], reverse=True)
    return image_scores

def plot_cam_with_proper_colormap(model, image_path, target_layer_list=None, size=(256, 256)):
    """Generate GradCAM with proper blue-to-red colormap"""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
    import matplotlib.pyplot as plt
    
    # Set target layers
    target_layers = [model.layer_5[-1]] if target_layer_list is None else target_layer_list
    
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # Load and preprocess image for GradCAM (this still needs normalization for GradCAM to work)
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, size)
        rgb_img = np.float32(rgb_img) / 255
        
        input_tensor = preprocess_image(rgb_img,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        
        # Generate GradCAM
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                          targets=None,
                          aug_smooth=True,
                          eigen_smooth=True)
        
        # Apply proper colormap manually
        grayscale_cam = grayscale_cam[0, :]
        
        # Use matplotlib's jet colormap (blue-green-yellow-orange-red) - fixed deprecation
        jet_colormap = plt.colormaps['jet']
        colored_cam = jet_colormap(grayscale_cam)
        colored_cam = colored_cam[:, :, :3]  # Remove alpha channel
        
        # Blend with original image
        cam_image = 0.4 * colored_cam + 0.6 * rgb_img
        cam_image = np.clip(cam_image, 0, 1)
        
        return rgb_img, cam_image

def create_gradcam_grid(model, class_name, class_path, target_class_idx, output_dir):
    """Create a 5x4 grid of top 10 GradCAM results for a given class"""
    print(f"\nProcessing {class_name}...")
    
    # Get confidence scores using EXACT same method as hectnet_evaluation.py
    image_scores = get_confidence_scores_hectnet_method(model, class_path, target_class_idx)
    
    # Debug: Print top 3 confidence scores
    print(f"Top 3 confidence scores for {class_name}:")
    for i, (img_path, score) in enumerate(image_scores[:3]):
        img_name = os.path.basename(img_path)
        print(f"  {i+1}. {img_name}: {score:.6f}")
    
    # Save confidence scores to file
    confidence_file = os.path.join(output_dir, f"{class_name}_confidence_scores.txt")
    with open(confidence_file, 'w') as f:
        f.write(f"Top 10 confidence scores for {class_name}:\n")
        for i, (img_path, score) in enumerate(image_scores[:10]):
            img_name = os.path.basename(img_path)
            f.write(f"{i+1}. {img_name}: {score:.6f}\n")
    
    # Get top 10 images
    top_10_images = image_scores[:10]
    
    # Create temporary directory for grid images
    temp_dir = os.path.join(output_dir, f"{class_name}_temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate and save individual images for the grid
    for i, (image_path, confidence) in enumerate(top_10_images):
        # Generate GradCAM
        original_img, cam_img = plot_cam_with_proper_colormap(model, image_path)
        
        # Convert to PIL for text overlay
        original_pil = Image.fromarray((original_img * 255).astype(np.uint8))
        cam_pil = Image.fromarray((cam_img * 255).astype(np.uint8))
        
        # Add confidence labels
        draw_orig = ImageDraw.Draw(original_pil)
        draw_cam = ImageDraw.Draw(cam_pil)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Add labels
        draw_orig.text((10, 10), f"Original {confidence:.3f}", fill="white", font=font)
        draw_cam.text((10, 10), f"GradCAM {confidence:.3f}", fill="white", font=font)
        
        # Save individual images
        original_pil.save(os.path.join(temp_dir, f"{i:02d}_original.jpg"))
        cam_pil.save(os.path.join(temp_dir, f"{i:02d}_gradcam.jpg"))
    
    # Use concat_images with correct parameters
    image_pattern = os.path.join(temp_dir, "*.jpg")
    
    # Create combine_cam directory if it doesn't exist
    combine_dir = os.path.join(temp_dir, "combine_cam")
    os.makedirs(combine_dir, exist_ok=True)
    
    # Call concat_images with correct parameters (4 columns for 5x4 grid)
    concat_images(
        image_path=image_pattern,
        size=(256, 256),
        num_column=4,  # 4 columns (original + gradcam pairs)
        padding=5
    )
    
    # Move the result to the final location
    source_path = os.path.join(combine_dir, "combine_cam.jpg")
    final_path = os.path.join(output_dir, f"{class_name}_top10_gradcam_grid.jpg")
    
    if os.path.exists(source_path):
        import shutil
        shutil.move(source_path, final_path)
        print(f"Grid saved to: {final_path}")
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
    else:
        print(f"Error: Could not create grid for {class_name}")
    
    return final_path

def main():
    # Define class mappings
    class_names = ['fried_rice', 'kaya_toast', 'nasi_lemak', 'roti_canai', 'satay']
    
    # Define test data paths
    test_data_base = r"C:\Users\User\OneDrive\Desktop\DATASETS\malaysian_food_processed\test"
    test_folders = {
        'fried_rice': os.path.join(test_data_base, 'fried_rice'),
        'kaya_toast': os.path.join(test_data_base, 'kaya_toast'),
        'nasi_lemak': os.path.join(test_data_base, 'nasi_lemak'),
        'roti_canai': os.path.join(test_data_base, 'roti_canai'),
        'satay': os.path.join(test_data_base, 'satay')
    }
    
    # Create output directory
    output_dir = "cam_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model
    print("Loading HECTNet model...")
    model = create_model_with_config()
    
    # Generate GradCAM grids for each class
    for class_idx, (class_name, class_path) in enumerate(test_folders.items()):
        if os.path.exists(class_path):
            create_gradcam_grid(model, class_name, class_path, class_idx, output_dir)
        else:
            print(f"Warning: Path {class_path} does not exist")
    
    print("\nGradCAM grid generation completed!")

if __name__ == "__main__":
    main()