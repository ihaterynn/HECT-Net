import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import argparse
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from itertools import cycle
import cv2
import random
from matplotlib.patches import Rectangle

# Add the CVnets directory to Python path correctly
cvnets_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cvnets_dir)

# Add these imports at the very top, before any other CVnets imports
import models.classification  # Force registration of all classification models
from models.classification import ehfr_net  # Explicitly import ehfr_net

# Add this debug print to verify registration
from models.classification import CLS_MODEL_REGISTRY
print(f"DEBUG: Available models in registry: {list(CLS_MODEL_REGISTRY.keys())}")

from models import get_model
from options.opts import get_eval_arguments
from utils import logger
from common import DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT

class CVnetsEvaluator:
    def __init__(self, model_path, test_folders_dict, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_path_for_report = model_path
        self.test_folders_dict = test_folders_dict
        self.class_names = list(test_folders_dict.keys())

        print(f"Model path: {model_path}")
        print(f"Test folders dictionary: {test_folders_dict}")

        # Use the same configuration approach as main_train.py
        opts = get_eval_arguments()
        
        # Apply the EXACT same configuration as main_train.py
        # Common settings
        setattr(opts, "common.auto_resume", True)
        setattr(opts, "common.mixed_precision", True)
        setattr(opts, "common.channels_last", False)
        setattr(opts, "common.tensorboard_logging", False)
        setattr(opts, "common.grad_clip", 10.0)
        setattr(opts, "common.accum_freq", 2)
        setattr(opts, "common.results_loc", "hectnet_results")
        setattr(opts, "common.run_label", "hectnet_width050_100epochs")  
        setattr(opts, "common.log_freq", 100)
        
        # Sampler settings
        setattr(opts, "sampler.name", "batch_sampler")
        setattr(opts, "sampler.bs.crop_size_width", 256)
        setattr(opts, "sampler.bs.crop_size_height", 256)
        
        # Image augmentation settings
        setattr(opts, "image_augmentation.resize.size", 256)
        setattr(opts, "image_augmentation.resize.enable", True)
        setattr(opts, "image_augmentation.resize.interpolation", "bicubic")
        setattr(opts, "image_augmentation.center_crop.enable", True)
        setattr(opts, "image_augmentation.center_crop.size", 256)
        setattr(opts, "image_augmentation.random_resized_crop.enable", True)
        setattr(opts, "image_augmentation.random_resized_crop.interpolation", "bicubic")
        setattr(opts, "image_augmentation.random_horizontal_flip.enable", True)
        setattr(opts, "image_augmentation.rand_augment.enable", True)
        setattr(opts, "image_augmentation.random_erase.enable", True)
        setattr(opts, "image_augmentation.random_erase.p", 0.50)
        setattr(opts, "image_augmentation.mixup.enable", True)
        setattr(opts, "image_augmentation.mixup.alpha", 0.2)
        setattr(opts, "image_augmentation.cutmix.enable", True)
        setattr(opts, "image_augmentation.cutmix.alpha", 1.0)
        
        # Dataset settings - match training exactly
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
        
        # Model settings - EXACT match with NEW training configuration
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
        
        # Optimizer settings
        setattr(opts, "optimizer.name", "adamw")
        setattr(opts, "optimizer.weight_decay", 0.005)
        setattr(opts, "optimizer.no_decay_bn_filter_bias", True)
        setattr(opts, "optimizer.adamw.beta1", 0.9)
        setattr(opts, "optimizer.adamw.beta2", 0.999)
        
        # Training/Scheduler settings
        setattr(opts, "scheduler.name", "cosine")
        setattr(opts, "scheduler.max_epochs", 100)
        setattr(opts, "scheduler.is_iteration_based", False)
        setattr(opts, "scheduler.warmup_iterations", 1000)
        setattr(opts, "scheduler.warmup_init_lr", 1e-6)
        setattr(opts, "scheduler.cosine.max_lr", 0.0005)
        setattr(opts, "scheduler.cosine.min_lr", 0.00005)
        
        # Stats settings
        setattr(opts, "stats.val", ["loss", "top1", "top5"])
        setattr(opts, "stats.train", ["loss"])
        setattr(opts, "stats.checkpoint_metric", "top1")
        setattr(opts, "stats.checkpoint_metric_max", True)
        
        # EMA settings
        setattr(opts, "ema.enable", True)
        setattr(opts, "ema.momentum", 0.0005)
        
        # GPU/CUDA settings
        if torch.cuda.is_available():
            setattr(opts, "dev.device", torch.device("cuda"))
            setattr(opts, "dev.num_gpus", torch.cuda.device_count())
            setattr(opts, "dev.device_id", 0)
            print(f"CUDA is available! Using {torch.cuda.device_count()} GPU(s)")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Using CPU.")
            setattr(opts, "dev.device", torch.device("cpu"))
            setattr(opts, "dev.num_gpus", 0)
            
        print(f"DEBUG: Model Name: {getattr(opts, 'model.classification.name')}")
        # The line below was causing an error because it referred to ehfr_net, which is not the model being used.
        # It's better to access the parameter for the currently configured model.
        print(f"DEBUG: Width Multiplier: {getattr(opts, 'model.classification.hectnet_multiscale.width_multiplier')}")
        print(f"DEBUG: N_Classes: {getattr(opts, 'model.classification.n_classes')}")
        
        # Initialize model using the constructed opts
        self.model = get_model(opts)
        
        # Load trained weights
        try:
            model_state_dict = torch.load(model_path, map_location=device)
            if 'model' in model_state_dict:
                self.model.load_state_dict(model_state_dict['model'])
            elif 'state_dict' in model_state_dict:
                self.model.load_state_dict(model_state_dict['state_dict'])
            else:
                self.model.load_state_dict(model_state_dict)
            print(f"Successfully loaded model weights from {model_path}")
        except RuntimeError as e:
            print(f"RuntimeError loading state_dict: {e}")
            print("Attempting to load with strict=False")
            try:
                if 'model' in model_state_dict:
                    self.model.load_state_dict(model_state_dict['model'], strict=False)
                elif 'state_dict' in model_state_dict:
                    self.model.load_state_dict(model_state_dict['state_dict'], strict=False)
                else:
                    self.model.load_state_dict(model_state_dict, strict=False)
                print("Successfully loaded model weights with strict=False.")
            except Exception as e_strict_false:
                print(f"Error loading model weights even with strict=False: {e_strict_false}")
                raise
        except FileNotFoundError:
            print(f"ERROR: Model checkpoint file not found at {model_path}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while loading the model: {e}")
            raise
            
        self.model.to(device)
        self.model.eval()
        
        # Evaluation transforms (matching CVnets evaluation pipeline)
        self.transform = transforms.Compose([
            transforms.Resize((DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict_image(self, image_path):
        """Predict single image and return class probabilities"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]
    
    def evaluate_folder(self, folder_path, expected_class):
        """Evaluate all images in a folder"""
        results = []
        image_files = [f for f in os.listdir(folder_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in tqdm(image_files, desc=f"Processing {expected_class}"):
            img_path = os.path.join(folder_path, img_file)
            try:
                predicted_idx, confidence, probabilities = self.predict_image(img_path)
                predicted_class = self.class_names[predicted_idx]
                
                results.append({
                    'image_path': img_path,
                    'expected_class': expected_class,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'correct': predicted_class == expected_class
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
        return results
    
    def run_evaluation(self, data_folders_dict):
        """Run complete evaluation matching EHFR-Net output format"""
        all_results = []
        
        for expected_class, folder_path in data_folders_dict.items():
            if expected_class not in self.class_names:
                print(f"Warning: Class '{expected_class}' not in model classes. Skipping.")
                continue
            class_results = self.evaluate_folder(folder_path, expected_class)
            all_results.extend(class_results)
        
        if not all_results:
            print("No results to evaluate.")
            return
        
        # Calculate metrics
        correct = sum(1 for r in all_results if r['correct'])
        total = len(all_results)
        accuracy = correct / total if total > 0 else 0
        
        print(f"\nOverall Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"Total True Predictions: {correct}")
        print(f"Total False Predictions: {total - correct}")
        
        # Per-class accuracy
        for class_name in self.class_names:
            class_results = [r for r in all_results if r['expected_class'] == class_name]
            if class_results:
                class_correct = sum(1 for r in class_results if r['correct'])
                class_total = len(class_results)
                class_accuracy = class_correct / class_total
                print(f"{class_name} Accuracy: {class_accuracy:.4f} ({class_correct}/{class_total})")
        
        # Generate classification report
        y_true = [r['expected_class'] for r in all_results]
        y_pred = [r['predicted_class'] for r in all_results]
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Define output directory
        output_dir = "C:\\Users\\User\\OneDrive\\Desktop\\Capstone Project\\CODE\\Snackly App\\HECTNet\\hect_netperformance"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save classification report to file
        report_text = classification_report(y_true, y_pred, target_names=self.class_names)
        report_filepath = os.path.join(output_dir, 'cvnets_classification_report.txt')
        with open(report_filepath, 'w') as f:
            f.write(f"Model: {self.model_path_for_report}\n\n")
            f.write(f"Overall Accuracy: {accuracy:.4f} ({correct}/{total})\n")
            f.write(f"Total True Predictions: {correct}\n")
            f.write(f"Total False Predictions: {total - correct}\n\n")
            for class_name in self.class_names:
                class_results = [r for r in all_results if r['expected_class'] == class_name]
                if class_results:
                    class_correct = sum(1 for r in class_results if r['correct'])
                    class_total = len(class_results)
                    class_accuracy = class_correct / class_total
                    f.write(f"{class_name} Accuracy: {class_accuracy:.4f} ({class_correct}/{class_total})\n")
            f.write(f"\nClassification Report:\n{report_text}")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=self.class_names)
        self.plot_confusion_matrix(cm, self.class_names, output_dir=output_dir)
        
        # Generate confidence distribution plot
        self.plot_confidence_distribution(all_results, output_dir=output_dir)
        
        # Generate ROC curve
        roc_auc_scores = self.plot_roc_curve(all_results, output_dir=output_dir)
        
        # Generate GradCAM heatmap grid
        self.generate_gradcam_grid(data_folders_dict, output_dir=output_dir)

        return all_results
    
    def plot_confusion_matrix(self, cm, class_names, filename='hectnet_confusion_matrix.png', output_dir='.'):
        """Plot and save confusion matrix"""
        filepath = os.path.join(output_dir, filename)
        plt.figure(figsize=(max(6, len(class_names) + 1), max(4, int(len(class_names)*0.8) + 1)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('CVnets Model Confusion Matrix')
        plt.tight_layout()
        plt.savefig(filepath)
        print(f"Confusion matrix saved as '{filepath}'")
        plt.close()
    
    def plot_confidence_distribution(self, results, filename='hectnet_confidence_distribution.png', output_dir='.'):
        """Plot confidence distribution"""
        filepath = os.path.join(output_dir, filename)
        correct_confidences = [r['confidence'] for r in results if r['correct']]
        incorrect_confidences = [r['confidence'] for r in results if not r['correct']]

        plt.figure(figsize=(10, 6))
        plt.hist(correct_confidences, bins=20, alpha=0.7, label='Correct Predictions', color='green')
        plt.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect Predictions', color='red')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('hectnet Model Confidence Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(filepath)
        print(f"Confidence distribution saved as '{filepath}'")
        plt.close()

    def plot_roc_curve(self, results, filename='hectnet_roc_curve.png', output_dir='.'):
        """Plot ROC curve for multi-class classification"""
        filepath = os.path.join(output_dir, filename)
        
        # Extract true labels and predicted probabilities
        y_true = [r['expected_class'] for r in results]
        y_scores = np.array([r['probabilities'] for r in results])
        
        # Convert class names to indices
        class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        y_true_idx = [class_to_idx[class_name] for class_name in y_true]
        
        # Binarize the output for multi-class ROC
        y_true_bin = label_binarize(y_true_idx, classes=range(len(self.class_names)))
        n_classes = len(self.class_names)
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        
        # Plot micro-average ROC curve
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
                 color='deeppink', linestyle=':', linewidth=4)
        
        # Plot ROC curve for each class
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of {self.class_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('hectnet Model ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved as '{filepath}'")
        plt.close()
        
        return roc_auc

    def generate_gradcam_grid(self, data_folders_dict, filename='hectnet_gradcam_grid.png', output_dir='.', num_samples_per_class=2):
        """Generate GradCAM heatmap grid visualization"""
        filepath = os.path.join(output_dir, filename)
        
        # Find a suitable target layer (last convolutional layer before classifier)
        target_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)) and 'classifier' not in name:
                target_layer = name
        
        if target_layer is None:
            # Fallback: try to find any convolutional layer
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    target_layer = name
        
        if target_layer is None:
            print("Warning: Could not find suitable target layer for GradCAM")
            return
        
        print(f"Using target layer for GradCAM: {target_layer}")
        
        # Initialize GradCAM
        gradcam = GradCAM(self.model, target_layer)
        
        # Collect sample images from each class
        all_samples = []
        
        for class_name, folder_path in data_folders_dict.items():
            if class_name not in self.class_names:
                continue
                
            image_files = [f for f in os.listdir(folder_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Randomly sample images
            selected_files = random.sample(image_files, min(num_samples_per_class, len(image_files)))
            
            for img_file in selected_files:
                img_path = os.path.join(folder_path, img_file)
                all_samples.append((img_path, class_name))
        
        # Create grid layout (5x4 grid: 5 classes, 4 columns: original, heatmap, original, heatmap)
        fig, axes = plt.subplots(5, 4, figsize=(16, 20))
        fig.suptitle('GradCAM Heatmap Analysis - Model Attention Visualization', fontsize=16, fontweight='bold')
        
        sample_idx = 0
        for class_idx, class_name in enumerate(self.class_names):
            class_samples = [(path, cls) for path, cls in all_samples if cls == class_name]
            
            for sample_num in range(min(2, len(class_samples))):
                if sample_num >= len(class_samples):
                    break
                    
                img_path, _ = class_samples[sample_num]
                
                try:
                    # Load and preprocess image
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    
                    # Generate prediction
                    with torch.no_grad():
                        output = self.model(image_tensor)
                        predicted_class_idx = output.argmax(dim=1).item()
                        confidence = F.softmax(output, dim=1).max().item()
                    
                    # Generate GradCAM
                    cam = gradcam.generate_cam(image_tensor, predicted_class_idx)
                    
                    # Resize CAM to match input image size
                    cam_resized = cv2.resize(cam, (256, 256))
                    
                    # Convert image to numpy for visualization
                    img_np = np.array(image.resize((256, 256)))
                    
                    # Create heatmap overlay
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    
                    # Blend original image with heatmap
                    overlay = 0.6 * img_np + 0.4 * heatmap
                    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                    
                    # Plot original image
                    col_idx = sample_num * 2
                    axes[class_idx, col_idx].imshow(img_np)
                    axes[class_idx, col_idx].set_title(f'{class_name}\nOriginal\nConf: {confidence:.3f}', fontsize=10)
                    axes[class_idx, col_idx].axis('off')
                    
                    # Add border color based on prediction correctness
                    predicted_class = self.class_names[predicted_class_idx]
                    border_color = 'green' if predicted_class == class_name else 'red'
                    rect = Rectangle((0, 0), 255, 255, linewidth=3, edgecolor=border_color, facecolor='none')
                    axes[class_idx, col_idx].add_patch(rect)
                    
                    # Plot heatmap overlay
                    axes[class_idx, col_idx + 1].imshow(overlay)
                    axes[class_idx, col_idx + 1].set_title(f'GradCAM Heatmap\nPred: {predicted_class}', fontsize=10)
                    axes[class_idx, col_idx + 1].axis('off')
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    # Fill with blank if error
                    axes[class_idx, sample_num * 2].axis('off')
                    axes[class_idx, sample_num * 2 + 1].axis('off')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='green', lw=3, label='Correct Prediction'),
            plt.Line2D([0], [0], color='red', lw=3, label='Incorrect Prediction'),
            plt.Line2D([0], [0], color='blue', lw=0, label='Heatmap: Red=High Attention, Blue=Low Attention')
        ]
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"GradCAM grid saved as '{filepath}'")
        plt.close()
        
        # Clean up
        gradcam.remove_hooks()
        
        return filepath

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        # Find target layer and register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_backward_hook(backward_hook))
                break
    
    def generate_cam(self, input_tensor, class_idx=None):
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[:, class_idx]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3])
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='CVnets Model Evaluation')
    
    parser.add_argument('--model_path', type=str, 
                        default='C:/Users/User/OneDrive/Desktop/Capstone Project/CODE/Snackly App/HECTNet/hectnet_results/hectnet_multiscale_width050_100epochs/checkpoint_best.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--class_folders', nargs='+', 
                        default=[
                            'fried_rice:C:/Users/User/OneDrive/Desktop/DATASETS/malaysian_food_processed/test/fried_rice',
                            'kaya_toast:C:/Users/User/OneDrive/Desktop/DATASETS/malaysian_food_processed/test/kaya_toast',
                            'nasi_lemak:C:/Users/User/OneDrive/Desktop/DATASETS/malaysian_food_processed/test/nasi_lemak',
                            'roti_canai:C:/Users/User/OneDrive/Desktop/DATASETS/malaysian_food_processed/test/roti_canai',
                            'satay:C:/Users/User/OneDrive/Desktop/DATASETS/malaysian_food_processed/test/satay'
                        ],
                        help='Class folders in format class_name:folder_path')
    
    args = parser.parse_args()
    
    # Parse class folders
    data_folders_dict = {}
    for class_folder in args.class_folders:
        parts = class_folder.split(':', 1)
        if len(parts) == 2:
            class_name, folder_path = parts
            data_folders_dict[class_name] = folder_path
        else:
            print(f"Warning: Skipping malformed class_folder entry: {class_folder}")
            
    if not data_folders_dict:
        print("Error: No valid class folders provided. Exiting.")
        return

    # Run evaluation
    evaluator = CVnetsEvaluator(args.model_path, data_folders_dict)
    evaluator.run_evaluation(data_folders_dict)

    output_dir = "C:\\Users\\User\\OneDrive\\Desktop\\Capstone Project\\CODE\\Snackly App\\HECTNet\\hectnet_performance"

    print("\nEvaluation completed!")
    print("Results saved to:")
    print(f"- {os.path.join(output_dir, 'hectnet_classification_report.txt')}")
    print(f"- {os.path.join(output_dir, 'hectnet_confusion_matrix.png')}")
    print(f"- {os.path.join(output_dir, 'hectnet_confidence_distribution.png')}")
    print(f"- {os.path.join(output_dir, 'hectnet_roc_curve.png')}")
    print(f"- {os.path.join(output_dir, 'hectnet_gradcam_grid.png')}")

if __name__ == '__main__':
    main()