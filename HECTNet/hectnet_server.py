import torch
import torch.nn as nn
import os
import sys
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import io
import uvicorn
from typing import Dict, List
import base64

# Add the CVnets directory to Python path correctly
cvnets_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cvnets_dir)

# Import CVnets modules
import models.classification
from models.classification import ehfr_net
from models.classification import CLS_MODEL_REGISTRY
from models import get_model
from options.opts import get_eval_arguments
from utils import logger
from common import DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT

# Initialize FastAPI app
app = FastAPI(title="HECTNet Classification API", version="1.0.0")

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module level - no indentation
# Add after the BASE_NUTRITION_DATA dictionary (around line 184)
BASE_NUTRITION_DATA = {
    "nasi_lemak": {"calories": 644, "carbs": 86, "protein": 17, "fat": 26},
    "roti_canai": {"calories": 300, "carbs": 42, "protein": 8, "fat": 12},
    "satay": {"calories": 400, "carbs": 20, "protein": 30, "fat": 20},
    "fried_rice": {"calories": 500, "carbs": 70, "protein": 15, "fat": 15},
    "kaya_toast": {"calories": 350, "carbs": 45, "protein": 6, "fat": 15},
}

def get_nutrition_info(predicted_class: str):
    nutrition_key = predicted_class.lower().replace(' ', '_')
    return BASE_NUTRITION_DATA.get(nutrition_key, {
        "calories": 0, "carbs": 0, "protein": 0, "fat": 0
    })

class HECTNetPredictor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_names = ['fried_rice', 'kaya_toast', 'nasi_lemak', 'roti_canai', 'satay']
        self.model_path = 'C:/Users/User/OneDrive/Desktop/Capstone Project/CODE/Snackly App/HECTNet/hectnet_results/hectnet_multiscale_width050_100epochs/checkpoint_best.pt'
        
        print(f"Initializing HECTNet model on {self.device}...")
        self._load_model()
        self._setup_transforms()
        print("Model loaded successfully!")
    
    def _load_model(self):
        """Load the HECTNet model with exact training configuration"""
        opts = get_eval_arguments()
        
        # Apply the EXACT same configuration as training
        # Common settings
        setattr(opts, "common.auto_resume", True)
        setattr(opts, "common.mixed_precision", True)
        setattr(opts, "common.channels_last", False)
        setattr(opts, "common.tensorboard_logging", False)
        setattr(opts, "common.grad_clip", 10.0)
        setattr(opts, "common.accum_freq", 1)
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
        
        # Model settings - EXACT match with training configuration
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
        
        # Scheduler settings
        setattr(opts, "scheduler.name", "cosine")
        setattr(opts, "scheduler.max_epochs", 100)
        setattr(opts, "scheduler.is_iteration_based", False)
        setattr(opts, "scheduler.warmup_iterations", 1000)
        setattr(opts, "scheduler.warmup_init_lr", 1e-6)
        setattr(opts, "scheduler.cosine.max_lr", 0.0005)
        setattr(opts, "scheduler.cosine.min_lr", 0.00005)
        
        # Stats settings
        setattr(opts, "stats.val", ["loss", "top1", "top5"])
        setattr(opts, "stats.train", ["loss", "top1"])
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
        else:
            setattr(opts, "dev.device", torch.device("cpu"))
            setattr(opts, "dev.num_gpus", 0)
        
        # Initialize model
        self.model = get_model(opts)
        
        # Load trained weights
        try:
            model_state_dict = torch.load(self.model_path, map_location=self.device)
            if 'model' in model_state_dict:
                self.model.load_state_dict(model_state_dict['model'], strict=False)
            elif 'state_dict' in model_state_dict:
                self.model.load_state_dict(model_state_dict['state_dict'], strict=False)
            else:
                self.model.load_state_dict(model_state_dict, strict=False)
            print(f"Successfully loaded model weights from {self.model_path}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            raise
        
        self.model.to(self.device)
        self.model.eval()
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transform = transforms.Compose([
            transforms.Resize((DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)),
            transforms.ToTensor(),
        ])
    
    def predict(self, image: Image.Image) -> Dict:
        """Predict class for a single image"""
        try:
            # Preprocess image
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Get prediction results
            predicted_idx = predicted.item()
            confidence_score = confidence.item()
            predicted_class = self.class_names[predicted_idx]
            
            # Get all class probabilities
            all_probabilities = probabilities.cpu().numpy()[0]
            class_probabilities = {
                class_name: float(prob) 
                for class_name, prob in zip(self.class_names, all_probabilities)
            }
            
            # Add nutrition info
            nutrition_info = get_nutrition_info(predicted_class)
            
            return {
                "label": predicted_class,  # Changed from predicted_class
                "confidence": confidence_score,
                "nutrition_info": nutrition_info,  # Added
                "class_probabilities": class_probabilities
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Initialize the predictor
predictor = HECTNetPredictor()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "HECTNet Classification API",
        "version": "1.0.0",
        "model": "HECTNet Multiscale",
        "classes": predictor.class_names,
        "endpoints": {
            "/predict": "POST - Upload image for classification",
            "/health": "GET - Health check",
            "/classes": "GET - Get available classes"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": predictor.device,
        "model_loaded": True
    }

@app.get("/classes")
async def get_classes():
    """Get available classification classes"""
    return {
        "classes": predictor.class_names,
        "num_classes": len(predictor.class_names)
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Predict class for uploaded image"""
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Make prediction
        result = predictor.predict(image)
        
        return {
            "filename": file.filename,
            "prediction": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict_base64")
async def predict_base64_image(data: Dict):
    """Predict class for base64 encoded image"""
    try:
        # Decode base64 image
        if "image" not in data:
            raise HTTPException(status_code=400, detail="Missing 'image' field in request")
        
        image_data = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Make prediction
        result = predictor.predict(image)
        
        return {
            "prediction": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing base64 image: {str(e)}")

if __name__ == "__main__":
    print("Starting HECTNet Classification Server...")
    print(f"Model device: {predictor.device}")
    print(f"Available classes: {predictor.class_names}")
    print("\nServer will be available at:")
    print("- http://localhost:8000 (API documentation)")
    print("- http://localhost:8000/docs (Interactive API docs)")
    print("- http://localhost:8000/predict (POST endpoint for image classification)")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)