"""
Model Predictor - Handles model loading and inference
Uses PyTorch weights directly (Python version agnostic)
"""

import json
import io
from pathlib import Path
import logging
from PIL import Image
from PIL import UnidentifiedImageError
import torch
import torch.nn as nn
from torchvision import transforms
from fastai.vision.all import *
from timm import create_model

logger = logging.getLogger(__name__)


class DiseasePredictor:
    """Loads and uses the trained ConvNeXt model for predictions"""

    def __init__(self):
        """Initialize the predictor and load model"""

        self.model_dir = Path(__file__).parent.parent / "model" / "models"

        # Map between class indices and readable names
        self.class_mapping = {
            "Cassava___bacterial_blight": "Bacterial Blight",
            "Cassava___brown_streak_disease": "Brown Streak Disease",
            "Cassava___green_mottle": "Green Mottle",
            "Cassava___healthy": "Healthy",
            "Cassava___mosaic_disease": "Mosaic Disease"
        }

        self.classes = list(self.class_mapping.keys())

        # Disease information for recommendations
        self.disease_info = {
            "Cassava___bacterial_blight": {
                "severity": "High",
                "description": "Angular leaf spots, yellow halos",
                "action": "Use disease-free cuttings, copper bactericides"
            },
            "Cassava___brown_streak_disease": {
                "severity": "Critical",
                "description": "Yellow streaks, necrotic root lesions",
                "action": "Destroy infected plants immediately"
            },
            "Cassava___green_mottle": {
                "severity": "Moderate",
                "description": "Mosaic and mottling on leaves",
                "action": "Control whitefly vectors, use resistant varieties"
            },
            "Cassava___healthy": {
                "severity": "None",
                "description": "No visible disease signs",
                "action": "Continue monitoring and good practices"
            },
            "Cassava___mosaic_disease": {
                "severity": "High",
                "description": "Mosaic patterns and distortion",
                "action": "Use resistant varieties, control whiteflies"
            }
        }

        try:
            self.learn = self._load_model()
            self.classes = list(self.learn.dls.vocab)
            logger.info("✓ Model loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            raise

    def _load_model(self):
        """Load ConvNeXt model from exported learner or PyTorch weights"""
        
        # **PRIORITY: Load from model.pkl (has complete trained model with head)**
        pkl_files = [
            self.model_dir / "model.pkl",
            self.model_dir / "phase2_checkpoint.pkl",
            self.model_dir / "phase1_checkpoint.pkl"
        ]
        
        for pkl_path in pkl_files:
            if pkl_path.exists():
                try:
                    logger.info(f"Loading {pkl_path.name}...")
                    learn = load_learner(pkl_path)
                    logger.info(f"✓ Model loaded successfully from {pkl_path.name}")
                    return learn
                except Exception as e:
                    logger.warning(f"Failed to load {pkl_path.name}: {e}")
                    continue
        
        # Fallback: Try weights files (version-agnostic PyTorch format)
        weights_files = [
            self.model_dir / "phase2_best.pth",
            self.model_dir / "weights.pth"
        ]
        
        for weights_path in weights_files:
            if weights_path.exists():
                try:
                    logger.info(f"Loading weights from {weights_path.name}...")
                    
                    # Create ConvNeXt Small model with ImageNet-22K pretraining
                    # (as specified in metrics.json: "model_arch": "convnext_small_in22k")
                    model = create_model('convnext_small_in22k', pretrained=False, num_classes=5)
                    
                    # Load weights
                    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
                    
                    # Filter and clean the state dict
                    # Keep only '0.model.' entries (the main model, not EMA weights)
                    clean_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith('0.model.'):
                            # Strip the '0.model.' prefix
                            new_key = k.replace('0.model.', '', 1)
                            clean_state_dict[new_key] = v
                    
                    logger.info(f"Filtered state dict: {len(state_dict)} → {len(clean_state_dict)} keys")
                    state_dict = clean_state_dict
                    
                    # Load into model
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    
                    # Initialize missing head weights with Kaiming normal (better than random)
                    if missing:
                        logger.info(f"Initializing {len(missing)} missing keys with Kaiming normal...")
                        for name, module in model.named_modules():
                            if isinstance(module, nn.Linear):
                                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                                if module.bias is not None:
                                    nn.init.constant_(module.bias, 0)
                    
                    model.eval()
                    
                    logger.info(f"✓ Model loaded successfully from {weights_path.name}")
                    
                    # Wrap in FastAI-like object for compatibility
                    class SimplePredictor:
                        def __init__(self, model, device='cpu'):
                            self.model = model.to(device)
                            self.device = device
                            self.dls = type('obj', (object,), {'vocab': ['Cassava___bacterial_blight', 'Cassava___brown_streak_disease', 'Cassava___green_mottle', 'Cassava___healthy', 'Cassava___mosaic_disease']})()
                        
                        def predict(self, img):
                            """Return (class_name, class_idx, probabilities)"""
                            import numpy as np
                            from torchvision import transforms
                            
                            with torch.no_grad():
                                # Handle PILImage from FastAI
                                if hasattr(img, 'data'):
                                    pil_img = img.data
                                elif isinstance(img, Image.Image):
                                    pil_img = img
                                else:
                                    pil_img = img
                                
                                # Convert to PIL if needed
                                if not isinstance(pil_img, Image.Image):
                                    pil_img = Image.fromarray(np.array(pil_img).astype('uint8'))
                                
                                # Convert to tensor using standard transforms
                                transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])
                                ])
                                
                                x = transform(pil_img).unsqueeze(0).to(self.device)
                                logits = self.model(x)
                                probs = torch.softmax(logits, dim=1)
                                pred_idx = probs.argmax(dim=1).item()
                                pred_class = self.dls.vocab[pred_idx]
                                
                                return pred_class, pred_idx, probs[0].cpu()
                    
                    return SimplePredictor(model)
                
                except Exception as e:
                    logger.warning(f"Failed to load {weights_path.name}: {e}")
                    continue
        
        # No valid model files found
        raise FileNotFoundError(
            f"No valid model files found at {self.model_dir}\n"
            f"Tried: model.pkl, phase2_checkpoint.pkl, phase1_checkpoint.pkl (FastAI)\n"
            f"And: phase2_best.pth, weights.pth (PyTorch)\n"
            f"Please ensure model files exist via: git lfs pull or download from Colab"
        )

    def predict(self, image_stream):
        """
        Make prediction on image

        Args:
            image_stream: BytesIO object or PIL Image

        Returns:
            dict: Prediction results with disease, confidence, and probabilities
        """

        try:
            if isinstance(image_stream, (bytes, bytearray)):
                stream = io.BytesIO(image_stream)
            else:
                stream = image_stream
            try:
                stream.seek(0)
            except Exception:
                pass
            try:
                image = Image.open(stream)
                image.load()
            except UnidentifiedImageError as e:
                logger.error("Invalid image data")
                raise ValueError(
                    "Unsupported or corrupted image. Please upload a valid JPG or PNG.") from e
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img = PILImage.create(image)
            pred_class, pred_idx, probs = self.learn.predict(img)
            disease_folder = pred_class
            disease_name = self.class_mapping.get(
                disease_folder,
                disease_folder.replace("Cassava___", "").replace("_", " ")
            )
            confidence = float(probs[pred_idx])
            all_probs = {}
            for i, cls in enumerate(self.classes):
                readable_name = self.class_mapping.get(
                    cls,
                    cls.replace("Cassava___", "").replace("_", " ")
                )
                all_probs[readable_name] = float(probs[i])
            disease_details = self.disease_info.get(disease_folder, {})
            result = {
                "disease": disease_name,
                "disease_folder": disease_folder,
                "confidence": confidence,
                "confidence_pct": f"{confidence*100:.2f}%",
                "all_predictions": all_probs,
                "severity": disease_details.get("severity", "Unknown"),
                "description": disease_details.get("description", ""),
                "action": disease_details.get("action", "")
            }
            logger.info(f"Prediction: {disease_name} ({confidence:.2%})")
            return result
        except Exception as e:
            logger.exception("Prediction error")
            raise ValueError(
                f"Failed to process image: {type(e).__name__}: {e}")

    def predict_batch(self, images):
        """
        Make predictions on multiple images

        Args:
            images: List of image streams

        Returns:
            list: List of prediction results
        """
        results = []
        for img in images:
            try:
                result = self.predict(img)
                results.append({"success": True, "data": result})
            except Exception as e:
                results.append({"success": False, "error": str(e)})

        return results
