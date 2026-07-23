import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

DISEASES = ["Cardiomegaly", "Edema", "Pleural Effusion"]

class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes=3, num_layers=2, nhead=8, dim_feedforward=2048):
        super().__init__()
        # CNN backbone: reuse DenseNet121's conv layers
        densenet = models.densenet121(weights="DEFAULT")
        self.cnn_backbone = densenet.features
        self.feature_dim = 1024

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.feature_dim))

        # Positional embeddings (49 patches + 1 cls = 50)
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, self.feature_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        feat_map = self.cnn_backbone(x)
        feat_map = F.relu(feat_map)

        B, C, H, W = feat_map.shape
        tokens = feat_map.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + self.pos_embed

        encoded = self.transformer(tokens)
        cls_out = encoded[:, 0]
        return self.classifier(cls_out)

class DiseaseClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Hybrid CNN Transformer
        self.model_hybrid = HybridCNNTransformer(num_classes=len(DISEASES)).to(self.device)
        if os.path.exists("best_hybrid_model.pth"):
            try:
                self.model_hybrid.load_state_dict(torch.load("best_hybrid_model.pth", map_location=self.device))
                print("Loaded Hybrid CNN-Transformer weights.")
            except Exception as e:
                print(f"Warning: Failed to load best_hybrid_model.pth - {e}")
        self.model_hybrid.eval()

        # Load DenseNet121 Baseline
        self.model_densenet = models.densenet121(weights="DEFAULT")
        num_features = self.model_densenet.classifier.in_features
        self.model_densenet.classifier = nn.Linear(num_features, len(DISEASES))
        self.model_densenet = self.model_densenet.to(self.device)
        if os.path.exists("best_chexpert_model.pth"):
            try:
                self.model_densenet.load_state_dict(torch.load("best_chexpert_model.pth", map_location=self.device))
                print("Loaded DenseNet121 weights.")
            except Exception as e:
                print(f"Warning: Failed to load best_chexpert_model.pth - {e}")
        self.model_densenet.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_and_explain(self, img_rgb_np, model_choice="Hybrid CNN-Transformer"):
        """
        Takes an RGB numpy array, predicts diseases, and generates heatmaps.
        Returns:
            probabilities: dict of {disease_name: probability}
            heatmaps: dict of {disease_name: heatmap_image_array}
        """
        pil_img = Image.fromarray(img_rgb_np)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # Prepare image for visualization
        rgb_image = cv2.resize(img_rgb_np, (224, 224))
        rgb_image = np.float32(rgb_image) / 255.0

        # Select model and target layer based on user dropdown choice
        if model_choice == "Hybrid CNN-Transformer":
            model_to_use = self.model_hybrid
            target_layer = model_to_use.cnn_backbone[-1]
        else:
            model_to_use = self.model_densenet
            target_layer = model_to_use.features[-1]

        with torch.no_grad():
            outputs = model_to_use(input_tensor)
            probs = torch.sigmoid(outputs)[0].cpu().numpy()

        results = {}
        heatmaps = {}

        # Initialize ScoreCAM for the selected model
        cam = ScoreCAM(model=model_to_use, target_layers=[target_layer])

        for i, disease in enumerate(DISEASES):
            prob = float(probs[i])
            results[disease] = prob
            
            # Generate CAM
            targets = [ClassifierOutputTarget(i)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
            visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
            heatmaps[disease] = visualization

        return results, heatmaps
