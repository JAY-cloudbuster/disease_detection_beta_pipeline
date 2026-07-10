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
    def __init__(self, model_path="best_hybrid_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HybridCNNTransformer(num_classes=len(DISEASES)).to(self.device)
        self.model.eval()
        
        # Load weights if available
        import os
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded classifier weights from {model_path}")
        else:
            print(f"Warning: {model_path} not found. Running with untrained weights for demonstration.")
            
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize ScoreCAM
        self.target_layer = self.model.cnn_backbone[-1]
        self.cam = ScoreCAM(model=self.model, target_layers=[self.target_layer])

    def predict_and_explain(self, img_rgb_np):
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

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.sigmoid(outputs)[0].cpu().numpy()

        results = {}
        heatmaps = {}

        for i, disease in enumerate(DISEASES):
            prob = float(probs[i])
            results[disease] = prob
            
            # Generate CAM
            targets = [ClassifierOutputTarget(i)]
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)[0]
            visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
            heatmaps[disease] = visualization

        return results, heatmaps
