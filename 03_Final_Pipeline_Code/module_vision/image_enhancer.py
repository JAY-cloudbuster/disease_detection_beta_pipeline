import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

# --- Zero-DCE Architecture ---
class ZeroDCE(nn.Module):
    def __init__(self, num_iterations=8):
        super().__init__()
        self.num_iterations = num_iterations
        number_f = 32

        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 3 * num_iterations, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        r_maps = torch.split(x_r, 3, dim=1)
        enhanced = x
        for r in r_maps:
            enhanced = enhanced + r * enhanced * (1.0 - enhanced)
        return enhanced

# --- Original U-Net V3 Architecture ---
class UNetV3(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=32):
        super().__init__()
        def CBR(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        self.enc1 = CBR(in_ch, base_ch)
        self.enc2 = CBR(base_ch, base_ch*2)
        self.enc3 = CBR(base_ch*2, base_ch*4)
        self.pool = nn.MaxPool2d(2)
        
        self.up2  = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = CBR(base_ch*4, base_ch*2)
        self.up1  = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = CBR(base_ch*2, base_ch)
        self.final = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        
        u2 = self.up2(x3)
        if u2.shape != x2.shape:
            u2 = F.interpolate(u2, size=x2.shape[2:], mode='bilinear', align_corners=False)
        c2 = torch.cat([u2, x2], dim=1)
        d2 = self.dec2(c2)
        
        u1 = self.up1(d2)
        if u1.shape != x1.shape:
            u1 = F.interpolate(u1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        c1 = torch.cat([u1, x1], dim=1)
        d1 = self.dec1(c1)
        return torch.sigmoid(self.final(d1))

# --- Attention U-Net + CBAM Architecture ---
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (nn.Sequential(
                             nn.Conv2d(in_ch, out_ch, 1, bias=False),
                             nn.BatchNorm2d(out_ch))
                         if in_ch != out_ch else nn.Identity())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
    def forward(self, x):
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1))
        mx  = self.mlp(F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1))
        scale = torch.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))

class AttentionGate(nn.Module):
    def __init__(self, f_g, f_x, f_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, 1, bias=False),
            nn.BatchNorm2d(f_int))
        self.W_x = nn.Sequential(
            nn.Conv2d(f_x, f_int, 1, bias=False),
            nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
    def forward(self, g, x):
        g1  = self.W_g(g)
        x1  = self.W_x(x)
        psi = self.psi(F.relu(g1 + x1, inplace=True))
        return x * psi

class AttentionUNetCBAM(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=32):
        super().__init__()
        b = base_ch
        self.enc1 = ResBlock(in_ch,  b)
        self.enc2 = ResBlock(b,      b*2)
        self.enc3 = ResBlock(b*2,    b*4)
        self.enc4 = ResBlock(b*4,    b*8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ResBlock(b*8, b*16)
        self.cbam       = CBAM(b*16)

        self.up4 = nn.ConvTranspose2d(b*16, b*8,  2, stride=2)
        self.up3 = nn.ConvTranspose2d(b*8,  b*4,  2, stride=2)
        self.up2 = nn.ConvTranspose2d(b*4,  b*2,  2, stride=2)
        self.up1 = nn.ConvTranspose2d(b*2,  b,    2, stride=2)

        self.ag4 = AttentionGate(f_g=b*8,  f_x=b*8,  f_int=b*4)
        self.ag3 = AttentionGate(f_g=b*4,  f_x=b*4,  f_int=b*2)
        self.ag2 = AttentionGate(f_g=b*2,  f_x=b*2,  f_int=b)
        self.ag1 = AttentionGate(f_g=b,    f_x=b,    f_int=b//2)

        self.dec4 = ResBlock(b*16, b*8)
        self.dec3 = ResBlock(b*8,  b*4)
        self.dec2 = ResBlock(b*4,  b*2)
        self.dec1 = ResBlock(b*2,  b)

        self.head = nn.Sequential(
            nn.Conv2d(b, out_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b  = self.cbam(self.bottleneck(self.pool(e4)))

        u4 = F.interpolate(self.up4(b), size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([u4, self.ag4(u4, e4)], dim=1))

        u3 = F.interpolate(self.up3(d4), size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, self.ag3(u3, e3)], dim=1))

        u2 = F.interpolate(self.up2(d3), size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, self.ag2(u2, e2)], dim=1))

        u1 = F.interpolate(self.up1(d2), size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, self.ag1(u1, e1)], dim=1))

        return self.head(d1)

# --- Base Image Enhancer Pipeline ---
def apply_clahe(img_np, clip_limit=2.0, tile_grid=(8, 8)):
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    enhanced = clahe.apply(img_gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

class ImageEnhancer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self._load_models()
        self.transform = T.Compose([
            T.Resize((256, 256)), # Models expect 256x256
            T.ToTensor()
        ])

    def _load_models(self):
        # Initialize architectures
        self.models["Zero-DCE"] = ZeroDCE().to(self.device)
        self.models["U-Net V3"] = UNetV3().to(self.device)
        self.models["Attention U-Net + CBAM"] = AttentionUNetCBAM().to(self.device)
        self.models["Attention U-Net + CLAHE Pipeline"] = AttentionUNetCBAM().to(self.device)
        
        # Load weights if available, otherwise just use untrained for demonstration purposes
        weights = {
            "Zero-DCE": "zerodce_xray.pt",
            "U-Net V3": "unet_v3.pth",
            "Attention U-Net + CBAM": "attention_unet.pth",
            "Attention U-Net + CLAHE Pipeline": "attention_unet.pth"
        }
        for name, path in weights.items():
            if os.path.exists(path):
                try:
                    self.models[name].load_state_dict(torch.load(path, map_location=self.device))
                except Exception as e:
                    print(f"Warning: Failed to load {path} - {e}")
            self.models[name].eval()

    def enhance(self, img_path, model_choice="Attention U-Net + CLAHE Pipeline"):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_shape = img_rgb.shape[:2]

        if model_choice == "Original":
            return img_rgb, "Original"
        elif model_choice == "CLAHE":
            return apply_clahe(img_rgb), "CLAHE"
        elif model_choice == "UniMIE":
            # Mock UniMIE fallback as per original code
            return apply_clahe(img_rgb), "CLAHE (UniMIE Fallback)"
        elif model_choice in self.models:
            # Process with PyTorch models
            input_tensor = self.transform(Image.fromarray(img_rgb)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out_tensor = self.models[model_choice](input_tensor)
            
            # Post-process tensor back to numpy image
            out_img = out_tensor.squeeze(0).cpu().numpy()
            out_img = np.transpose(out_img, (1, 2, 0))
            out_img = (out_img * 255.0).clip(0, 255).astype(np.uint8)
            
            # Resize back to original
            out_img = cv2.resize(out_img, (original_shape[1], original_shape[0]))

            # Special case for the pipeline which chains U-Net + CLAHE
            if model_choice == "Attention U-Net + CLAHE Pipeline":
                out_img = apply_clahe(out_img, clip_limit=2.0)
                
            return out_img, model_choice
            
        return img_rgb, "Original (Unknown Model)"
