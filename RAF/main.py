import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision import models, transforms
import clip
import pickle
import os
from PIL import Image

class RegionAwareFilterProcessor:
    def __init__(self, 
                 voc_root, 
                 pkl_path, 
                 clip_model_name="ViT-B/32", 
                 device='cuda'):
        """
        Initialize the RAF Processor for VOC2007.
        
        Args:
            voc_root: Path to VOCdevkit/VOC2007
            pkl_path: Path to the region proposals .pkl file
            clip_model_name: CLIP model variant
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.voc_root = voc_root
        
        # 1. Load Data from PKL
        print(f"Loading proposals from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.boxes = data['boxes']      # List of arrays [N, 4]
        self.scores = data['scores']    # List of arrays [N]
        self.indexes = data['indexes']  # List of ints (Image IDs)
        
        # 2. Load Models
        # Visual Encoder (ResNet-50) - Extracts features from RoI
        self.resnet = models.resnet50(pretrained=True).to(device)
        self.resnet.eval()
        # Remove FC layer, we only need features up to layer4
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2]) 
        
        # Text Encoder (CLIP) - Generates category embeddings
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.clip_model.eval()
        
        # Projection Layer: Align ResNet (2048) to CLIP (512) space
        # This is crucial for calculating similarity in Eq. (2)
        self.projection = nn.Linear(2048, 512).to(device)
        
        # RoIAlign Configuration
        # output_size=7x7 is standard for ResNet backbones
        self.roi_align = roi_align
        
        # VOC2007 Categories (Standard 20 classes)
        self.classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
            "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
            "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]
        
    def get_image_path(self, index_id):
        """
        Map VOC index (e.g., 2008000001) to file path.
        VOC filenames are usually zero-padded to 6 digits.
        """
        # Convert int to string and remove potential prefix if needed, 
        # but usually VOC IDs map directly to filename stem.
        # Example: 2008000001 -> 000001.jpg
        img_name = f"{str(index_id)[-6:]}.jpg"
        return os.path.join(self.voc_root, "JPEGImages", img_name)

    @torch.no_grad()
    def process_and_filter(self, batch_size=32, alpha=0.4):
        """
        Main execution loop.
        Iterates through images, extracts features, computes similarity, and filters.
        """
        results = []
        total_images = len(self.indexes)
        
        # Pre-compute text embeddings for all classes (Dynamic Prompts)
        # Using "a photo of a {class}" template as per paper
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.classes]).to(self.device)
        text_features = self.clip_model.encode_text(text_inputs) # Shape: [20, 512]
        text_features = F.normalize(text_features, dim=-1)
        
        print("Starting Region-Aware Filtering...")
        
        # Group boxes by image index for efficient processing
        # In real implementation, you'd iterate unique indexes
        for i in range(min(10, total_images)): # Limit to 10 images for demo
            img_idx = self.indexes[i]
            img_path = self.get_image_path(img_idx)
            
            if not os.path.exists(img_path):
                continue
                
            # 1. Load Image & Preprocess
            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            
            # Get boxes for this image (Assuming 1 image per entry in this simplified loop)
            # Note: Your PKL structure implies lists of length 10991. 
            # Usually one entry = one image's proposals.
            boxes = torch.tensor(self.boxes[i], dtype=torch.float32).to(self.device)
            
            # Prepare Image Tensor for ResNet
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)), # Resize for feature extraction consistency
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img_tensor = preprocess(image).unsqueeze(0).to(self.device)
            
            # 2. Extract Visual Features (ResNet Layer 4)
            # Output shape: [1, 2048, 7, 7]
            visual_features = self.feature_extractor(img_tensor)
            
            # 3. RoIAlign
            # boxes format for roi_align: [N, 5] where col 0 is batch index
            batch_indices = torch.zeros(boxes.size(0), 1, device=self.device)
            rois = torch.cat([batch_indices, boxes], dim=1)
            
            # Align to 7x7 spatial size
            # IMPORTANT: If image was resized to 224x224 for ResNet, boxes must be scaled too!
            scale_x = 224.0 / w
            scale_y = 224.0 / h
            scaled_boxes = boxes.clone()
            scaled_boxes[:, 0::2] *= scale_x
            scaled_boxes[:, 1::2] *= scale_y
            
            rois_scaled = torch.cat([batch_indices, scaled_boxes], dim=1)
            
            # Extract region features: [N, 2048, 7, 7] -> [N, 2048]
            region_feats = self.roi_align(visual_features, rois_scaled, output_size=(7, 7))
            region_feats = region_feats.mean(dim=[2, 3]) # Global Average Pooling
            
            # Project to CLIP space: [N, 512]
            region_feats_proj = self.projection(region_feats)
            region_feats_norm = F.normalize(region_feats_proj, dim=-1)
            
            # 4. Calculate Similarity (Eq. 2 logic)
            # Matrix multiplication: [N, 512] x [20, 512]^T = [N, 20]
            similarities = region_feats_norm @ text_features.T
            
            # Find max similarity for each region (Best matching class score)
            max_scores, _ = torch.max(similarities, dim=1)
            
            # 5. Filter based on threshold alpha
            keep_mask = max_scores > alpha
            kept_boxes = boxes[keep_mask].cpu().numpy()
            kept_scores = max_scores[keep_mask].cpu().numpy()
            
            print(f"Image {img_idx}: Original {len(boxes)} regions -> Kept {len(kept_boxes)} regions")
            
            results.append({
                'index': img_idx,
                'kept_boxes': kept_boxes,
                'confidence': kept_scores
            })
            
        return results

# ==========================================
# Usage Example
# ==========================================
if __name__ == "__main__":
    # CONFIGURATION
    VOC_ROOT = r"/root/datasets\VOCdevkit\VOC2007" 
    PKL_PATH = r"/root/datasets\voc_proposals.pkl" 
    
    processor = RegionAwareFilterProcessor(
        voc_root=VOC_ROOT,
        pkl_path=PKL_PATH,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    filtered_results = processor.process_and_filter(alpha=0.4)
