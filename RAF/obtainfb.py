import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import os
import random

NUM_PROXIES = 512  # Target number of background proxies
CHANNELS = 2048  # Output channels of ResNet50 layer4
SPATIAL_SIZE = 7  # Spatial size of ResNet50 layer4 output
BATCH_SIZE = 32  # Batch size for feature extraction

# Dataset paths (Modify these paths according to your local environment)
INDOOR_DATASET_PATH = r"/root\datasets\mit_indoor"
OUTDOOR_DATASET_PATH = r"/root\datasets\places365_outdoor"

# Output path for the extracted features
OUTPUT_FEATURE_PATH = r"/root\background_proxies_512.pth"



class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load ImageNet pre-trained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Keep layers up to layer4 as the feature extractor
        self.feature_extractor = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Output shape: [Batch, 2048, 7, 7]
        return self.feature_extractor(x)



class Model(nn.Module):
    def __init__(self, num_bg_proxies, channels, spatial_size):
        super().__init__()

        # Initialize with zeros as placeholders (will be overwritten by real features)
        self.background_proxies = nn.Parameter(
            torch.zeros(num_bg_proxies, channels, spatial_size, spatial_size)
        )

    def init_background_proxies(self, feature_path):
        """
        Dedicated method to load indoor/outdoor dataset features.
        """
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file not found: {feature_path}. Please check the path.")

        print(f"Loading background proxies from {feature_path}...")
        loaded_features = torch.load(feature_path, map_location='cpu')
        if loaded_features.shape != self.background_proxies.shape:
            raise ValueError(
                f"Shape mismatch! Loaded shape: {loaded_features.shape}, "
                f"Expected shape: {self.background_proxies.shape}."
            )
        self.background_proxies.data.copy_(loaded_features)




def extract_and_save_features():
    """Step 1: Extract and save features from datasets."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    indoor_dataset = datasets.ImageFolder(root=INDOOR_DATASET_PATH, transform=transform)

    outdoor_dataset = datasets.Places365(
        root=OUTDOOR_DATASET_PATH,
        split='val',  # Use validation set for efficiency
        small=True,  # Use 256x256 resized images
        download=True,  # Auto-download if not present
        transform=transform
    )

    num_indoor = NUM_PROXIES // 2
    num_outdoor = NUM_PROXIES - num_indoor

    print(f"🎲 Randomly sampling {num_indoor} indoor and {num_outdoor} outdoor images...")
    indoor_indices = random.sample(range(len(indoor_dataset)), num_indoor)
    outdoor_indices = random.sample(range(len(outdoor_dataset)), num_outdoor)

    # Merge subsets
    indoor_subset = Subset(indoor_dataset, indoor_indices)
    outdoor_subset = Subset(outdoor_dataset, outdoor_indices)
    combined_subset = ConcatDataset([indoor_subset, outdoor_subset])
    dataloader = DataLoader(combined_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Extract features
    extractor = ResNet50FeatureExtractor().to(device).eval()
    all_features = []

    print("⚙️ Extracting features...")
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            features = extractor(images)  # [B, 2048, 7, 7]
            all_features.append(features.cpu())

    # Concatenate and validate dimensions
    final_features = torch.cat(all_features, dim=0)  # [512, 2048, 7, 7]
    assert final_features.shape == (NUM_PROXIES, CHANNELS, SPATIAL_SIZE, SPATIAL_SIZE), \
        f"Dimension error! Current: {final_features.shape}, Expected: {(NUM_PROXIES, CHANNELS, SPATIAL_SIZE, SPATIAL_SIZE)}"

    # Save as .pth file
    torch.save(final_features, OUTPUT_FEATURE_PATH)
    print(f"Feature extraction complete! Saved to: {OUTPUT_FEATURE_PATH}")
    print(f"Final Tensor Shape: {final_features.shape}\n")


def load_model_with_proxies():
    model = Model(
        num_bg_proxies=NUM_PROXIES,
        channels=CHANNELS,
        spatial_size=SPATIAL_SIZE
    )

    model.init_background_proxies(OUTPUT_FEATURE_PATH)

    print(f"Model parameter shape: {model.background_proxies.shape}")
    print(
        f"Parameter mean: {model.background_proxies.mean().item():.4f}, std: {model.background_proxies.std().item():.4f}")
    return model



if __name__ == "__main__":

    if not os.path.exists(OUTPUT_FEATURE_PATH):
        extract_and_save_features()
    else:
        print(f"Existing feature file found: {OUTPUT_FEATURE_PATH}. Skipping extraction.")
    model = load_model_with_proxies()
