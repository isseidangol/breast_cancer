import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.ops import RoIAlign
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data Augmentation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class BreastCancerDataset(Dataset):
    def __init__(self, image_dir, mask_dir, annotations_file, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        label = row['label']
        image_path = os.path.join(self.image_dir, 'malignant' if label == 1 else 'benign', row['image'])
        mask_path = os.path.join(self.mask_dir, 'malignant' if label == 1 else 'benign', row['image'].replace('_image.jpg', '_mask.jpg'))

        # Load and process image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None

        # Extract ROI coordinates
        x_min, y_min, x_max, y_max = row[['x_min', 'y_min', 'x_max', 'y_max']]
        rois = torch.tensor([[0, x_min, y_min, x_max, y_max]], dtype=torch.float32)

        return image, rois, label

def plot_sample(dataset, idx):
    image, rois, label = dataset[idx]
    image = (image.permute(1, 2, 0).numpy() * 0.5) + 0.5  # Denormalize

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    # Draw the bounding box
    x_min, y_min, x_max, y_max = rois[0, 1:].numpy()
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    label_str = 'Malignant' if label == 1 else 'Benign'
    ax.set_title(f"Label: {label_str}")
    plt.axis('off')
    plt.show()

# Dataset and DataLoader
image_dir = 'mri/dataset/images/'
mask_dir = 'mri/dataset/masks/'
annotations_file = 'mri/dataset/annotations/roi_annotations.csv'
dataset = BreastCancerDataset(image_dir=image_dir, mask_dir=mask_dir, annotations_file=annotations_file, transform=transform)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Plot example sample
plot_sample(dataset, idx=2)

class FPNWithROI(nn.Module):
    def __init__(self, backbone, feature_channels):
        super(FPNWithROI, self).__init__()
        self.backbone = backbone
        self.fpn = nn.Conv2d(feature_channels, 256, kernel_size=1)
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1.0, sampling_ratio=2)
        self.reconstruction_head = nn.Sequential(
            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        self.dgfe = DifferenceMapGuidedEnhancement()
        self.classifier = nn.Linear(256 * 7 * 7, 2)

    def forward(self, x, rois):
        features = self.backbone(x)
        fpn_features = self.fpn(features)
        reconstructed_image = self.reconstruction_head(fpn_features)
        difference_map = torch.abs(reconstructed_image - x).mean(dim=1, keepdim=True)
        enhanced_features = self.dgfe(fpn_features, difference_map)

        roi_features = self.roi_align(enhanced_features, rois)
        roi_features = roi_features.view(roi_features.size(0), -1)
        output = self.classifier(roi_features)
        return output

class DifferenceMapGuidedEnhancement(nn.Module):
    def __init__(self):
        super(DifferenceMapGuidedEnhancement, self).__init__()
        self.conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, difference_map):
        attention = self.sigmoid(self.conv(difference_map))
        return features * attention

# Initialize Model
backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove classification layers

model = FPNWithROI(backbone=backbone, feature_channels=2048).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

def train_model(model, dataloader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, rois, labels in tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]"):
            inputs, rois, labels = inputs.to(device), rois.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs, rois)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

train_model(model, train_loader, criterion, optimizer)
