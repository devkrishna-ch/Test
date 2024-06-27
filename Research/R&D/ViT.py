import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from PIL import Image
from torchvision import transforms
import timm

# Define the paths to your image folders
deepfakes_folder = 'Deepfakes'
real_faces_folder = 'RealFaces'
vasa_deepfakes_folder = 'VASA-Deepfakes'  # Add path to VASA-Deepfake folder

# Load the pre-trained ViT model (Vision Transformer)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
vit_model = timm.create_model('vit_base_patch16_384', pretrained=True).eval().to(device)

# Function to extract features using the ViT model
def extract_vit_features(image_folder):
    features = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path).convert('RGB')  # Load image and convert to RGB mode
            img = img.resize((384, 384))  # Resize image for ViT
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            img = transform(img).unsqueeze(0).to(device)  # Preprocess and add batch dimension
            with torch.no_grad():
                feature = vit_model(img).squeeze(0).cpu().numpy()  # Extract features
                features.append(feature)
    return np.array(features)

# Extract features for deepfakes, real faces, and VASA-Deepfakes using ViT
deepfakes_vit_features = extract_vit_features(deepfakes_folder)
real_faces_vit_features = extract_vit_features(real_faces_folder)
vasa_deepfakes_vit_features = extract_vit_features(vasa_deepfakes_folder)

# Combine features and labels for ViT
vit_all_features = np.concatenate((deepfakes_vit_features, real_faces_vit_features, vasa_deepfakes_vit_features))
vit_labels = np.array(['Deepfake'] * len(deepfakes_vit_features) + ['Real'] * len(real_faces_vit_features) + ['VASA-Deepfake'] * len(vasa_deepfakes_vit_features))

# Perform t-SNE dimensionality reduction for ViT
vit_tsne = TSNE(n_components=2, random_state=42)
vit_features_tsne = vit_tsne.fit_transform(vit_all_features)

# Plot the t-SNE visualization for ViT
plt.figure(figsize=(10, 8))
plt.scatter(vit_features_tsne[vit_labels == 'Deepfake', 0], vit_features_tsne[vit_labels == 'Deepfake', 1], c='red', label='Deepfake')
plt.scatter(vit_features_tsne[vit_labels == 'Real', 0], vit_features_tsne[vit_labels == 'Real', 1], c='blue', label='Real')
plt.scatter(vit_features_tsne[vit_labels == 'VASA-Deepfake', 0], vit_features_tsne[vit_labels == 'VASA-Deepfake', 1], c='green', label='VASA-Deepfake')
plt.title('t-SNE Visualization of ViT Extracted Features')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.tight_layout()
plt.show()
