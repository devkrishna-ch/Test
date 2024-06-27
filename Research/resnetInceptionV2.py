import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from PIL import Image
from torchvision import transforms, models

# Define the paths to your image folders
deepfakes_folder = 'Deepfakes'
real_faces_folder = 'RealFaces'

# Load the pre-trained InceptionV3 model
inception_v3_model = models.inception_v3(pretrained=True, aux_logits=True).eval()

# Function to extract features using the InceptionV3 model
def extract_inception_v3_features(image_folder):
    features = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path).convert('RGB')  # Load image and convert to RGB mode
            img = img.resize((299, 299))  # Resize image to (299, 299) for InceptionV3
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            img = transform(img).unsqueeze(0)  # Preprocess and add batch dimension
            with torch.no_grad():
                feature = inception_v3_model(img).squeeze().cpu().numpy()  # Extract features
            features.append(feature)
    return np.array(features)

# Extract features for deepfakes and real faces
deepfakes_inception_v3_features = extract_inception_v3_features(deepfakes_folder)
real_faces_inception_v3_features = extract_inception_v3_features(real_faces_folder)

# Perform t-SNE dimensionality reduction for InceptionV3 features
inception_v3_tsne = TSNE(n_components=2, random_state=42)
inception_v3_features_tsne = inception_v3_tsne.fit_transform(np.concatenate((deepfakes_inception_v3_features, real_faces_inception_v3_features)))

# Plot the t-SNE visualizations
plt.figure(figsize=(8, 6))
plt.scatter(inception_v3_features_tsne[:len(deepfakes_inception_v3_features), 0], inception_v3_features_tsne[:len(deepfakes_inception_v3_features), 1], c='red', label='Deepfake')
plt.scatter(inception_v3_features_tsne[len(deepfakes_inception_v3_features):, 0], inception_v3_features_tsne[len(deepfakes_inception_v3_features):, 1], c='blue', label='Real')
plt.title('t-SNE Visualization of InceptionV3 Extracted Features')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.tight_layout()
plt.show()
