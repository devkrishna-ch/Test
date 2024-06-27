import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from PIL import Image
from torchvision import transforms
import timm
from facenet_pytorch import InceptionResnetV1

# Define the paths to your image folders
deepfakes_folder = 'Deepfakes'
real_faces_folder = 'RealFaces'

# Load the pre-trained ViT model (Vision Transformer)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
vit_model = timm.create_model('vit_base_patch16_224', pretrained=True).eval().to(device)

# Load the pre-trained OpenFace (InceptionResnetV1) model
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Function to extract features using the ViT model
def extract_vit_features(image_folder):
    features = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path).convert('RGB')  # Load image and convert to RGB mode
            img = img.resize((224, 224))  # Resize image to (224, 224) for ViT
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            img = transform(img).unsqueeze(0).to(device)  # Preprocess and add batch dimension
            with torch.no_grad():
                feature = vit_model(img).squeeze(0).cpu().numpy()  # Extract features
                features.append(feature)
    return np.array(features)


# Function to extract features using the OpenFace model
def extract_facenet_features(image_folder):
    features = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path).convert('RGB')  # Load image and convert to RGB mode
            img = img.resize((160, 160))  # Resize image to (160, 160)
            x = np.array(img)  # Convert image to numpy array
            x = np.expand_dims(x, axis=0)  # Add batch dimension
            x = (x / 255.0 - 0.5) * 2.0  # Preprocess input
            x = np.transpose(x, (0, 3, 1, 2))  # Transpose to channel-first format (expected by PyTorch)
            x = torch.from_numpy(x).to(device)  # Convert to torch.Tensor
            facenet_model.double()  # Convert model weights to Double data type
            embedding = facenet_model(x).detach().cpu().numpy()
            features.append(embedding[0])
    return np.array(features)


# Extract features for deepfakes and real faces using ViT and OpenFace
deepfakes_vit_features = extract_vit_features(deepfakes_folder)
real_faces_vit_features = extract_vit_features(real_faces_folder)
deepfakes_facenet_features = extract_facenet_features(deepfakes_folder)
real_faces_facenet_features = extract_facenet_features(real_faces_folder)

# Reduce ViT features to 512 dimensions to match openfaces dimensions
deepfakes_vit_features = deepfakes_vit_features[:, :512]
real_faces_vit_features = real_faces_vit_features[:, :512]

# Combine ViT and OpenFace features
vit_all_features = np.concatenate((deepfakes_vit_features, real_faces_vit_features))
facenet_all_features = np.concatenate((deepfakes_facenet_features, real_faces_facenet_features))
vit_labels = np.array(['Deepfake'] * len(deepfakes_vit_features) + ['Real'] * len(real_faces_vit_features))
facenet_labels = np.array(['Deepfake'] * len(deepfakes_facenet_features) + ['Real'] * len(real_faces_facenet_features))

# Perform t-SNE dimensionality reduction for combined features
combined_features = np.concatenate((vit_all_features, facenet_all_features))
combined_labels = np.concatenate((vit_labels, facenet_labels))

tsne = TSNE(n_components=2, random_state=42)
combined_features_tsne = tsne.fit_transform(combined_features)

# Plot the t-SNE visualization for combined ViT and OpenFace features
plt.figure(figsize=(8, 6))
plt.scatter(combined_features_tsne[combined_labels == 'Deepfake', 0], combined_features_tsne[combined_labels == 'Deepfake', 1], c='red', label='Deepfake')
plt.scatter(combined_features_tsne[combined_labels == 'Real', 0], combined_features_tsne[combined_labels == 'Real', 1], c='blue', label='Real')
plt.title('t-SNE Visualization of Combined ViT and OpenFace Features')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.tight_layout()
plt.show()
