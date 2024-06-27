import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import torch
import cv2
from deepface import DeepFace
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import tensorflow as tf
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
import insightface

# Define the paths to your image folders
deepfakes_folder = 'Deepfakes'
real_faces_folder = 'RealFaces'

# Load the pre-trained models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Google's Facenet
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Facebook's DeepFace
deepface_model = DeepFace.build_model('VGG-Face')

# Oxford's VGG Face
vggface_model = VGG16(weights='imagenet', include_top=False)

# OpenFace from Carnegie Mellon University
mtcnn = MTCNN(device=device)
openface_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Function to extract features using Google's Facenet
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
            x = torch.from_numpy(x).double().to(device)  # Convert to torch.Tensor and Double type
            facenet_model.double()  # Convert model weights to Double data type
            embedding = facenet_model(x).detach().cpu().numpy()
            features.append(embedding[0])
    return np.array(features)


# DeepFace
def extract_deepface_features(image_folder):
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
            x = torch.from_numpy(x).double().to(device)  # Convert to torch.Tensor and Double type
            embedding = deepface_model(x).detach().cpu().numpy()
            features.append(embedding.squeeze())  # Remove the batch dimension
    # Pad or resize embeddings to a fixed shape if necessary
    max_shape = max(feature.shape for feature in features)
    features = [np.pad(feature, ((0, max_shape[0] - feature.shape[0]), (0, 0)), mode='constant') for feature in features]
    return np.array(features)


# Function to extract features using Oxford's VGG Face
def extract_vggface_features(image_folder):
    features = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_folder, filename)
            img = tf_image.load_img(img_path, target_size=(224, 224))
            x = tf_image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = vgg_preprocess_input(x)
            features.append(vggface_model.predict(x).flatten())
    return np.array(features)

# Function to extract features using OpenFace from Carnegie Mellon University
def extract_openface_features(image_folder):
    features = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path).convert('RGB')
            img_cropped = mtcnn(img)
            if img_cropped is not None:
                img_cropped = img_cropped.unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = openface_model(img_cropped).squeeze().detach().cpu().numpy()
                features.append(embedding)
    return np.array(features)

# Extract features for deepfakes and real faces
deepfakes_facenet_features = extract_facenet_features(deepfakes_folder)
real_faces_facenet_features = extract_facenet_features(real_faces_folder)

deepfakes_deepface_features = extract_deepface_features(deepfakes_folder)
real_faces_deepface_features = extract_deepface_features(real_faces_folder)

deepfakes_vggface_features = extract_vggface_features(deepfakes_folder)
real_faces_vggface_features = extract_vggface_features(real_faces_folder)

deepfakes_openface_features = extract_openface_features(deepfakes_folder)
real_faces_openface_features = extract_openface_features(real_faces_folder)

# Combine features and labels
facenet_all_features = np.concatenate((deepfakes_facenet_features, real_faces_facenet_features))
deepface_all_features = np.concatenate((deepfakes_deepface_features, real_faces_deepface_features))
vggface_all_features = np.concatenate((deepfakes_vggface_features, real_faces_vggface_features))
openface_all_features = np.concatenate((deepfakes_openface_features, real_faces_openface_features))

facenet_labels = np.array(['Deepfake'] * len(deepfakes_facenet_features) + ['Real'] * len(real_faces_facenet_features))
deepface_labels = np.array(['Deepfake'] * len(deepfakes_deepface_features) + ['Real'] * len(real_faces_deepface_features))
vggface_labels = np.array(['Deepfake'] * len(deepfakes_vggface_features) + ['Real'] * len(real_faces_vggface_features))
openface_labels = np.array(['Deepfake'] * len(deepfakes_openface_features) + ['Real'] * len(real_faces_openface_features))

# Perform t-SNE dimensionality reduction
facenet_tsne = TSNE(n_components=2, random_state=42)
facenet_features_tsne = facenet_tsne.fit_transform(facenet_all_features)

deepface_tsne = TSNE(n_components=2, random_state=42)
deepface_features_tsne = deepface_tsne.fit_transform(deepface_all_features)

vggface_tsne = TSNE(n_components=2, random_state=42)
vggface_features_tsne = vggface_tsne.fit_transform(vggface_all_features)

openface_tsne = TSNE(n_components=2, random_state=42)
openface_features_tsne = openface_tsne.fit_transform(openface_all_features)

# Plot the t-SNE visualizations
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.scatter(facenet_features_tsne[facenet_labels == 'Deepfake', 0], facenet_features_tsne[facenet_labels == 'Deepfake', 1], c='red', label='Deepfake')
plt.scatter(facenet_features_tsne[facenet_labels == 'Real', 0], facenet_features_tsne[facenet_labels == 'Real', 1], c='blue', label='Real')
plt.title('t-SNE Visualization of Facenet Extracted Features')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(deepface_features_tsne[deepface_labels == 'Deepfake', 0], deepface_features_tsne[deepface_labels == 'Deepfake', 1], c='red', label='Deepfake')
plt.scatter(deepface_features_tsne[deepface_labels == 'Real', 0], deepface_features_tsne[deepface_labels == 'Real', 1], c='blue', label='Real')
plt.title('t-SNE Visualization of DeepFace Extracted Features')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()

plt.subplot(2, 2, 3)
plt.scatter(vggface_features_tsne[vggface_labels == 'Deepfake', 0], vggface_features_tsne[vggface_labels == 'Deepfake', 1], c='red', label='Deepfake')
plt.scatter(vggface_features_tsne[vggface_labels == 'Real', 0], vggface_features_tsne[vggface_labels == 'Real', 1], c='blue', label='Real')
plt.title('t-SNE Visualization of VGG Face Extracted Features')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(openface_features_tsne[openface_labels == 'Deepfake', 0], openface_features_tsne[openface_labels == 'Deepfake', 1], c='red', label='Deepfake')
plt.scatter(openface_features_tsne[openface_labels == 'Real', 0], openface_features_tsne[openface_labels == 'Real', 1], c='blue', label='Real')
plt.title('t-SNE Visualization of OpenFace Extracted Features')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()

plt.tight_layout()
plt.show()
