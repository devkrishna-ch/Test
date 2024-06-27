import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# Define the paths to your image folders
deepfakes_folder = 'Deepfakes'
real_faces_folder = 'RealFaces'
bgfilter_deepfakes_folder = 'Bgfilter-Deepfakes'
bgfilter_realfaces_folder = 'Bgfilter-RealFaces'

# Load the pre-trained VGG16 model
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Load the pre-trained OpenFace (InceptionResnetV1) model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(device=device)
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Function to extract features using the VGG16 model
def extract_vgg16_features(image_folder):
    features = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_folder, filename)
            img = load_img(img_path, target_size=(224, 224))
            img = img.convert('RGB') 
            print(f"Image shape before preprocessing: {img.size}")
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature = vgg16_model.predict(x)
            feature = feature.reshape(feature.shape[1], feature.shape[2], feature.shape[3])
            feature = np.mean(feature, axis=(0, 1))
            features.append(feature)
    return np.array(features)

from PIL import Image


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

# Extract features for deepfakes, real faces, and their background-filtered versions
deepfakes_vgg16_features = extract_vgg16_features(deepfakes_folder)
real_faces_vgg16_features = extract_vgg16_features(real_faces_folder)
bgfilter_deepfakes_vgg16_features = extract_vgg16_features(bgfilter_deepfakes_folder)
bgfilter_realfaces_vgg16_features = extract_vgg16_features(bgfilter_realfaces_folder)

deepfakes_facenet_features = extract_facenet_features(deepfakes_folder)
real_faces_facenet_features = extract_facenet_features(real_faces_folder)
bgfilter_deepfakes_facenet_features = extract_facenet_features(bgfilter_deepfakes_folder)
bgfilter_realfaces_facenet_features = extract_facenet_features(bgfilter_realfaces_folder)

# Combine features and labels
vgg16_all_features = np.concatenate((deepfakes_vgg16_features, real_faces_vgg16_features, bgfilter_deepfakes_vgg16_features, bgfilter_realfaces_vgg16_features))
facenet_all_features = np.concatenate((deepfakes_facenet_features, real_faces_facenet_features, bgfilter_deepfakes_facenet_features, bgfilter_realfaces_facenet_features))
vgg16_labels = np.array(['Deepfake'] * len(deepfakes_vgg16_features) + ['Real'] * len(real_faces_vgg16_features) + ['Bgfilter-Deepfake'] * len(bgfilter_deepfakes_vgg16_features) + ['Bgfilter-Real'] * len(bgfilter_realfaces_vgg16_features))
facenet_labels = np.array(['Deepfake'] * len(deepfakes_facenet_features) + ['Real'] * len(real_faces_facenet_features) + ['Bgfilter-Deepfake'] * len(bgfilter_deepfakes_facenet_features) + ['Bgfilter-Real'] * len(bgfilter_realfaces_facenet_features))

# Perform t-SNE dimensionality reduction
vgg16_tsne = TSNE(n_components=2, random_state=42)
vgg16_features_tsne = vgg16_tsne.fit_transform(vgg16_all_features)
facenet_tsne = TSNE(n_components=2, random_state=42)
facenet_features_tsne = facenet_tsne.fit_transform(facenet_all_features)

# Plot the t-SNE visualizations
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.scatter(vgg16_features_tsne[vgg16_labels == 'Deepfake', 0], vgg16_features_tsne[vgg16_labels == 'Deepfake', 1], c='red', label='Deepfake')
plt.scatter(vgg16_features_tsne[vgg16_labels == 'Bgfilter-Deepfake', 0], vgg16_features_tsne[vgg16_labels == 'Bgfilter-Deepfake', 1], c='orange', label='Bgfilter-Deepfake')
plt.scatter(vgg16_features_tsne[vgg16_labels == 'Real', 0], vgg16_features_tsne[vgg16_labels == 'Real', 1], c='blue', label='Real')
plt.scatter(vgg16_features_tsne[vgg16_labels == 'Bgfilter-Real', 0], vgg16_features_tsne[vgg16_labels == 'Bgfilter-Real', 1], c='green', label='Bgfilter-Real')
plt.title('t-SNE Visualization of VGG16 Extracted Features')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(facenet_features_tsne[facenet_labels == 'Deepfake', 0], facenet_features_tsne[facenet_labels == 'Deepfake', 1], c='red', label='Deepfake')
plt.scatter(vgg16_features_tsne[vgg16_labels == 'Bgfilter-Deepfake', 0], vgg16_features_tsne[vgg16_labels == 'Bgfilter-Deepfake', 1], c='orange', label='Bgfilter-Deepfake')
plt.scatter(facenet_features_tsne[facenet_labels == 'Real', 0], facenet_features_tsne[facenet_labels == 'Real', 1], c='blue', label='Real')
plt.scatter(facenet_features_tsne[facenet_labels == 'Bgfilter-Real', 0], facenet_features_tsne[facenet_labels == 'Bgfilter-Real', 1], c='green', label='Bgfilter-Real')
plt.title('t-SNE Visualization of OpenFace Extracted Features')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()

plt.tight_layout()
plt.show()