import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Define the paths to your image folders
deepfakes_folder = 'Deepfakes'
real_faces_folder = 'RealFaces'
# vasa_deepfakes_folder = 'VASA-Deepfakes' 


vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(device=device)
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def extract_vgg16_features(image_folder):
    features = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_folder, filename)
            img = load_img(img_path, target_size=(224, 224))
            img = img.convert('RGB')  # Convert image to RGB mode (in case it's grayscale or has alpha channel)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature = vgg16_model.predict(x)
            feature = feature.reshape(feature.shape[1], feature.shape[2], feature.shape[3])
            feature = np.mean(feature, axis=(0, 1))
            features.append(feature)
    return np.array(features)


def extract_facenet_features(image_folder):
    features = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_folder, filename)
            try:
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
            except (OSError, Image.DecompressionBombError) as e:
                print(f"Skipping invalid image {img_path}: {e}")
    return np.array(features)





if __name__ == "__main__":
    # deepfakes_vgg16_features = extract_vgg16_features(deepfakes_folder)
    # real_faces_vgg16_features = extract_vgg16_features(real_faces_folder)
    # vasa_deepfakes_vgg16_features = extract_vgg16_features(vasa_deepfakes_folder)

    deepfakes_facenet_features = extract_facenet_features(deepfakes_folder)
    real_faces_facenet_features = extract_facenet_features(real_faces_folder)
    # vasa_deepfakes_facenet_features = extract_facenet_features(vasa_deepfakes_folder)

    # Combine features and labels for VGG16
    # vgg16_all_features = np.concatenate((deepfakes_vgg16_features, real_faces_vgg16_features))
    # vgg16_labels = np.array(['Deepfake'] * len(deepfakes_vgg16_features) + ['Real'] * len(real_faces_vgg16_features))

    # Combine features and labels for OpenFace
    facenet_all_features = np.concatenate((deepfakes_facenet_features, real_faces_facenet_features))
    facenet_labels = np.array(['Deepfake'] * len(deepfakes_facenet_features) + ['Real'] * len(real_faces_facenet_features))

    #saving extracted data as weights
    np.save('facenet_labels.npy', facenet_labels)
    # np.save('vgg16_labels.npy', vgg16_labels)

    np.save('facenet_weights.npy', facenet_all_features)
    # np.save('vgg16_weights.npy', vgg16_all_features)

    #Training Classifier and saving model state
    clf = KNeighborsClassifier()
    clf.fit(facenet_all_features, facenet_labels)

    joblib.dump(clf, 'kn_classifier_weights.joblib')

    # np.save('kn_classifier_weights.npy', clf)



    # Perform t-SNE dimensionality reduction for VGG16 features
    # vgg16_tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    # vgg16_features_tsne = vgg16_tsne.fit_transform(vgg16_all_features)

    # Perform t-SNE dimensionality reduction for OpenFace features
    facenet_tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    facenet_features_tsne = facenet_tsne.fit_transform(facenet_all_features)

    # Plot the t-SNE visualizations
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    # plt.scatter(vgg16_features_tsne[vgg16_labels == 'Deepfake', 0], vgg16_features_tsne[vgg16_labels == 'Deepfake', 1], c='red', label='Deepfake')
    # plt.scatter(vgg16_features_tsne[vgg16_labels == 'Real', 0], vgg16_features_tsne[vgg16_labels == 'Real', 1], c='blue', label='Real')
    # # plt.scatter(vgg16_features_tsne[vgg16_labels == 'VASA-Deepfake', 0], vgg16_features_tsne[vgg16_labels == 'VASA-Deepfake', 1], c='green', label='VASA-Deepfake')  # Add VASA-Deepfake
    # plt.title('t-SNE Visualization of VGG16 Extracted Features')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(facenet_features_tsne[facenet_labels == 'Deepfake', 0], facenet_features_tsne[facenet_labels == 'Deepfake', 1], c='red', label='Deepfake')
    plt.scatter(facenet_features_tsne[facenet_labels == 'Real', 0], facenet_features_tsne[facenet_labels == 'Real', 1], c='blue', label='Real')
    # plt.scatter(facenet_features_tsne[facenet_labels == 'VASA-Deepfake', 0], facenet_features_tsne[facenet_labels == 'VASA-Deepfake', 1], c='green', label='VASA-Deepfake')  # Add VASA-Deepfake
    plt.title('t-SNE Visualization of OpenFace Extracted Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()

    plt.tight_layout()
    plt.show()
