import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
from googledriver import download
import numpy as np
from PIL import Image
import cv2
from DownloadGFile import download_file
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).eval()

model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=3, device=DEVICE)

DESTINATION_FILE_PATH = 'Weights&Labels/resnetinceptionv1_epoch.pth'
download_file(destination_file_path=DESTINATION_FILE_PATH)

checkpoint = torch.load("Weights&Labels/resnetinceptionv1_epoch.pth", map_location=torch.device('cpu'))
# model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

def predict(input_image: Image.Image):
    """Predict the label of the input_image"""
    if input_image.mode == 'RGBA':
        input_image = input_image.convert('RGB')
    face = mtcnn(input_image)
    if face is None:
        raise Exception('No face detected')
    face = face.unsqueeze(0)  # add the batch dimension
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
    
    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    prev_face = prev_face.astype('uint8')

    face = face.to(DEVICE)
    face = face.to(torch.float32)
    face = face / 255.0
    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

    target_layers = [model.block8.branch1[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0), ClassifierOutputTarget(1), ClassifierOutputTarget(2)]

    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
    face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

    with torch.no_grad():
        output = torch.softmax(model(face).squeeze(0), dim=0)
        class_indices = {0: 'real', 1: 'fake', 2: 'ai_generated'}
        prediction = class_indices[torch.argmax(output).item()]

        confidences = {
            'real': output[0].item(),
            'fake': output[1].item(),
            'ai_generated': output[2].item()
        }
    return confidences, prediction, face_with_mask

def calculate_metrics(true_labels, predictions):
    metrics = {
        "accuracy": accuracy_score(true_labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(true_labels, predictions),
        "f1_score": f1_score(true_labels, predictions, average='weighted'),
        "precision": precision_score(true_labels, predictions, average='weighted'),
        "recall": recall_score(true_labels, predictions, average='weighted')
    }
    return metrics
