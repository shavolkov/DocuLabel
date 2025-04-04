import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import os
import cv2


## A Computer vision model that is made to differentiate documents

MODEL_DIR_PATH = "./visualize_model"

def load_model(name):
    model_path = os.path.join(MODEL_DIR_PATH, name)
    print(model_path)
    checkpoint = torch.load(model_path)
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear( model.fc.in_features, 16)
        
    ) 
    if not os.path.exists(model_path):
        print("Error: Model not in folder")

    
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def grad_cam(model, images, target_class, layer_name='layer4'):
    model.eval()
    gradients = []
    activations = []

    # Appends activations
    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    layer = dict(model.named_modules())[layer_name]
    layer.register_forward_hook(forward_hook)
    layer.register_backward_hook(backward_hook)

    # Forward pass
    outputs = model(images)
    one_hot = torch.zeros(outputs.shape, device=images.device)
    one_hot[0, target_class] = 1

    # Backward pass
    model.zero_grad()
    outputs.backward(gradient=one_hot)

    # Gets gradients and activation
    gradients = gradients[0].detach()
    activations = activations[0].detach()

    #Average Pooling
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    grad_cam_map = torch.sum(weights * activations, dim=1).squeeze()

    # Normalize
    grad_cam_map = torch.relu(grad_cam_map)
    grad_cam_map = grad_cam_map - grad_cam_map.min()
    grad_cam_map = grad_cam_map / grad_cam_map.max()
    return grad_cam_map.cpu().numpy()


def overlay_grad_cam(image, cam_map, alpha=0.5):
    # Convert grayscale to a heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Normalize original image
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())


    if len(image.shape) == 2:  # Grayscale image
     image = np.stack([image] * 3, axis=-1)  # Convert to (H, W, 3)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(np.uint8(255 * image), 1 - alpha, heatmap, alpha, 0)
    return overlay


transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
    ])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)
    if (device.type != "cuda"):
        print("Unable to access GPU")
    print(torch.cuda.get_device_name(0))  # Prints the name of your GPU


    DATA_FOLDER = './reduced_data'

    val_dataset = datasets.ImageFolder(os.path.join(DATA_FOLDER, "val"), transform=transform_test)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)
    model = load_model("resnet50_document_classifier2.pth")
    model = model.to(device)

    images, labels = next(iter(val_loader))  # Get a batch of images
    images = images.to(device)

    # Pick the first image and its true label
    image = images[3].unsqueeze(0)
    print(labels)
    target_class = labels[3].item()

    cam_map = grad_cam(model, image, target_class)

    # Visualize overlay
    overlay = overlay_grad_cam(image.squeeze(), cam_map)
    plt.imshow(overlay)
    plt.title(f"Grad-CAM for Class {target_class}")
    plt.axis('off')
    plt.show()

        
if __name__ == "__main__":
    main()