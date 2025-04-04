import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
import multiprocessing
from torch.utils.data import Dataset
import time
import re
import numpy as np
import os
import sys

MODEL_FOLDER = "./models"
GRAPHIC_FOLDER = "./graphical_data"
DATA_FOLDER = './reduced_data'


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, label = self.dataset[index]
        return data, label, index

    def __len__(self):
        return len(self.dataset)

# Ensures both folders exist for smooth operations
if not (os.path.exists(MODEL_FOLDER) and os.path.exists(GRAPHIC_FOLDER)):
    print("Error: Either Model Folder or Graphical Save Folder do not exist")
    sys.exit(1)

def main():
    # Uses CUDA


    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)
    if (device.type != "cuda"):
        print("Unable to access GPU")
        sys.exit(1)
    print(torch.cuda.get_device_name(0))  # Prints the name of your GPU

    # Transformations for training and validation
    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # 3 channels for ResNet
        transforms.Resize((224, 224)),  # Scale for resnet 224x224 images
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ColorJitter( brightness=0.1, contrast=0.1,  saturation=0.1),
        transforms.ToTensor(),
        transforms.RandomRotation(15),  
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
    ])

    
    

    ################################## DATA SETUP ############################################
    # Set up data
    train_dataset = datasets.ImageFolder(os.path.join(DATA_FOLDER, "train"), transform=transform_train)
    train_dataset = IndexedDataset(train_dataset)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_FOLDER, "val"), transform=transform_test)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_FOLDER, "test"), transform=transform_test)
    

    print("Data setup complete")


    # Loads the data
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)  # Optimized DataLoader
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True)
    print("Data loading complete")
    
    
    # Checks if I have a pretrained checkpoint in folder
    largest_number = 0
    largest_file = None

    # Finds the newest model in the case of saving multiple times
    for file_name in os.listdir(MODEL_FOLDER):
        match = re.search(r'(\d+)(?=\.\w+$)', file_name)
        if match:
            number = int(match.group(1))
            if number > largest_number:
                largest_number = number
                largest_file = file_name


    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear( model.fc.in_features, len(train_dataset.dataset.classes))
        
    ) 
    model = model.to(device)

    # loads optimizer up here
    optimizer = optim.Adam([
        {'params': model.fc.parameters(), 'lr': 0.001},  # high LR for fs layer
        {'params': model.layer4.parameters(), 'lr': 1e-4}  # low LR for finetune layers
    ])

    
    # either creates a new model, or loads most recent one            
    if largest_file:
        print("Loading Previous Model")
        print(os.path.join(MODEL_FOLDER,largest_file))
        checkpoint = torch.load(os.path.join(MODEL_FOLDER,largest_file))

        model.load_state_dict(checkpoint["model_state_dict"])
       
        clean_sample_indices = checkpoint.get("clean_indices", None)
        print(f"there are currently {len(clean_sample_indices)} clean files")
        # check to make sure that there is clean_indicies
        if clean_sample_indices:
            clean_dataset = torch.utils.data.Subset(train_dataset, clean_sample_indices)
            clean_loader = DataLoader(clean_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)
            
        else:
            clean_dataset = train_dataset
            clean_loader = train_loader
            clean_sample_indices = list(range(len(clean_dataset)))
        # load optimizer from the C_P
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        

    else:
        print("No files with numbers at the end were found. Creating new model")   
        clean_dataset = train_dataset
        clean_loader = train_loader
        clean_sample_indices = list(range(len(train_dataset)))  # use full dataset initially
   





    for param in model.parameters():
            param.requires_grad = False  # freeze all layers
    for param in model.fc.parameters():
        param.requires_grad = True  # fine tune FC layer
    for param in model.layer4.parameters():
        param.requires_grad = True


    scaler = torch.amp.GradScaler("cuda")

    

    # Loss functions
    # had to modify to reduce noise
    criterion = nn.CrossEntropyLoss(reduction='none')
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    

    






    ################################### TRAINING #####################################
    sample_losses = {i: 0.0 for i in clean_sample_indices}  
    print(f"sample losses size is {len(sample_losses)}")


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)

    # Benchmarking Data
    epoch_values = []
    loss_values_train = []
    loss_values_val = []
    accuracy_values_train = []
    accuracy_values_val = []

    print("Starting Epochs")
    num_epochs = 10
    cp_epoch = 0
    for epoch in range(num_epochs):
        # Cleanup
        


        model.train()
        start_time = time.time()  #  start time
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch_idx, (images, labels, indices), in enumerate(clean_loader):
            if (batch_idx % 10 == 0):
                 print(f"Processing Batch #{batch_idx}/{len(clean_loader)}")
            images, labels = images.to(device), labels.to(device)



            optimizer.zero_grad()
            with torch.autocast('cuda'):
                outputs = model(images)
                batch_losses = criterion(outputs, labels)
                loss = batch_losses.mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            _, predicted = torch.max(outputs, 1)
            running_train_loss += loss.item()
            correct_train +=  (predicted == labels).sum().item()
            total_train += labels.size(0)

            
            batch_losses_np = batch_losses.detach().cpu().numpy()
            for i, index in enumerate(indices):
                index = int(index)
                sample_losses[index] += batch_losses_np[i]









        # Validation phase
        model.eval()  
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad(), torch.autocast('cuda'):
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                batch_losses = criterion(outputs, labels)
                loss = batch_losses.mean()
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        
        end_time = time.time()  # end time
        train_loss = running_train_loss / len(clean_loader)
        val_loss = running_val_loss / len(val_loader)
        train_accuracy = (100 * correct_train/total_train)
        val_accuracy = 100 * correct_val / total_val

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}, Traing Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f} ")
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        print(f"Time: {end_time - start_time:.2f} seconds")
        
        loss_values_train.append(train_loss)
        loss_values_val.append(val_loss)
        accuracy_values_train.append(train_accuracy)
        accuracy_values_val.append(val_accuracy)
        epoch_values.append(epoch)




        # Cleanup
        if (epoch == 5):
            sorted_losses = sorted(sample_losses.items(), key=lambda x: x[1], reverse=True)
            num_noisy_samples = int(0.06 * len(clean_dataset))  # Top 7%
            noisy_sample_indices = [index for index, loss in sorted_losses[:num_noisy_samples]]
            
            clean_sample_indices = list(set(clean_sample_indices) - set(noisy_sample_indices))
            clean_dataset = torch.utils.data.Subset(train_dataset, clean_sample_indices)
            clean_loader = DataLoader(clean_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)
            print(f"Filtered {len(noisy_sample_indices)} noisy samples. Clean dataset size: {len(clean_sample_indices)}.")

            sample_losses = {i: sample_losses.get(i, 0.0) for i in clean_sample_indices}


    

    ############################## SAVING MODEL #################################
    MODEL_SAVE_PATH = os.path.join(MODEL_FOLDER, f'resnet50_document_classifier{largest_number + 1}.pth')
    # torch.save(model, MODEL_SAVE_PATH)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "clean_indices": clean_sample_indices if clean_sample_indices else list(range(len(train_dataset))),
    }, MODEL_SAVE_PATH)
    print(f"Model saved as resnet50_document_classifier{largest_number + 1}.pth")  

    save_file = os.path.join(GRAPHIC_FOLDER, f"model{largest_number + 1}data.npz")
    np.savez(save_file, epoch_values, loss_values_train, loss_values_val, accuracy_values_train, accuracy_values_val)
    print(f"Data saved as model{largest_number + 1}data.pth")  

    print("Finished running")
    
if __name__ == "__main__":
    main()