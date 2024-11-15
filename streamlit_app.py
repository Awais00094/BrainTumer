import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os
import streamlit as st
import zipfile

#import data set from the drive 

import sys
import requests
import kagglehub
     
def download_dataset():
    path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    print("Path to dataset files:", path)
    st.write("File downloaded successfully.")
    # st.write(path)
    Prepare_model(path)
    
    # extract_dataset(path=path)



# Streamlit UI

st.title("Brain Tumor Detection")

def extract_dataset(path):
    with zipfile.ZipFile(path,"r") as zip_ref:
        zip_ref.extractall("BrainTumerDataSet")
    st.write("data-set extracted")
    
def Prepare_model(dataset_path):

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms for the testing dataset
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Custom Dataset class for loading brain tumor data
    class BrainTumorDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
            self.image_paths = []
            self.labels = []

            for label, class_name in enumerate(self.classes):
                class_dir = os.path.join(root_dir, class_name)
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label

    model_path = 'trained_model.pth'
    state_dict = torch.load(model_path, map_location=device)


    # Define your model architecture
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.fc1 = nn.Linear(64 * 28 * 28, 512)
            self.fc2 = nn.Linear(512, 4)  # 4 classes

        def forward(self, x):
            x = self.pool(nn.functional.relu(self.conv1(x)))
            x = self.pool(nn.functional.relu(self.conv2(x)))
            x = self.pool(nn.functional.relu(self.conv3(x)))
            x = x.view(-1, 64 * 28 * 28)  # Flatten
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
            return x




    # Instantiate your model
    model = CNNModel().to(device)

    # Load state_dict into your model
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode

    # Create dataset instance for testing
    test_dataset = BrainTumorDataset(root_dir=dataset_path+'/Testing', transform=test_transform)

    # Randomly select 10 images from the test set
    num_images = 20
    random_indices = np.random.choice(len(test_dataset), num_images, replace=False)


    print("Actual Label | Predicted Label")
    print("-----------------------------")
    st.write("Actual Label | Predicted Label")
    st.write("-----------------------------")

    # Perform inference on each selected image
    for idx in random_indices:
        image, label = test_dataset[idx]
        image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output.data, 1)

        actual_label = test_dataset.classes[label]
        predicted_label = test_dataset.classes[predicted.item()]

        print(f"{actual_label:12} | {predicted_label:14}")
        st.write(f"{actual_label:12} | {predicted_label:14}")
        

if __name__ == "__main__":
    download_dataset()
    # extract_dataset()
