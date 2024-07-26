import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 18 * 18, 512)  # Adjust size based on your image dimensions
        self.fc2 = nn.Linear(512, num_classes)  # Number of classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 18 * 18)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# get_num_classes_from_model
def get_num_classes_from_model(model_path):
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Extract num_classes from the checkpoint
    num_classes = checkpoint['fc2.weight'].size(0)

    return num_classes

def get_image_paths(folder):
    from PIL import Image

    # Get all file paths in the folder
    image_paths = []
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            # Open the file and verify it's an image
            with Image.open(file_path) as img:
                img.verify()  # Verify the file is an image
                image_paths.append(file_path)
        except (IOError, SyntaxError) as e:
            # Ignore if the file is not a valid image
            continue
    return image_paths