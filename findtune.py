import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import SimpleCNN  # Assuming SimpleCNN is your model architecture


def fine_tune_model(train_data_dir, model_save_path, pre_trained_model_path):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the training data
    train_data = datasets.ImageFolder(root=train_data_dir, transform=transform)
    num_classes = len(train_data.classes)  # Number of classes
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)

    # Initialize the model, loss function, and optimizer
    model = SimpleCNN(num_classes=num_classes)

    # Load the pre-trained model weights
    model.load_state_dict(torch.load(pre_trained_model_path))

    # Optionally, freeze some layers if needed
    # for param in model.parameters():
    #     param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Smaller learning rate for fine-tuning

    # Training the model
    model.train()
    for epoch in range(10):  # Number of epochs for fine-tuning
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}', leave=False)
        for inputs, labels in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (len(train_loader.dataset) / train_loader.batch_size))

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

    # Save the fine-tuned model
    torch.save(model.state_dict(), model_save_path)


def main():
    train_data_dir = 'train_data'  # Replace with your train data directory
    pre_trained_model_path = 'trained_model/fine_tuned_emotion_model.pth'  # Path to pre-trained model
    fine_tuned_model_save_path = 'trained_model/fine_tuned_emotion_model_2.pth'  # Path to save the fine-tuned model

    fine_tune_model(train_data_dir, fine_tuned_model_save_path, pre_trained_model_path)


if __name__ == '__main__':
    main()
