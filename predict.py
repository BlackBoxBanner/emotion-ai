from PIL import Image
import torch
from torchvision import transforms
from utils import SimpleCNN, get_num_classes_from_model


def load_model(model_path, num_classes):
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict_emotion(image_path, model_path='trained_model/emotion_model.pth'):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the trained model
    # Replace num_classes with the number of classes in your dataset
    num_classes = get_num_classes_from_model(model_path)  # Adjust this based on your actual number of classes
    model = load_model(model_path, num_classes)

    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension

    # Predict emotion
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    # Emotion labels (ensure this matches your training labels)
    class_labels = ['angry', 'happy', 'sad', 'surprise', 'neutral']
    predicted_emotion = class_labels[predicted.item()]

    return predicted_emotion


# Example usage
if __name__ == '__main__':
    img_path = 'test_images/1.webp'  # Replace with your image path
    emotion = predict_emotion(img_path)
    print(f'Predicted emotion: {emotion}')
