from PIL import Image
import torch
from torchvision import transforms
import os
import pandas as pd
from utils import SimpleCNN, get_num_classes_from_model


def load_model(model_path, num_classes):
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict_emotion(image_path, model_path='trained_model/fine_tuned_emotion_model_2.pth'):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the trained model
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
    class_labels = ['angry', 'happy', 'sad', 'surprise', 'neutral', 'fear', 'disgust']
    predicted_emotion = class_labels[predicted.item()]

    return predicted_emotion


def analyze_images_in_folders(folder_paths, model_path='trained_model/fine_tuned_emotion_model_2.pth'):
    results = []
    errors = []

    for folder_path in folder_paths:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                img_path = os.path.join(folder_path, filename)
                try:
                    emotion = predict_emotion(img_path, model_path)
                    analysis = {
                        'file_path': img_path,
                        'emotion': emotion
                    }
                    results.append(analysis)
                except Exception as e:
                    errors.append(img_path)
                    print(f"Error analyzing {img_path}: {e}")

    return results, errors


def main():
    folder_paths = [
        "image/child",
        "image/teen",
        "image/adults"
    ]

    repeat = 1

    for index in range(repeat):
        results, errors = analyze_images_in_folders(folder_paths)

        df = pd.DataFrame(results)
        print("Analysis results DataFrame:")
        print(df)

        result_file = f'result/deepface_analysis_results_{index}.csv'
        df.to_csv(result_file, index=False)
        print(f"Analysis results saved to {result_file}")

        if errors:
            print("\nImages that could not be analyzed:")
            for error in errors:
                print(error)
            error_file = f'result/deepface_analysis_errors_{index}.txt'
            with open(error_file, 'w') as f:
                for error in errors:
                    f.write(f"{error}\n")
            print(f"Error list saved to {error_file}")


if __name__ == "__main__":
    main()
