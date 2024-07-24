from predict import predict_emotion


def main():
    result = predict_emotion(image_path='train_data/neutral/4.png',
                             model_path="trained_model/emotion_model.pth")

    print(result)
    return


if __name__ == '__main__':
    main()
