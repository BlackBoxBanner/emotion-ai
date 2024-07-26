import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Read the CSV file
df = pd.read_csv('./fer2013.csv')


# Function to convert pixel string to an image array
def string_to_image(pixel_string):
    pixel_array = np.fromstring(pixel_string, sep=' ')
    image = pixel_array.reshape(48, 48)
    return image


# Create directories for each emotion if they don't exist
emotions = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'neutral']
for emotion in tqdm(emotions, desc="Create directories for each emotion if they don't exist", total=len(emotions)):
    if not os.path.exists(emotion):
        os.makedirs(emotion)


# Function to save images according to emotion
def save_images_by_emotion(df):
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Saving images"):
        emotion = emotions[row['emotion']]
        img = string_to_image(row['pixels'])
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.savefig(f"{emotion}/{index}.png", bbox_inches='tight', pad_inches=0)
        plt.close()


# Save images
save_images_by_emotion(df)
