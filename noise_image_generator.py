import os
import logging
import pandas as pd
from collections import defaultdict


class NoiseImageGenerator:
    def __init__(self):
        self.original_image_filepath = "dataset/original/train"

    def load_original_image(self):
        image_label_dict = defaultdict(list)
        for i in range(7):
            emotion_image_path = os.path.join(self.original_image_filepath, str(i))
            emotion_image_files = os.listdir(emotion_image_path)
            image_label_dict["Image"] += [os.path.join(emotion_image_path, path) for path in emotion_image_files]
            image_label_dict["Label"] += [0 for i in range(len(emotion_image_files))]
            image_label_df = pd.DataFrame(image_label_dict)
            print(image_label_df)
        return image_label_df

    def augment_image_with_noise(self):
        pass


if __name__ == "__main__":
    noise_generator = NoiseImageGenerator()
    noise_generator.load_original_image()
