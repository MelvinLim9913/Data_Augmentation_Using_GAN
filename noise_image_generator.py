import os
import logging
import numpy as np
import tqdm
import PIL
import argparse
import imgaug.augmenters as iaa
import pandas as pd
from collections import defaultdict


class NoiseImageGenerator:
    def __init__(self, noise_type, a):
        self.logger = logging.getLogger('noise' + __name__)
        self.original_image_filepath = "dataset/original/train"
        self.generated_image_filepath = f"dataset/noise/{noise_type}_{a}/"
        os.makedirs(self.generated_image_filepath, exist_ok=True)
        self.logger.info(f"Type of noise chosen: {noise_type}")
        self.logger.info(f"Parameter for noise chosen: {a}")
        self.sequence = None
        self.logger.info(f"Image generated will be saved into {self.generated_image_filepath}")

    def load_original_image(self):
        image_label_dict = defaultdict(list)
        for i in range(7):
            emotion_image_path = os.path.join(self.original_image_filepath, str(i))
            emotion_image_files = os.listdir(emotion_image_path)
            image_label_dict["Image"] += [os.path.join(emotion_image_path, path) for path in emotion_image_files]
            image_label_dict["Label"] += [i for _ in range(len(emotion_image_files))]
        image_label_df = pd.DataFrame(image_label_dict)
        image_label_df = image_label_df.astype(str)
        return image_label_df

    def get_noise_network(self, noise_type, a):
        if noise_type == "gaussian":
            self.sequence = iaa.AdditiveGaussianNoise(scale=a)
        elif noise_type == "poisson":
            self.sequence = iaa.AdditivePoissonNoise(lam=a)
        elif noise == "laplace":
            self.sequence = iaa.AdditiveLaplaceNoise(scale=a)
        elif noise == "salt&pepper":
            self.sequence = iaa.SaltAndPepper(p=a)

    def get_num_of_image_to_generate(self):
        image_generation_cycle_dict = {}
        image_label_df = self.load_original_image()
        for label in image_label_df["Label"].unique():
            image_generation_cycle_dict[str(label)] = 10000 // (image_label_df["Label"] == label).sum()
        self.logger.info(
            f"How many times an image the image will be generated in each class:\n{image_generation_cycle_dict}")
        return image_label_df, image_generation_cycle_dict

    def generate_noise_image(self):
        image_label_df, image_generation_cycle_dict = self.get_num_of_image_to_generate()
        for i in tqdm.tqdm(range(len(image_label_df["Image"]))):
            images = np.array(PIL.Image.open(image_label_df["Image"][i]).convert("RGB"))
            number_of_times_adding_noise = image_generation_cycle_dict[image_label_df["Label"][i]]

            for k in range(number_of_times_adding_noise):
                augmented_image = PIL.Image.fromarray(self.sequence(image=images))
                augmented_image.save(
                    self.generated_image_filepath + image_label_df["Label"][i] + "/" + "Noise" + str(i) + str(
                        k) + ".png")
        self.logger.info("Noise image generation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",
                        default="gaussian",
                        type=str,
                        choices=["gaussian", "poisson", "laplace" "salt&pepper"])
    parser.add_argument("--stddv",
                        type=float)
    args = parser.parse_args()
    noise = args.type()
    standard_deviation = args.stddv

    noise_generator = NoiseImageGenerator(noise_type=noise, a=standard_deviation)
    noise_generator.get_noise_network(noise_type=noise, a=standard_deviation)
    noise_generator.generate_noise_image()
