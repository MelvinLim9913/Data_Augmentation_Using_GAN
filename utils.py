import logging.handlers
import pathlib
import json
import glob
import os
import numpy as np

# Instantiate logger using module name and set level to INFO.
logger = logging.getLogger("cnn")
logger.setLevel(logging.INFO)
# Instantiate streamhandler (which outputs to sys.stderr by default) and set level to INFO.
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# Create file handler which logs error messages.
pathlib.Path("log").mkdir(parents=True, exist_ok=True)
pathlib.Path("log/gan.log").touch(exist_ok=True)
fh = logging.handlers.TimedRotatingFileHandler("log/gan.log", when="midnight", backupCount=8)
fh.setLevel(logging.INFO)
# Instantiate formatters and add it to handlers.
ch_formatter = logging.Formatter('%(process)d|%(name)s|%(funcName)s|%(levelname)s| %(message)s')
fh_formatter = logging.Formatter('%(asctime)s|%(process)d|%(name)s|%(funcName)s|%(levelname)s| %(message)s')
ch.setFormatter(ch_formatter)
fh.setFormatter(fh_formatter)
if logger.hasHandlers():
    logger.handlers.clear()
# Add handler to logger.
logger.addHandler(ch)
logger.addHandler(fh)
logger.propagate = False
# Use child logger.
logger = logging.getLogger("cnn." + __name__)


def get_classifier_model_type(configs):
    """
    Returns the enabled model_type in CnnClassifier.

    :param: None
    :return: model_type
    """
    models_config = ((configs.get("model_configs")).get("CnnClassifier")).get("models", "")
    tl_model_types = []
    for config in models_config:
        if not config.get("enabled"):
            continue

        model_type = config.get("type", "")
        if model_type in ["resnet50", "resnet101", "vgg16", "vgg19", "efficientnet-b2b"]:
            tl_model_types.append(model_type)

    if "resnet50" in tl_model_types:
        return "resnet50"

    if "resnet101" in tl_model_types:
        return "resnet101"

    if "vgg16" in tl_model_types:
        return "vgg16"

    if "vgg19" in tl_model_types:
        return "vgg19"

    if "efficientnet-b2b" in tl_model_types:
        return "efficientnet-b2b"

    return None


def get_classifier_train_or_valid_params_by_type(configs, model_type):
    """
    Get configuration for one model by type. Return empty dict if none are found.

    :param configs: configs file
    :param model_type: desired model type
    :return: dictionary containing configuration for one model in specific module and language
    """
    if model_type is None:
        return {}

    models_config = ((configs.get("model_configs")).get("CnnClassifier")).get("models", "")
    for config in models_config:
        if config.get("type", "") == model_type:
            return config

    return {}


def read_valid_or_test_data(data_path):
    img_list = list()
    label_list = list()
    for class_number in range(7):
        new_images = glob.glob(os.path.join(data_path, str(class_number), '*.png'))
        img_list.extend(new_images)
        for _ in range(len(new_images)):
            label_list.append(class_number)
    return img_list, label_list


def initialise_configs_file():
    with open('configs.json', 'r') as f:
        configs = json.load(f)
    return configs


def average(result):
    return sum(result) / len(result)


def calculate_accuracy(prediction, ground_truth):
    num_correct = (np.array(prediction) == np.array(ground_truth)).sum()
    return (num_correct / len(prediction)) * 100
