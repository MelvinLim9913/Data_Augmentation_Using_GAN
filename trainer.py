import pandas
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import glob
import os
import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# Self made modules
import utils
from emotiondataset import EmotionDataset
from cnn_model import CNNModel

logger = logging.getLogger("cnn." + __name__)


class Classifier:

    def __init__(self):
        # Child logger
        self.logger = logging.getLogger("cnn." + __name__)
        self.weight_dir = None
        self.graph_dir = None
        self.train_path = None
        self.valid_path = None
        self.test_path = None
        self.train_ds = None
        self.valid_ds = None
        self.test_ds = None
        self.dataset = dataset
        self.configs = utils.initialise_configs_file()
        self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        self.get_data_file_path()
        # Model
        self.__cnn_model_type = utils.get_classifier_model_type(self.configs)
        self.logger.info(f"Model used for training: {self.__cnn_model_type}")
        self.weight_dir = f"weights/{self.__cnn_model_type}/{dataset}/"
        os.makedirs(self.weight_dir, exist_ok=True)

    def get_data_file_path(self):
        image_file_path = self.configs.get("image_file_path", "")
        self.train_path = image_file_path[self.dataset]["train"]
        self.valid_path = image_file_path[self.dataset]["valid"]
        self.test_path = image_file_path[self.dataset]["test"]

    def read_train_data_from_path(self):
        train_img_list = list()
        train_label_list = list()
        for train in self.train_path:
            for class_number in range(7):
                new_images = glob.glob(os.path.join(train, str(class_number), '*.png'))
                train_img_list.extend(new_images)
                for _ in range(len(new_images)):
                    train_label_list.append(class_number)
        return train_img_list, train_label_list

    def load_and_transform_data(self):
        train_img, train_label = self.read_train_data_from_path()
        valid_img, valid_label = utils.read_valid_or_test_data(self.valid_path)
        test_img, test_label = utils.read_valid_or_test_data(self.test_path)
        img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.train_ds = EmotionDataset(
            train_img,
            train_label,
            transform=img_transform
        )
        self.valid_ds = EmotionDataset(
            valid_img,
            valid_label,
            transform=img_transform
        )
        self.test_ds = EmotionDataset(
            test_img,
            test_label,
            transform=img_transform
        )

    def initialise_excel_workbook(self):
        with pandas.ExcelWriter(f'{self.__cnn_model_type}_{self.dataset}_Classification_Report.xlsx', engine='openpyxl',
                                mode='w') as writer:
            book = writer.book
            book.create_sheet("Sheet1")
            writer.save()

    def train_model(self, model, optimizer, train_dl):
        running_loss = []
        running_corrects = []
        criterion = nn.CrossEntropyLoss().to(self.device)
        model.train()

        for img, label in tqdm.tqdm(train_dl):
            img = img.to(self.device)
            label = label.to(self.device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                logits = model(img)
                loss = criterion(logits, label)
                _, preds = torch.max(logits, 1)
                loss.backward()
                optimizer.step()

            running_loss.append(loss.item() * img.size(0))
            running_corrects.append(torch.sum(preds == label.data))

        ave_train_loss = sum(running_loss) / len(running_loss)
        ave_train_acc = float(sum(running_corrects)) / len(running_corrects)

        with open(f'{self.__cnn_model_type}_{dataset}_Log_File.txt', "a") as f:
            f.write(f'\tTrain Loss: {ave_train_loss:.3f} | Train Acc: {ave_train_acc:.2f}%\n')

        return ave_train_acc, ave_train_loss

    def validate_model(self, model, valid_dl):
        running_loss = 0
        total = 0
        running_corrects = 0
        criterion = nn.CrossEntropyLoss().to(self.device)
        model.eval()

        for img, label in tqdm.tqdm(valid_dl):
            img = img.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                logits = model(img)
                loss = criterion(logits, label)
                prediction = torch.max(logits.data, 1)

            running_loss += (loss.item() * img.size(0))
            running_corrects += torch.sum(prediction == label.data)
            total += label.size

        ave_val_loss = running_loss / len(valid_dl)
        ave_val_acc = (running_corrects / total) * 100

        with open(f'{self.__cnn_model_type}_{dataset}_Log_File.txt', "a") as f:
            f.write(f'\t Val. Loss: {ave_val_loss:.3f}   |  Val. Acc: {ave_val_acc:.2f}%\n')

        return ave_val_acc, ave_val_loss

    def test_model(self, model, test_dl):
        model.eval()
        ground_truths = []
        predictions = []

        for img, label in tqdm.tqdm(test_dl):
            img = img.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                logits = model(img)
                prediction = torch.argmax(logits, 1)
                predictions.extend(prediction.tolist())
                ground_truths.extend(label.tolist())
        ave_test_accuracy = utils.calculate_accuracy(predictions, ground_truths)

        with open(f'{self.__cnn_model_type}_{dataset}_Log_File.txt', "a") as f:
            f.write(f'\t Test Accuracy: {ave_test_accuracy:.3f}\n')

        return ave_test_accuracy, ground_truths, predictions

    def write_model_type_used_and_dataset_used_to_text(self):
        with open(f'{self.__cnn_model_type}_{dataset}_Log_File.txt', "w") as f:
            f.write(f"Classifier: {self.__cnn_model_type}\n")
            f.write(f"Dataset used: {args.dataset}\n\n")

    def create_dataloader(self):
        parameter_details = utils.get_classifier_train_or_valid_params_by_type(self.configs, self.__cnn_model_type)
        train_params = parameter_details.get("train_params")
        valid_params = parameter_details.get("valid_params")
        test_params = parameter_details.get("test_params")
        train_dl = DataLoader(self.train_ds, **train_params, shuffle=True, drop_last=True, pin_memory=True)
        valid_dl = DataLoader(self.valid_ds, **valid_params, pin_memory=True)
        test_dl = DataLoader(self.test_ds, **test_params, pin_memory=True)
        return train_dl, valid_dl, test_dl

    def train_with_backbone_freeze(self, num_epoch, train_dl, valid_dl, simulation_idx):
        baseline_model = CNNModel(self.__cnn_model_type)
        baseline_model.freeze_backbone()
        baseline_model.to(self.device)
        learning_rate = 1e-3
        optimizer = optim.Adam(baseline_model.parameters(), lr=learning_rate, weight_decay=1e-5)
        optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_dl), T_mult=num_epoch * len(train_dl))
        self.logger.info("Starting to train model with backbone freeze.")
        with open(f'{self.__cnn_model_type}_{dataset}_Log_File.txt', "a") as f:
            f.write("Model is FREEZE\n")
        for epoch in range(num_epoch):
            with open(f'{self.__cnn_model_type}_{dataset}_Log_File.txt', "a") as f:
                f.write(f"\nEPOCH {epoch + 1} of Cycle{simulation_idx + 1}\n")
            self.train_model(baseline_model, optimizer, train_dl)
            self.validate_model(baseline_model, valid_dl)
        return baseline_model

    def train_with_backbone_unfreeze(self, model, num_epoch, train_dl, valid_dl, simulation_idx):
        highest_acc = 0
        # model = CNNModel(self.__cnn_model_type)
        model.unfreeze_backbone()
        # model.to(self.device)
        learning_rate = 5e-4
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=2e-7)
        optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_dl), T_mult=len(train_dl) * num_epoch)
        self.logger.info(f"Unfreeze the backbone and train with {num_epoch} epoch.")
        with open(f'{self.__cnn_model_type}_{dataset}_Log_File.txt', "a") as f:
            f.write(f"Classifier: {self.__cnn_model_type}\n")
            f.write(f"Dataset used: {dataset}\n\n")
            f.write("Model is UNFREEZE\n")
        for epoch in range(num_epoch):
            with open(f'{self.__cnn_model_type}_{dataset}_Log_File.txt', "a") as f:
                f.write(f"\nEPOCH {epoch + 1} of Cycle{simulation_idx + 1}\n")
            self.logger.info(f"Cycle-{simulation_idx + 1}\tEPOCH--{epoch + 1}")
            ave_train_acc, ave_train_loss = self.train_model(model, optimizer, train_dl)
            ave_valid_acc, ave_valid_loss = self.validate_model(model, valid_dl)

            self.logger.info(f'\tTrain Loss: {ave_train_loss:.3f} | Train Acc: {ave_train_acc:.2f}%')
            self.logger.info(f'\t Val. Loss: {ave_valid_loss:.3f} |  Val. Acc: {ave_valid_acc:.2f}%')

            if ave_train_acc > highest_acc:
                highest_acc = ave_train_acc
                torch.save(model.state_dict(), os.path.join(
                    self.weight_dir, "weights_cycle{}.pth".format(simulation_idx + 1)))

    def test_model_performance(self, model, test_dl, simulation_idx):
        state_dict = torch.load(os.path.join(self.weight_dir, "weights_cycle{}.pth".format(simulation_idx + 1)))
        model.load_state_dict(state_dict)
        test_accuracy, ground_truth, prediction = self.test_model(model, test_dl)
        self.write_classification_report(ground_truth, prediction, simulation_idx)
        self.logger.info(f"Done training. Testing model performance now.")
        self.logger.info(f"\tTesting Accuracy: {test_accuracy}")

    def write_classification_report(self, ground_truths, predictions, simulation_idx):
        report = classification_report(ground_truths, predictions, output_dict=True)
        df = pandas.DataFrame(report).transpose()

        with pandas.ExcelWriter(
                f'{self.__cnn_model_type}_{dataset}_Classification_Report.xlsx',
                engine='openpyxl', mode='a') as writer:
            df.to_excel(writer, sheet_name='Cycle {}'.format(simulation_idx))
            writer.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default="original",
                        type=str,
                        choices=["original", "esrgan", "wgan"])
    args = parser.parse_args()
    dataset = args.dataset

    cnn_classifier = Classifier()
    cnn_classifier.load_and_transform_data()
    cnn_classifier.initialise_excel_workbook()
    cnn_classifier.write_model_type_used_and_dataset_used_to_text()

    train_dataloader, valid_dataloader, test_dataloader = cnn_classifier.create_dataloader()

    for i in range(5):
        train_model = cnn_classifier.train_with_backbone_freeze(
            num_epoch=3,
            train_dl=train_dataloader,
            valid_dl=valid_dataloader,
            simulation_idx=i
        )

        cnn_classifier.train_with_backbone_unfreeze(
            model=train_model,
            num_epoch=30,
            train_dl=train_dataloader,
            valid_dl=valid_dataloader,
            simulation_idx=i
        )

        cnn_classifier.test_model_performance(
            model=train_model,
            test_dl=test_dataloader,
            simulation_idx=i
        )
