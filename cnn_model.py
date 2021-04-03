import torch.nn as nn
from pytorchcv.model_provider import get_model
from torchvision import models


class CNNModel(nn.Module):
    def __init__(self, model_name):
        super(CNNModel, self).__init__()
        # Load pretrained network as backbone
        pretrained = get_model(model_name, pretrained=True)
        # remove last layer of fc
        self.backbone = pretrained.features
        self.output = pretrained.output
        # self.dropout1 = nn.Dropout(0.4)
        self.classifier = nn.Linear(1000, 7)

        del pretrained

    def forward(self, x):
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1)
        x = self.output(x)
        x = self.classifier(x)
        return x

    def freeze_backbone(self):
        """Freeze the backbone network weight"""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Freeze the backbone network weight"""
        for p in self.backbone.parameters():
            p.requires_grad = True


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class CnnModel:
    def __init__(self, model_name, num_classes=7, feature_extract=None, use_pretrained=True):
        self.model_ft = None
        self.input_size = 0
        if model_name == "resnet18":
            """ Resnet18
            """
            self.model_ft = models.resnet18(pretrained=use_pretrained)
            set_parameter_requires_grad(self.model_ft, feature_extract)
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, num_classes)
            self.input_size = 224

        elif model_name == "resnet50":
            """
            Resnet50
            """
            self.model_ft = models.resnet50(pretrained=use_pretrained)
            set_parameter_requires_grad(self.model_ft, feature_extract)
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, num_classes)
            self.input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            self.model_ft = models.alexnet(pretrained=use_pretrained)
            set_parameter_requires_grad(self.model_ft, feature_extract)
            num_ftrs = self.model_ft.classifier[6].in_features
            self.model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            self.input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            self.input_size = 224

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
            self.input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            self.input_size = 224

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            self.input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

    def forward(self):
        return self.model_ft
