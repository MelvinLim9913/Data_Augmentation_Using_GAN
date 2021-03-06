import torch.nn as nn
from pytorchcv.model_provider import get_model


class CNNModel(nn.Module):
    def __init__(self, model_name):
        super(CNNModel, self).__init__()
        # Load pretrained network as backbone
        pretrained = get_model(model_name, pretrained=True)
        # remove last layer of fc
        self.backbone = pretrained.features
        # pretrained.output.fc3 = nn.Linear(4096, 7)
        self.output = pretrained.output
        self.classifier = nn.Linear(1000, 7)

        #        nn.init.zeros_(self.classifier.fc3.bias)
        #        nn.init.normal_(self.classifier.fc3.weight, mean=0.0, std=0.02)

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
