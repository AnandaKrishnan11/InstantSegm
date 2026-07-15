import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights


class CustomSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.activation = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x, with_probs=True):
        input_shape = x.shape[-2:]
       
        x = self.backbone(x)["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        if with_probs:
            x = self.activation(x)
        return x
    

def create_model(num_class=2, in_channels=2):

    ori_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT, weights_backbone=ResNet50_Weights.IMAGENET1K_V1)
    
    model = CustomSegmentationModel(backbone=ori_model.backbone, classifier=ori_model.classifier, num_classes=num_class)
    print("----- Model: DeepLabV3 with ResNet50 backbone -----")
    
    model.classifier[-1] = nn.Conv2d(256, num_class, kernel_size=(1, 1), stride=(1, 1)) # this approach is mainly to benefit from pretrained models
    model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    del ori_model
    
    return model
