import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=10):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 입력 크기 맞춤
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
