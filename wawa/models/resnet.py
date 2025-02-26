import torch.nn as nn
from torchvision.models import resnet50

def get_model():
    model = resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 10)  # MNIST는 10개의 클래스
    return model
